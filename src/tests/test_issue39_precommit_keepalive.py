"""Issue #39: SSE transport keepalive must not commit a Claude attempt."""
import asyncio
import json
import time

import httpx
import pytest
from starlette.requests import Request

from src.tests import test_protocol_fake_upstreams as fake

_import_modules = fake._import_modules


def event(name, **data):
    return (f"event: {name}\ndata: " + json.dumps({"type": name, **data}) + "\n\n").encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("early_visible,terminal_kind", [(False, "text"), (False, "tool"), (True, "text")])
async def test_issue39_delayed_headers(m, monkeypatch, early_visible, terminal_kind):
    fake._setup(m)
    fake._install_keys(m, fake._default_key())
    from src.channel.oauth_channel import OAuthChannel
    from src import oauth_manager
    from src.transports import precommit_sse
    monkeypatch.setattr(precommit_sse, "KEEPALIVE_INTERVAL_SECONDS", 0.05)
    account = {
        "provider": "claude", "email": "issue39@example.test",
        "access_token": "fixture-access", "refresh_token": "fixture-refresh",
        "expires_at": 32503680000, "models": ["claude-opus-5-5"],
    }
    channel = OAuthChannel(account)
    fake._install_channels(m, [channel])

    async def token(_channel):
        return "fixture-access"

    monkeypatch.setattr(oauth_manager, "ensure_channel_token", token)
    release = asyncio.Event()
    progress_done = asyncio.Event()
    headers_sent = asyncio.Event()
    started = time.monotonic()
    timeline = {}
    upstream_frames = []
    downstream = []

    class DelayedClaudeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            timeline["upstream_first_frame_ms"] = (time.monotonic() - started) * 1000
            first = event("message_start", message={
                "id": "msg_issue39", "type": "message", "role": "assistant",
                "model": "claude-opus-5-5", "content": [], "stop_reason": None,
                "usage": {"input_tokens": 6, "output_tokens": 1},
            })
            upstream_frames.append("message_start")
            yield first
            if early_visible:
                upstream_frames.append("text_start")
                yield event("content_block_start", index=0, content_block={"type": "text", "text": ""})
            else:
                yield event("content_block_start", index=0, content_block={"type": "thinking", "thinking": ""})
                upstream_frames.append("empty_thinking_start")
                for number in range(12):
                    await asyncio.sleep(0.005)
                    upstream_frames.append("empty_thinking_delta")
                    yield event("content_block_delta", index=0, delta={
                        "type": "thinking_delta", "thinking": "", "estimated_tokens": 100 + number,
                    })
                    upstream_frames.append("ping")
                    yield event("ping")
                upstream_frames.append("signature")
                yield event("content_block_delta", index=0, delta={"type": "signature_delta", "signature": "fixture-signature"})
                yield event("content_block_stop", index=0)
            progress_done.set()
            await release.wait()
            timeline["text_released_ms"] = (time.monotonic() - started) * 1000
            text_index = 0 if early_visible else 1
            if terminal_kind == "tool":
                upstream_frames.append("tool_start")
                yield event("content_block_start", index=text_index, content_block={"type": "tool_use", "id": "call_issue39", "name": "check_status", "input": {}})
                yield event("content_block_stop", index=text_index)
                yield event("message_delta", delta={"stop_reason": "tool_use", "stop_sequence": None}, usage={"output_tokens": 5})
                yield event("message_stop")
                return
            if not early_visible:
                upstream_frames.append("text_start")
                yield event("content_block_start", index=text_index, content_block={"type": "text", "text": ""})
            yield event("content_block_delta", index=text_index, delta={"type": "text_delta", "text": "pong"})
            yield event("content_block_stop", index=text_index)
            yield event("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None}, usage={"output_tokens": 5})
            yield event("message_stop")

    def upstream(request):
        timeline["upstream_headers_ms"] = (time.monotonic() - started) * 1000
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=DelayedClaudeStream())

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), timeout=5)
    m["upstream"].set_client(client)
    body = json.dumps({
        "model": "claude-opus-5-5", "stream": True, "max_output_tokens": 32,
        "input": "ping", "reasoning": {"effort": "high"},
    }).encode()
    received = False
    disconnected = asyncio.Event()

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        downstream.append(message)
        if message["type"] == "http.response.start":
            timeline["downstream_headers_ms"] = (time.monotonic() - started) * 1000
            headers_sent.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/v1/responses", "raw_path": b"/v1/responses", "query_string": b"",
        "headers": [(b"authorization", b"Bearer ccp-test"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 22122),
    }

    async def application():
        request = Request(scope, receive)
        response = await m["openai_handler"].handle(request, ingress_protocol="responses")
        await response(scope, receive, send)

    task = asyncio.create_task(application())
    try:
        await asyncio.wait_for(progress_done.wait(), 5)
        try:
            await asyncio.wait_for(headers_sent.wait(), 0.3)
            timeline["header_wait_timeout"] = False
        except asyncio.TimeoutError:
            timeline["header_wait_timeout"] = True
        assert headers_sent.is_set()
        await asyncio.sleep(0.15)
        if not early_visible:
            waiting_bytes = b"".join(row.get("body", b"") for row in downstream)
            assert waiting_bytes.count(b": parrot keepalive") >= 2
            assert b"response.created" not in waiting_bytes
            assert b"response.output_item" not in waiting_bytes
        timeline["observed_wait_ms"] = (time.monotonic() - started) * 1000
        release.set()
        await asyncio.wait_for(task, 5)
        assert headers_sent.is_set()
        start_message = next(row for row in downstream if row["type"] == "http.response.start")
        assert start_message["status"] == 200
        content = b"".join(row.get("body", b"") for row in downstream if row["type"] == "http.response.body")
        assert b"response.created" in content and b"response.completed" in content
        if terminal_kind == "tool":
            assert b"response.function_call_arguments.done" in content
            assert b"check_status" in content
        else:
            assert b"response.output_text.delta" in content
        assert timeline["downstream_headers_ms"] < timeline["text_released_ms"]
        assert content.count(b"event: response.created\n") == 1
        print("ISSUE39_EVIDENCE " + json.dumps({
            "scenario": "text-start-control" if early_visible else "empty-thinking-then-" + terminal_kind,
            "upstream_frame_count": len(upstream_frames),
            "downstream_http_status": start_message["status"],
            **{k: round(v, 2) if isinstance(v, float) else v for k, v in timeline.items()},
        }))
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await client.aclose()
