"""Pre-commit transport keepalive: failover, errors and cancellation ownership."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from starlette.responses import JSONResponse, StreamingResponse

from src.transports import precommit_sse
from src.tests import test_protocol_fake_upstreams as fake

_import_modules = fake._import_modules


def event(name, **data):
    return fake._responses_sse_event(name, {"type": name, **data})


@pytest.mark.asyncio
async def test_fast_error_preserves_http_status_headers_and_body():
    expected = JSONResponse({"error": {"message": "invalid"}}, status_code=400, headers={"x-request-id": "original"})
    async def operation():
        return expected
    assert await precommit_sse.run_with_keepalive(operation) is expected


@pytest.mark.asyncio
async def test_keepalive_closed_before_first_next_cancels_running_operation():
    closed = asyncio.Event()
    async def operation():
        precommit_sse.notify_buffered_response_start()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    response = await precommit_sse.run_with_keepalive(operation)
    await response.body_iterator.aclose()
    assert closed.is_set()
    assert response._owned_iterator.owner.task.cancelled()


@pytest.mark.asyncio
async def test_resolved_but_unconsumed_stream_aborts_and_releases_once():
    from src import failover
    from src.protocols.runtime import AttemptResult
    release = asyncio.Event()
    counts = {"abort": 0, "release": 0}
    async def body():
        yield b"unconsumed"
    inner = StreamingResponse(body(), media_type="text/event-stream")
    async def abort():
        counts["abort"] += 1
    def release_slot():
        counts["release"] += 1
    result = failover._set_pending_stream_owner(AttemptResult(outcome="success", response=inner), abort)
    failover._attach_release_to_response(inner, release_slot)
    failover._transfer_pending_stream_result(result)
    async def operation():
        precommit_sse.notify_buffered_response_start()
        await release.wait()
        return inner
    response = await precommit_sse.run_with_keepalive(operation)
    release.set()
    await response._owned_iterator.owner.task
    await response.body_iterator.aclose()
    await response.body_iterator.aclose()
    assert counts == {"abort": 1, "release": 1}


@pytest.mark.asyncio
async def test_asgi_header_send_failure_cancels_waiting_operation():
    finished = asyncio.Event()
    async def operation():
        precommit_sse.notify_buffered_response_start()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
    response = await precommit_sse.run_with_keepalive(operation)
    async def receive():
        await asyncio.Event().wait()
    async def send(_message):
        raise OSError("client closed before first body byte")
    with pytest.raises(Exception):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert finished.is_set()
    assert response._owned_iterator.owner.task.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_backup", [False, True])
async def test_thinking_error_keeps_failover_and_never_exposes_failed_id(m, monkeypatch, has_backup):
    fake._setup(m)
    fake._install_keys(m, fake._default_key())
    monkeypatch.setattr(m["failover"], "_transient_retry_limit", lambda _cfg=None: 0)
    release = asyncio.Event()
    closed = asyncio.Event()
    class ThinkingThenError(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event("message_start", message={"id": "failed-upstream-id", "model": "claude-real", "usage": {"input_tokens": 3, "output_tokens": 1}})
            yield event("content_block_start", index=0, content_block={"type": "thinking", "thinking": ""})
            await release.wait()
            yield event("error", error={"type": "api_error", "message": "fixture thinking failure"})
        async def aclose(self):
            closed.set()
    router = fake.MockRouter()
    router.register("https://failed-thinking.example", lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ThinkingThenError()))
    channels = [fake._make_anthropic_channel(m, "failed-thinking", "https://failed-thinking.example", alias="sonnet", real="claude-real")]
    if has_backup:
        router.register("https://thinking-winner.example", lambda _request: fake._anthropic_sse_response("winner text"))
        channels.append(fake._make_anthropic_channel(m, "thinking-winner", "https://thinking-winner.example", alias="sonnet", real="claude-real"))
    fake._install_channels(m, channels)
    response, client = await fake._call_openai_handler(m, router, "responses", {"model": "sonnet", "stream": True, "input": "ping", "max_output_tokens": 32})
    try:
        assert response.status_code == 200
        assert await response.body_iterator.__anext__() == b": parrot keepalive\n\n"
        release.set()
        output = (await fake._consume_streaming_to_string(response)).encode()
        assert b"failed-upstream-id" not in output
        if has_backup:
            assert len(router.requests) == 2
            assert output.count(b"event: response.created\n") == 1
            assert b"winner text" in output
            assert b"response.completed" in output
        else:
            assert len(router.requests) == 1
            assert b"event: error\n" in output
            assert b"response.created" not in output
        assert closed.is_set()
        latest = m["log_db"]._get_conn().execute("SELECT status, final_channel_key FROM request_log ORDER BY id DESC LIMIT 1").fetchone()
        assert latest["status"] == ("success" if has_backup else "error")
        assert latest["final_channel_key"] == ("api:thinking-winner" if has_backup else "api:failed-thinking")
    finally:
        release.set()
        await response.body_iterator.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_cancel_during_empty_thinking_closes_upstream_and_settles_499(m):
    from src import concurrency
    fake._setup(m)
    fake._install_keys(m, fake._default_key())
    m["config"].update(lambda cfg: cfg.setdefault("concurrency", {}).__setitem__("enabled", True))
    closed = asyncio.Event()
    class HangingThinking(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event("message_start", message={"id": "thinking-cancel", "model": "claude-real", "usage": {"input_tokens": 3, "output_tokens": 1}})
            yield event("content_block_start", index=0, content_block={"type": "thinking", "thinking": ""})
            await asyncio.Event().wait()
        async def aclose(self):
            closed.set()
    router = fake.MockRouter()
    router.register("https://cancel-thinking.example", lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=HangingThinking()))
    fake._install_channels(m, [fake._make_anthropic_channel(m, "cancel-thinking", "https://cancel-thinking.example", alias="sonnet", real="claude-real")])
    response, client = await fake._call_openai_handler(m, router, "responses", {"model": "sonnet", "stream": True, "input": "ping", "max_output_tokens": 32})
    try:
        assert await response.body_iterator.__anext__() == b": parrot keepalive\n\n"
        await response.body_iterator.aclose()
        assert closed.is_set()
        latest = m["log_db"]._get_conn().execute("SELECT status, http_status FROM request_log ORDER BY id DESC LIMIT 1").fetchone()
        assert dict(latest) == {"status": "cancelled", "http_status": 499}
        assert m["cooldown"].get_state("api:cancel-thinking", "claude-real") is None
        assert all(row["in_flight"] == 0 for row in concurrency.snapshot())
    finally:
        await response.body_iterator.aclose()
        await client.aclose()
