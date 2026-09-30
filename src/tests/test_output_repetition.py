"""Output-loop detection, abort, protocol error and durable-cause regressions."""
from __future__ import annotations

import asyncio
import json
import random
import string
import time

import httpx
import pytest

from src.protocols.output_repetition import (
    OUTPUT_REPETITION_CODE, OutputRepetitionGuard, TextRepetitionDetector,
)
from src.tests import test_protocol_fake_upstreams as h
from src.tests import test_stream_terminal_ownership as t
from src.tests import test_openai_responses_ws as w


def _import_modules():
    return t._import_modules()


def _pieces(text, mode):
    if mode == "whole":
        yield text
        return
    rng = random.Random(19)
    offset = 0
    while offset < len(text):
        size = rng.randint(1, 83) if mode == "random" else int(mode)
        yield text[offset:offset + size]
        offset += size


@pytest.mark.parametrize("length", [2, 7, 31, 127, 128, 129, 255, 256, 298, 511, 512])
@pytest.mark.parametrize("mode", ["whole", "1", "7", "random"])
def test_unknown_random_patterns_and_chunk_invariance(length, mode):
    rng = random.Random(length * 79)
    pattern = "◆" + "".join(rng.choice(string.ascii_letters + "中文🙂") for _ in range(length - 1))
    prefix = "正常前置文本§"
    detector = TextRepetitionDetector()
    required = max(1024, 8 * length)
    text = prefix + pattern * ((required + length - 1) // length + 1)
    for piece in _pieces(text, mode):
        if detector.feed(piece):
            break
    assert detector.hit is not None
    assert detector.hit.rule == "exact_repetition"
    assert detector.hit.period_chars == length
    assert detector.hit.position == len(prefix) + required


@pytest.mark.parametrize("char", [" ", "e", "嗯", "🙂", "\n", "\t"])
def test_identical_character_threshold_is_100(char):
    detector = TextRepetitionDetector()
    assert detector.feed(char * 99) is None
    assert detector.feed(char) is not None
    assert detector.hit.rule == "identical_character"
    assert detector.hit.position == 100


def test_whitespace_and_negative_controls_and_bounded_memory():
    detector = TextRepetitionDetector()
    assert detector.feed(" \t\n" * 33) is None
    assert detector.feed(" ").rule == "continuous_whitespace"
    rng = random.Random(90)
    pattern = "◆" + "".join(rng.choice(string.ascii_letters) for _ in range(190))
    for text in (pattern * 7, pattern * 4 + "§" + pattern * 4,
                 "".join(pattern[:-1] + chr(0x4e00 + i) for i in range(20)),
                 json.dumps({"new_string": " " * 60})):
        assert TextRepetitionDetector().feed(text) is None
    detector = TextRepetitionDetector()
    assert detector.feed("".join(chr(0x4e00 + i) for i in range(6000))) is None
    assert len(detector._history) == 512
    assert len(detector._positions) == 512
    assert len(detector._matches) == 513


def test_metadata_empty_thinking_and_independent_blocks():
    guard = OutputRepetitionGuard()
    for i in range(200):
        assert guard.observe({"type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "", "estimated_tokens": i}}) is None
        assert guard.observe({"type": "content_block_delta", "index": 0,
            "delta": {"type": "signature_delta", "signature": "A" * 5000}}) is None
    for index in range(3):
        assert guard.observe({"type": "content_block_delta", "index": index,
            "delta": {"type": "text_delta", "text": " " * 60}}) is None
    assert guard.observe({"type": "content_block_delta", "index": 2,
        "delta": {"type": "text_delta", "text": " " * 40}}) is not None
    assert guard.hit.scope == ("anthropic", 2, "text")
    assert "threshold_chars=100" in guard.hit.message
    assert "run_chars=100" in guard.hit.message
    assert "scope=anthropic:2:text" in guard.hit.message


def _event(event):
    return h._responses_sse_event(event["type"], event)


def _frames(protocol, kind="text"):
    if protocol == "responses":
        typ = {"text": "response.output_text.delta", "thinking": "response.reasoning_text.delta",
               "tool": "response.function_call_arguments.delta"}[kind]
        start = _event({"type": "response.created", "response": {"id": "loop-test", "status": "in_progress"}})
        delta = _event({"type": typ, "output_index": 0, "content_index": 0, "delta": "e" * 50})
        terminal = _event({"type": "response.completed", "response": {"id": "loop-test",
            "status": "completed", "output": [], "usage": {"input_tokens": 7, "output_tokens": 100}}})
    elif protocol == "anthropic":
        start = _event({"type": "message_start", "message": {"id": "loop-test", "role": "assistant",
            "model": "test-model", "content": [], "usage": {"input_tokens": 7, "output_tokens": 0}}})
        start += _event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        delta = _event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "e" * 50}})
        terminal = _event({"type": "message_stop"})
    else:
        def chat(delta):
            return b"data: " + json.dumps({"id": "loop-test", "object": "chat.completion.chunk", "model": "test-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}).encode() + b"\n\n"
        start = chat({"role": "assistant"})
        delta = chat({"content": "e" * 50})
        terminal = b"data: [DONE]\n\n"
    return [start, delta, delta + terminal, b"data: NEVER_READ_OR_FORWARD\n\n"]


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


def _row(m):
    conn = m["log_db"]._get_conn()
    row = dict(conn.execute("SELECT * FROM request_log ORDER BY id DESC LIMIT 1").fetchone())
    row["attempts"] = [dict(r) for r in conn.execute(
        "SELECT outcome,error_detail,ended_at FROM retry_chain WHERE request_id=? ORDER BY id", (row["request_id"],))]
    return row


def _assert_logged_failure(m):
    row = _row(m)
    assert row["status"] == "error", row
    assert OUTPUT_REPETITION_CODE in row["error_message"], row
    assert "rule=identical_character" in row["error_message"]
    assert "position_chars=100" in row["error_message"]
    assert "threshold_chars=100" in row["error_message"]
    assert row["finished_at"] is not None
    assert row["attempts"] and all(r["outcome"] != "open" and r["ended_at"] for r in row["attempts"])
    assert OUTPUT_REPETITION_CODE in row["attempts"][-1]["error_detail"]
    return row


@pytest.mark.parametrize("protocol,ingress", [(p, i) for p in ("responses", "chat", "anthropic")
                                             for i in ("responses", "chat", "anthropic")])
async def test_http_abort_and_cause_persist_before_error_and_client_close(m, protocol, ingress):
    t._configure(m)
    base = "https://loop-test.example"
    ch = (h._make_anthropic_channel(m, "loop-test", base, alias="test-model", real="claude-real")
          if protocol == "anthropic" else h._make_openai_channel("loop-test", base,
              protocol="openai-" + protocol, alias="test-model", real="gpt-real"))
    h._install_channels(m, [ch])
    stream = TrackingStream(_frames(protocol))
    router = h.MockRouter()
    router.register(base, lambda req: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"}))
    body = {"model": "test-model", "stream": True, "max_tokens": 512,
            "messages": [{"role": "user", "content": "test"}]} if ingress != "responses" else {
                "model": "test-model", "stream": True, "input": "test"}
    if ingress == "anthropic":
        resp, client, _ = await h._call_anthropic_core(m, router, body)
    else:
        resp, client = await h._call_openai_handler(m, router, ingress, body)
    emitted = b""
    try:
        assert resp.status_code == 200
        async for chunk in resp.body_iterator:
            emitted += chunk.encode() if isinstance(chunk, str) else chunk
            if OUTPUT_REPETITION_CODE.encode() in emitted:
                assert stream.closed, "upstream must close before exposing error"
                _assert_logged_failure(m)
                break  # Simulate client immediately leaving after the error.
        assert OUTPUT_REPETITION_CODE.encode() in emitted
        assert b"NEVER_READ_OR_FORWARD" not in emitted
        assert b"response.completed" not in emitted
        assert b"message_stop" not in emitted
        assert b"[DONE]" not in emitted
        assert stream.reads == 3
        assert len(router.requests) == 1, "no retry after downstream output"
    finally:
        await resp.body_iterator.aclose()
        await client.aclose()
    _assert_logged_failure(m)


@pytest.mark.parametrize("protocol", ["anthropic", "chat", "responses"])
async def test_precommit_first_batch_is_rejected_even_with_completion(m, protocol):
    from src.transports.http_runtime import prepare_stream_response_start
    from src.tests.test_stream_as_non_stream_errors import _Ctx
    base = "https://loop-test.example"
    ch = (h._make_anthropic_channel(m, "loop-test", base, alias="test-model", real="claude-real")
          if protocol == "anthropic" else h._make_openai_channel("loop-test", base,
              protocol="openai-" + protocol, alias="test-model", real="gpt-real"))
    ctx = _Ctx()
    stream = TrackingStream([b"".join(_frames(protocol)[:3]), b"NEVER_READ"])
    response = httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    result = await prepare_stream_response_start(ctx, response, ch, dynamic_map=None, connect_ms=1,
        deadline_ts=time.time() + 10, first_byte_timeout=3, idle_timeout=3, ingress_protocol=protocol)
    assert result.error and OUTPUT_REPETITION_CODE in result.error.error_detail
    assert result.error.error_code == OUTPUT_REPETITION_CODE
    assert not result.first_downstream_chunks
    assert ctx.closed and stream.reads == 1


@pytest.mark.parametrize("kind", ["thinking", "tool"])
async def test_stream_only_aggregation_also_aborts_repeated_generated_fields(m, kind):
    from src.transports.http_runtime import aggregate_stream_as_non_stream_response
    from src.tests.test_stream_as_non_stream_errors import _Ctx, _Channel
    stream = TrackingStream(_frames("responses", kind))
    ctx = _Ctx()
    started = time.time()
    result = await aggregate_stream_as_non_stream_response(ctx,
        httpx.Response(200, stream=stream), _Channel(), "test-model", dynamic_map=None,
        connect_ms=1, start_time=started, deadline_ts=started + 10, total_timeout=10,
        first_byte_timeout=3, idle_timeout=3)
    assert result.error and OUTPUT_REPETITION_CODE in result.error.error_detail
    assert ("reasoning" if kind == "thinking" else "tool_arguments") in result.error.error_detail
    assert result.error.error_code == OUTPUT_REPETITION_CODE
    assert result.obj is None and ctx.closed and stream.reads == 3


@pytest.mark.parametrize("transport", ["ws", "sse"])
async def test_ws_ingress_emits_error_not_repetition_and_keeps_cause(m, monkeypatch, transport):
    t._configure(m, ws=True)
    base = "https://loop-ws.example"
    h._install_channels(m, [h._make_openai_channel("loop-ws", base, protocol="openai-responses",
        alias="test-model", real="gpt-real", extra={"responsesWsUpstreamTransport": transport})])
    frames = [{"type": "response.created", "response": {"id": "loop-ws"}},
              {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "e" * 50},
              {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "e" * 50},
              {"type": "response.completed", "response": {"id": "loop-ws", "status": "completed", "output": []}}]
    snapshots = []
    class Client(w.FakeWebSocket):
        async def send_text(self, text):
            if OUTPUT_REPETITION_CODE in text:
                snapshots.append(_assert_logged_failure(m))
                if stream is not None:
                    assert stream.closed
                if upstream_ws is not None:
                    assert upstream_ws.closed
                self._closed.set()  # Disconnect as soon as the explicit error arrives.
            await super().send_text(text)
    ws = Client({"type": "response.create", "model": "test-model", "input": "test", "stream": True})
    client = None
    upstream_ws = None
    stream = None
    if transport == "ws":
        upstream_ws = h.FakeOAuthResponseWs(frames)
        async def connect(*args, **kwargs):
            return upstream_ws
        monkeypatch.setattr(m["responses_ws"], "_connect_upstream_ws", connect)
    else:
        stream = TrackingStream(_frames("responses"))
        router = h.MockRouter()
        router.register(base, lambda req: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"}))
        client = httpx.AsyncClient(transport=httpx.MockTransport(router.handle))
        m["upstream"].set_client(client)
    try:
        await asyncio.wait_for(m["responses_ws"].handle_responses_ws(ws), 5)
        events = [json.loads(s) for s in ws.sent_texts]
        assert len(snapshots) == 1
        assert sum(e.get("type") == "response.output_text.delta" for e in events) == 1
        assert not any(e.get("type") == "response.completed" for e in events)
        error = next(e for e in events if e.get("type") == "error")
        assert error["error"]["code"] == OUTPUT_REPETITION_CODE
        _assert_logged_failure(m)
        if stream is not None:
            assert stream.closed and stream.reads == 3
        if upstream_ws is not None:
            assert upstream_ws.closed
    finally:
        if client is not None:
            await client.aclose()


@pytest.mark.parametrize("protocol", ["anthropic", "chat", "responses"])
def test_tracker_utf8_byte_boundaries_and_terminal_cannot_overwrite_cause(m, protocol):
    tracker_cls = {"anthropic": m["upstream"].SSEUsageTracker,
                   "chat": m["upstream"].ChatSSEUsageTracker,
                   "responses": m["upstream"].ResponsesSSEUsageTracker}[protocol]
    tracker = tracker_cls()
    payload = b"".join(_frames(protocol)[:3]).replace(b"e" * 50, ("嗯" * 50).encode())
    for byte in payload:
        tracker.feed(bytes([byte]))
    assert tracker.saw_stream_error
    assert tracker.stream_error_code == OUTPUT_REPETITION_CODE
    assert "position_chars=100" in tracker.stream_error_message
    assert not tracker.saw_stream_end
    tracker.feed(_frames(protocol)[2])
    assert tracker.stream_error_code == OUTPUT_REPETITION_CODE
    assert not tracker.saw_stream_end


async def test_native_oauth_ws_to_http_also_aborts_and_logs_before_error(m, monkeypatch):
    frames = [{"type": "response.created", "response": {"id": "loop-oauth"}},
              {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "e" * 50},
              {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "e" * 50},
              {"type": "response.completed", "response": {"id": "loop-oauth", "status": "completed", "output": []}}]
    captured = []
    original = h.FakeOAuthResponseWs
    def make_upstream(events):
        obj = original(events)
        captured.append(obj)
        return obj
    monkeypatch.setattr(h, "FakeOAuthResponseWs", make_upstream)
    monkeypatch.setattr(t, "_events", lambda error=False, terminal_only=False: frames)
    resp, client, _ = await t._http_response(m, monkeypatch, "oauth_ws")
    emitted = b""
    try:
        async for chunk in resp.body_iterator:
            emitted += chunk.encode() if isinstance(chunk, str) else chunk
            if OUTPUT_REPETITION_CODE.encode() in emitted:
                _assert_logged_failure(m)
                assert captured[0].closed
                break
        assert OUTPUT_REPETITION_CODE.encode() in emitted
        assert b"response.completed" not in emitted
        assert len(captured) == 1
    finally:
        await resp.body_iterator.aclose()
        await client.aclose()
    _assert_logged_failure(m)
