"""W-01/02/03 + P-11: isolated WS lanes, complete history and terminal fidelity."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketState

from src.tests import test_openai_responses_ws as legacy
from src.tests.test_openai_responses_ws import _isolate_ws_config  # noqa: F401
from src.tests.test_sqlite_lock_root_fix import isolated_store  # noqa: F401
from src import failover
from src.openai import responses_ws as ws, store
from src.openai.responses_ws_runtime import map_ws_create_frame_for_upstream, request_body_from_ws_create


@pytest.fixture
def runtime(monkeypatch):
    modules = legacy._import_modules()
    cfg = legacy._setup(modules)
    cfg["retry"] = {"transient": {"enabled": False}}
    async def denied(*args, **kwargs):
        raise AssertionError("unexpected real upstream access")
    monkeypatch.setattr(ws, "_connect_upstream_ws", denied)
    monkeypatch.setattr(ws, "open_response_with_proxy_chain", denied)
    return modules


class Client:
    """Receiving blocks until client input/disconnect, not a response terminal."""
    def __init__(self):
        self.headers = legacy.FakeHeaders({"Authorization": "Bearer sk-ws"})
        self.client = SimpleNamespace(host="1.2.3.4")
        self.application_state = None
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.events = []
        self.close_calls = []
        self.readers = 0

    async def accept(self):
        self.application_state = WebSocketState.CONNECTED

    async def receive(self):
        self.readers += 1
        assert self.readers == 1
        try:
            return await self.incoming.get()
        finally:
            self.readers -= 1

    async def send_text(self, text):
        obj = json.loads(text)
        self.events.append(obj)
        await self.outgoing.put(obj)

    async def close(self, code=1000, reason=""):
        self.close_calls.append((code, reason))
        self.application_state = WebSocketState.DISCONNECTED
        self.disconnect()

    def send(self, frame):
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(frame)})

    def disconnect(self):
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    async def until(self, predicate):
        async with asyncio.timeout(3):
            while True:
                event = await self.outgoing.get()
                if predicate(event):
                    return event


def create(stream_id=None, text="hello", **extra):
    frame = {"type": "response.create", "model": "test-model", "input": text, **extra}
    if stream_id is not None:
        frame["stream_id"] = stream_id
    return frame


def message(name, text=None):
    return {"id": "msg_" + name, "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text or name, "annotations": []}]}


def terminal(name, *, reason=None):
    response = {"id": "resp_" + name, "status": "incomplete" if reason else "completed",
                "output": [message(name)], "usage": {"input_tokens": 5, "output_tokens": 2}}
    if reason:
        response["incomplete_details"] = {"reason": reason}
    return {"type": "response.incomplete" if reason else "response.completed", "response": response}


class Upstream:
    def __init__(self, dispatched):
        self.dispatched = dispatched
        self.sent = []
        self.events = asyncio.Queue()
        self.response = SimpleNamespace(headers={})
        self.closed = False

    async def send(self, data, text=None):
        frame = json.loads(data)
        self.sent.append(frame)
        if frame.get("type") == "response.create":
            self.dispatched.put_nowait((self, frame))

    async def recv(self):
        return json.dumps(await self.events.get())

    async def close(self, *args, **kwargs):
        self.closed = True

    def feed(self, *events):
        for event in events:
            self.events.put_nowait(event)


async def install_native(monkeypatch, runtime):
    legacy._make_channel(runtime)
    dispatched = asyncio.Queue()
    connections = []
    async def connect(*args, **kwargs):
        upstream = Upstream(dispatched)
        connections.append(upstream)
        return upstream
    monkeypatch.setattr(ws, "_connect_upstream_ws", connect)
    return dispatched, connections


async def next_dispatch(queue):
    return await asyncio.wait_for(queue.get(), 3)


async def finish(client, task):
    client.disconnect()
    await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
async def test_named_lanes_parallel_same_lane_fifo_and_default(monkeypatch, runtime):
    dispatched, connections = await install_native(monkeypatch, runtime)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create("planner", "p1"))
        first, first_frame = await next_dispatch(dispatched)
        client.send(create("planner", "p2"))
        client.send(create("research", "r1"))
        client.send(create(None, "d1"))
        other = [await next_dispatch(dispatched), await next_dispatch(dispatched)]
        assert {frame["input"] for _, frame in other} == {"r1", "d1"}
        assert dispatched.empty() and len(first.sent) == 1
        assert all("stream_id" not in frame for _, frame in [(first, first_frame), *other])
        for upstream, frame in other:
            name = frame["input"]
            upstream.feed({"type": "response.output_text.delta", "delta": name}, terminal(name))
        first.feed({"type": "response.output_text.delta", "delta": "p1"}, terminal("p1"))
        second, second_frame = await next_dispatch(dispatched)
        assert second is first and second_frame["input"] == "p2"
        second.feed(terminal("p2"))
        while len([e for e in client.events if e["type"] == "response.completed"]) < 4:
            await client.until(lambda e: e["type"] == "response.completed")
        completions = [e for e in client.events if e["type"] == "response.completed"]
        assert {(e.get("stream_id"), e["response"]["id"]) for e in completions} == {
            ("planner", "resp_p1"), ("planner", "resp_p2"), ("research", "resp_r1"), (None, "resp_d1")}
        assert not any(e["type"] == "error" for e in client.events)
    finally:
        await finish(client, task)
    assert all(up.closed for up in connections)


@pytest.mark.asyncio
async def test_oauth_named_lanes_do_not_share_thread_lock(monkeypatch, runtime):
    dispatched, connections = await install_native(monkeypatch, runtime)
    legacy._make_oauth_channel_for_failover(runtime)
    client = Client()
    client.headers = legacy.FakeHeaders({"Authorization": "Bearer sk-ws", "session-id": "shared-client-session"})
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        for name in ("a", "b"):
            client.send(create(name, name, client_metadata={"session_id": "shared-client-session"}))
        first, first_frame = await next_dispatch(dispatched)
        second, second_frame = await next_dispatch(dispatched)
        assert first is not second
        assert first_frame["prompt_cache_key"] != second_frame["prompt_cache_key"]
        assert all(frame["prompt_cache_key"] != "shared-client-session" for frame in (first_frame, second_frame))
        first.feed(terminal("oauth_a"))
        second.feed(terminal("oauth_b"))
        await client.until(lambda e: e["type"] == "response.completed")
        await client.until(lambda e: e["type"] == "response.completed")
    finally:
        await finish(client, task)
    from src.openai.codex_identity import active_thread_turn_queue_count
    assert active_thread_turn_queue_count() == 0 and all(up.closed for up in connections)


@pytest.mark.asyncio
async def test_bad_named_request_does_not_close_others_or_inherit_lineage(monkeypatch, runtime):
    dispatched, _ = await install_native(monkeypatch, runtime)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create("good", "first"))
        upstream, _ = await next_dispatch(dispatched)
        client.send({"type": "response.create", "stream_id": "bad", "input": "missing model"})
        error = await client.until(lambda e: e["type"] == "error")
        assert error["stream_id"] == "bad" and not client.close_calls
        upstream.feed(terminal("first"))
        await client.until(lambda e: e["type"] == "response.completed")
        client.send(create("good", "fresh"))
        same, frame = await next_dispatch(dispatched)
        assert same is upstream and frame["input"] == "fresh" and "previous_response_id" not in frame
        same.feed(terminal("fresh"))
        client.send(create("bad", "retry", previous_response_id="opaque_or_other_connection"))
        recovered, frame = await next_dispatch(dispatched)
        assert frame["previous_response_id"] == "opaque_or_other_connection"
        recovered.feed({"type": "error", "status": 400, "error": {"code": "previous_response_not_found", "message": "missing parent"}})
        error = await client.until(lambda e: e.get("error", {}).get("code") == "previous_response_not_found")
        assert error["stream_id"] == "bad"
        assert not any(e.get("stream_id") == "bad" and e["type"] == "response.completed" for e in client.events)
    finally:
        await finish(client, task)


@pytest.mark.asyncio
async def test_default_lane_overlapping_creates_are_fifo(monkeypatch, runtime):
    dispatched, _ = await install_native(monkeypatch, runtime)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create(text="one"))
        upstream, _ = await next_dispatch(dispatched)
        client.send(create(text="two"))
        upstream.feed(terminal("one"))
        same, frame = await next_dispatch(dispatched)
        assert same is upstream and frame["input"] == "two"
        upstream.feed(terminal("two"))
        await client.until(lambda e: e.get("response", {}).get("id") == "resp_two")
        assert all("stream_id" not in e for e in client.events)
    finally:
        await finish(client, task)


@pytest.mark.asyncio
async def test_fork_replays_complete_history_but_same_lane_uses_native_parent(monkeypatch, runtime):
    dispatched, _ = await install_native(monkeypatch, runtime)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create("main", "root", instructions="not inherited"))
        upstream, _ = await next_dispatch(dispatched)
        upstream.feed(terminal("root"))
        await client.until(lambda e: e["type"] == "response.completed")
        client.send(create("fork", "branch", previous_response_id="resp_root"))
        fork, frame = await next_dispatch(dispatched)
        assert fork is not upstream and "previous_response_id" not in frame
        assert frame["input"] == [{"role": "user", "content": "root"}, message("root"), {"role": "user", "content": "branch"}]
        assert "instructions" not in frame
        client.send(create("main", "continue", previous_response_id="resp_root"))
        same, frame = await next_dispatch(dispatched)
        assert same is upstream and frame["previous_response_id"] == "resp_root"
        assert frame["input"] == "continue"
        upstream.feed(terminal("continue"))
        fork.feed(terminal("branch"))
        seen = set()
        while len(seen) < 2:
            event = await client.until(lambda e: e["type"] == "response.completed")
            seen.add(event["response"]["id"])
    finally:
        await finish(client, task)


@pytest.mark.asyncio
async def test_cancel_targets_one_lane_and_disconnect_cleans_all(monkeypatch, runtime):
    dispatched, connections = await install_native(monkeypatch, runtime)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create("a", "a1"))
        a, _ = await next_dispatch(dispatched)
        a.feed({"type": "response.created", "response": {"id": "resp_a1"}},
               {"type": "response.output_text.delta", "delta": "partial a"})
        client.send(create("b", "b1"))
        b, _ = await next_dispatch(dispatched)
        await client.until(lambda e: e.get("delta") == "partial a")
        client.send(create("a", "a2"))
        client.send({"type": "response.cancel", "response_id": "resp_a1"})
        cancelled = await client.until(lambda e: e.get("error", {}).get("code") == "response_cancelled")
        assert cancelled["stream_id"] == "a" and a.closed and not b.closed
        a2, frame = await next_dispatch(dispatched)
        assert a2 is not a and frame["input"] == "a2"
        b.feed(terminal("b1"))
        await client.until(lambda e: e["type"] == "response.completed")
        assert not client.close_calls
    finally:
        await finish(client, task)
    assert all(up.closed for up in connections)
    logs = [dict(row) for row in runtime["log_db"]._get_conn().execute("SELECT * FROM request_log ORDER BY id")]
    assert sorted(row["status"] for row in logs[-3:]) == ["cancelled", "cancelled", "success"]


@pytest.mark.asyncio
async def test_http_bridge_named_parallel_fifo_continuation_and_cancel(monkeypatch, runtime):
    legacy._make_channel(runtime, extra={"responsesWsUpstreamTransport": "sse"})
    requests = asyncio.Queue()
    closed = []
    async def fake_open(**kwargs):
        body = json.loads(kwargs["upstream_req"].body)
        events = asyncio.Queue()
        requests.put_nowait((body, events))
        class Response:
            status_code = 200
            headers = {"content-type": "text/event-stream"}
            async def aiter_bytes(self):
                while True:
                    event = await events.get()
                    yield ("data: " + json.dumps(event) + "\n\n").encode()
        class Context:
            async def __aexit__(self, *args):
                closed.append(body)
        return SimpleNamespace(error=None, response=Response(), connect_ms=1, timing=None,
            proxy_name=None, proxy_bytes={"up": 1, "down": 1}, proxy_client=None,
            proxy_attempt_id=None, round_timeouts=None, ctx=Context())
    monkeypatch.setattr(ws, "open_response_with_proxy_chain", fake_open)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create("http", "first"))
        first, events = await next_dispatch(requests)
        assert "stream_id" not in first
        client.send(create("http", "next", previous_response_id="resp_first"))
        client.send(create("other", "other"))
        other, _ = await next_dispatch(requests)
        assert other["input"] == "other"
        events.put_nowait(terminal("first"))
        second, events2 = await next_dispatch(requests)
        assert "stream_id" not in second and "previous_response_id" not in second
        assert second["input"] == [{"role": "user", "content": "first"}, message("first"), {"role": "user", "content": "next"}]
        events2.put_nowait(terminal("next", reason="content_filter"))
        final = await client.until(lambda e: e["type"] == "response.incomplete")
        assert final["stream_id"] == "http" and final["response"]["usage"]["output_tokens"] == 2
        client.send({"type": "response.cancel", "stream_id": "other"})
        await client.until(lambda e: e.get("error", {}).get("code") == "response_cancelled")
        assert len(closed) == 3
    finally:
        await finish(client, task)


@pytest.mark.asyncio
async def test_mux_capacity_and_named_stream_validation(monkeypatch, runtime):
    active = 0
    peak = 0
    started = asyncio.Queue()
    gates = {}
    async def run_lane(lane):
        nonlocal active, peak
        frame = json.loads((await lane.receive_create())["text"])
        name = frame["input"]
        active += 1
        peak = max(peak, active)
        started.put_nowait(name)
        try:
            await gates.setdefault(name, asyncio.Event()).wait()
            await lane.send_text(json.dumps(terminal(name)))
        finally:
            active -= 1
    monkeypatch.setattr(ws, "_handle_responses_ws_lane", run_lane)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create("", "invalid"))
        error = await client.until(lambda e: e["type"] == "error")
        assert error["error"]["code"] == "invalid_stream_id"
        for i in range(33):
            client.send(create(f"lane{i}", str(i)))
        for _ in range(16):
            await next_dispatch(started)
        error = await client.until(lambda e: e.get("error", {}).get("code") == "websocket_stream_limit_reached")
        assert error["stream_id"] == "lane32" and active == peak == 16
        gates["0"].set()
        assert await next_dispatch(started) == "16"
        assert peak == 16
    finally:
        await finish(client, task)
    assert active == 0


@pytest.mark.asyncio
async def test_local_cache_error_eviction_respects_parent_owner(runtime):
    connection = ws._ResponsesWsConnection(Client())
    source = ws._ResponsesWsLane(connection, "source")
    fork = ws._ResponsesWsLane(connection, "fork")
    connection.lanes = {"source": source, "fork": fork}
    source.epoch = fork.epoch = 1
    source.creates.put_nowait(create("source", "original"))
    await source.receive_create()
    await source.send_text(json.dumps(terminal("root")))
    source.release_slot()
    fork.creates.put_nowait(create("fork", "branch", previous_response_id="resp_root"))
    frame = json.loads((await fork.receive_create())["text"])
    assert "previous_response_id" not in frame and len(frame["input"]) == 3
    failure = {"type": "error", "status": 400, "error": {"code": "invalid_request_error"}}
    await fork.send_text(json.dumps(failure))
    fork.release_slot()
    assert "resp_root" in connection.history
    source.creates.put_nowait(create("source", "continue", previous_response_id="resp_root"))
    await source.receive_create()
    await source.send_text(json.dumps(failure))
    source.release_slot()
    assert "resp_root" not in connection.history


@pytest.mark.asyncio
async def test_terminal_race_never_discards_a_completed_receive(monkeypatch, runtime):
    # Complete the next receive while the mux joins its activity waiter, after
    # asyncio.wait took its done snapshot. A reader.done() recheck is essential.
    client = Client()
    connection = ws._ResponsesWsConnection(client)
    first = asyncio.Event()
    original_wait = asyncio.wait
    async def racing_wait(tasks, **kwargs):
        done, pending = await original_wait(tasks, **kwargs)
        if not first.is_set() and pending and connection.lanes:
            first.set()
            client.send(create("lane", "second"))
            await asyncio.sleep(0)
        return done, pending
    seen = []
    async def lane_session(lane):
        frame = json.loads((await lane.receive_create())["text"])
        seen.append(frame["input"])
        await lane.send_text(json.dumps(terminal(frame["input"])))
        if len(seen) == 2:
            client.disconnect()
    monkeypatch.setattr(ws.asyncio, "wait", racing_wait)
    monkeypatch.setattr(ws, "_handle_responses_ws_lane", lane_session)
    await client.accept()
    client.send(create("lane", "first"))
    await asyncio.wait_for(connection.run(), 3)
    assert seen == ["first", "second"]


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("reason", ["content_filter", "max_output_tokens", "max_tokens", "other"])
def test_incomplete_is_never_success_or_input_context_error(reason, normalize):
    tracker = ws._WsTracker(normalize_max_output_incomplete=normalize)
    for delta in ("par", "tial"):
        tracker.feed_text(json.dumps({"type": "response.output_text.delta", "output_index": 0,
                                      "content_index": 0, "delta": delta}))
    tracker.feed_text(json.dumps(terminal("partial", reason=reason)))
    assert tracker.response_text_parts == ["par", "tial"]
    assert tracker.response_incomplete and not tracker.response_completed and not tracker.response_failed
    assert tracker.stream_error_code is None
    assert tracker.usage_observed and tracker.usage["output_tokens"] == 2
    assert tracker.get_output_items() == [message("partial")]


@pytest.mark.parametrize("transport", ["ws", "sse"])
@pytest.mark.parametrize("reason", ["content_filter", "max_output_tokens"])
@pytest.mark.asyncio
async def test_incomplete_wire_and_accounting_are_preserved(monkeypatch, runtime, reason, transport):
    legacy._make_channel(runtime, extra={"responsesWsUpstreamTransport": transport})
    event = terminal("partial", reason=reason)
    if transport == "ws":
        async def connect(*args, **kwargs):
            return legacy.FakeUpstreamWebSocket([event])
        monkeypatch.setattr(ws, "_connect_upstream_ws", connect)
    else:
        class Response:
            status_code = 200
            headers = {"content-type": "text/event-stream"}
            async def aiter_bytes(self):
                yield ("data: " + json.dumps(event) + "\n\n").encode()
        class Context:
            async def __aexit__(self, *args):
                pass
        async def opened(**kwargs):
            return SimpleNamespace(error=None, response=Response(), connect_ms=1, timing=None,
                proxy_name=None, proxy_bytes={"up": 1, "down": 1}, proxy_client=None,
                proxy_attempt_id=None, round_timeouts=None, ctx=Context())
        monkeypatch.setattr(ws, "open_response_with_proxy_chain", opened)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create("partial"))
        result = await client.until(lambda e: e["type"] == "response.incomplete")
        assert result == {**event, "stream_id": "partial"}
        row = legacy._last_request_log(runtime)
        assert row["status"] == "error" and row["usage_observed"] == 1
        assert row["http_status"] == (101 if transport == "ws" else 200)
        attempts = legacy._attempt_usage(runtime, row["request_id"])
        assert len(attempts) == 1 and attempts[0]["outcome"] == "response_incomplete"
        assert attempts[0]["input_tokens"] == 5 and attempts[0]["output_tokens"] == 2
        assert "context_length_exceeded" not in json.dumps(client.events)
    finally:
        await finish(client, task)


def test_ws_stream_id_is_preserved_by_mapper_but_not_http_body():
    channel = SimpleNamespace(protocol="openai-responses", type="api", provider="")
    frame = create("planner")
    mapped = map_ws_create_frame_for_upstream(frame, "real-model", channel=channel)
    assert mapped["stream_id"] == "planner"
    assert "stream_id" not in request_body_from_ws_create(frame)
    assert frame["stream_id"] == "planner"


def sparse_tracker(tracker_factory):
    tracker = tracker_factory()
    msg = message("m", "stream")
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "Read", "arguments": '{"path":"a"}', "status": "completed"}
    for index, item in [(2, msg), (4, call), (9, call)]:
        tracker.feed_text(json.dumps({"type": "response.output_item.done", "output_index": index, "item": item}))
    return tracker, msg, call


@pytest.mark.parametrize("tracker_factory", [ws._WsTracker, failover._WsResponsesTracker])
@pytest.mark.parametrize("snapshot", ["empty", "partial", "full", "reordered", "call_id_only"])
def test_sparse_duplicate_and_partial_terminal_identity_merge(snapshot, tracker_factory):
    tracker, msg, call = sparse_tracker(tracker_factory)
    final_msg = message("m", "terminal wins")
    final_call = {**call, "arguments": "{}"}
    outputs = {"empty": [], "partial": [final_call], "full": [final_msg, final_call],
               "reordered": [final_call, final_msg], "call_id_only": [{k: v for k, v in final_call.items() if k != "id"}]}
    tracker.feed_text(json.dumps({"type": "response.completed", "response": {"output": outputs[snapshot]}}))
    result = tracker.get_output_items()
    assert len(result) == 2
    if snapshot == "empty":
        assert result == [msg, call]
    elif snapshot in {"partial", "call_id_only"}:
        assert result == [msg, final_call]
    else:
        # Even a reordered terminal cannot move already displayed stream items.
        assert result == [final_msg, final_call]
    assert tracker.get_output_items() == result


def test_store_51_rounds_includes_original_constraints(isolated_store):
    for i in range(51):
        store.save(f"r{i}", f"r{i-1}" if i else None, api_key_name="key-a", model="test",
                   channel_key="api:test", input_items=[{"role": "user", "content": f"constraint-{i}"}],
                   output_items=[message(str(i))])
    items = store.expand_history("r50", api_key_name="key-a")
    assert len(items) == 102 and items[0]["content"] == "constraint-0" and items[-1] == message("50")
    with pytest.raises(store.ResponseHistoryError, match="max_depth"):
        store.expand_history("r50", api_key_name="key-a", max_depth=50)


@pytest.mark.parametrize("fault", ["cycle", "missing", "expired", "owner"])
def test_store_real_chain_errors_are_not_partial_history(isolated_store, fault):
    for name, parent in [("a", None), ("b", "a")]:
        store.save(name, parent, api_key_name="key-a", model="test", channel_key="api:test",
                   input_items=[{"role": "user", "content": name}], output_items=[])
    conn = store._get_conn()
    sql, error = {
        "cycle": ("UPDATE openai_response_store SET parent_id='b' WHERE response_id='a'", store.ResponseHistoryError),
        "missing": ("DELETE FROM openai_response_store WHERE response_id='a'", store.ResponseNotFound),
        "expired": ("UPDATE openai_response_store SET expires_at=0 WHERE response_id='a'", store.ResponseExpired),
        "owner": ("UPDATE openai_response_store SET api_key_name='key-b' WHERE response_id='a'", store.ResponseForbidden),
    }[fault]
    conn.execute(sql)
    conn.commit()
    with pytest.raises(error):
        store.expand_history("b", api_key_name="key-a")
