"""Full audit routing regressions: real entry/ownership, isolated fake upstreams."""
import asyncio
from contextlib import nullcontext
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from starlette.responses import JSONResponse

from src import compact_rescue, failover, log_db, model_metadata, scheduler, token_counter
from src.openai.channel.api_channel import OpenAIApiChannel
from src.protocols.runtime import AttemptResult
from src.tests.test_routing_review_fixes import setup


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_ingress_does_not_reject_discarded_foreign_thinking(monkeypatch, queued):
    import server
    setup(monkeypatch)
    ch = OpenAIApiChannel({"name": "audit-preflight", "baseUrl": "https://example.invalid",
                          "protocol": "openai-responses", "models": [{"alias": "audit-model", "real": "audit-model"}]})
    model_metadata.patch_override_fields("audit-model", scope_key=None, outbound_model=None,
        set_fields={"contextWindow": 4096, "maxInputTokens": 4096, "maxOutputTokens": 128})
    body = {"model": "audit-model", "max_tokens": 64, "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "z9X2Q7" * 10000},
                                           {"type": "text", "text": "Hello"}]},
        {"role": "user", "content": "continue"}]}
    assert token_counter.count_request_tokens(body, model="audit-model") > 4096
    route = scheduler.ScheduleResult([] if queued else [(ch, "audit-model")], None, False,
                                     saturated=[(ch, "audit-model")] if queued else [])
    monkeypatch.setattr(server.auth, "validate", lambda _headers: ("fixture-key", [], None))
    monkeypatch.setattr(server.scheduler, "schedule", lambda *a, **kw: route)
    seen = []

    async def forward(_route, forwarded, *a, **kw):
        wire = json.loads((await ch.build_upstream_request(forwarded, "audit-model", ingress_protocol="anthropic")).body)
        assert token_counter.count_request_tokens(wire, model="audit-model") < 100
        seen.append(wire)
        return JSONResponse({"type": "message", "role": "assistant", "content": [{"type": "text", "text": "ok"}],
                             "stop_reason": "end_turn", "usage": {"input_tokens": 11, "output_tokens": 1}})

    async def identity(body, **kw):
        return body

    monkeypatch.setattr(server.failover, "run_failover", forward)
    monkeypatch.setattr(server.translation, "translate_body", identity)

    class Request:
        headers = {"x-api-key": "fixture-key"}
        client = SimpleNamespace(host="127.0.0.1")
        async def body(self):
            return json.dumps(body).encode()

    response = await server.proxy_messages(Request())
    assert response.status_code == 200
    assert len(seen) == 1


@pytest.mark.parametrize("content_blocks", [False, True])
def test_compact_preserves_long_chinese_and_omits_explicit_image(monkeypatch, content_blocks):
    setup(monkeypatch)
    sentence = "用户要求保留全部历史消息并且不得删除原始业务数据任何修改必须经过完整验证以后才能交付使用。"
    text = sentence * 120
    content = [{"type": "text", "text": text}] if content_blocks else text
    image = "QUJD" * 2000
    body = {"model": "test-model", "messages": [
        {"role": "user", "content": content},
        {"role": "user", "content": [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": image}}]}]}
    assert not compact_rescue._is_probably_base64_blob(text)
    assert compact_rescue._is_probably_base64_blob(image)
    direct = compact_rescue.build_direct_summary_body(body, model="test-model", max_tokens=1000)
    segment = compact_rescue.build_segment_summary_body(body, body["messages"], segment_index=1, segment_count=1)
    for result in (direct, segment):
        prompt = result["messages"][0]["content"][0]["text"]
        assert text in prompt
        assert image not in prompt
        assert "image/base64 payload omitted" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("queued,retry_saturated", [(False, False), (True, False), (True, True)])
async def test_queued_candidate_uses_same_transient_retry_and_releases_slots(monkeypatch, queued, retry_saturated):
    _m, cfg = setup(monkeypatch)
    cfg["retry"] = {"transient": {"enabled": True, "maxExtraAttempts": 1, "backoffSeconds": [0]}}
    cfg["concurrency"] = {"enabled": True, "queueWaitSeconds": 1}
    ch = OpenAIApiChannel({"name": "audit-queue", "baseUrl": "https://example.invalid",
                          "protocol": "openai-responses", "models": [{"alias": "test-model", "real": "test-model"}]})
    held, attempts, releases, queue_calls = [], [], [], []
    contended = False

    async def acquire(key):
        nonlocal contended
        if retry_saturated and len(attempts) == 1 and not contended:
            contended = True
            return False
        assert not held
        held.append(key)
        return True

    async def queue(candidates, timeout):
        assert not held
        queue_calls.append(candidates)
        held.append(candidates[0][0])
        return candidates[0]

    def release(key):
        assert held == [key]
        held.clear()
        releases.append(key)

    async def delay(*a, **kw):
        assert not held
        return 0

    async def attempt(*a, **kw):
        assert len(held) == 1
        attempts.append(True)
        if len(attempts) == 1:
            return AttemptResult(outcome="http_error", http_status=503,
                error_detail='HTTP 503: {"error":{"type":"server_error","code":"server_is_overloaded","message":"server overloaded"}}')
        return AttemptResult(outcome="success", success=True, response=JSONResponse({"ok": True}))

    monkeypatch.setattr(failover.concurrency, "try_acquire", acquire)
    monkeypatch.setattr(failover.concurrency, "acquire_from_candidates", queue)
    monkeypatch.setattr(failover.concurrency, "release", release)
    monkeypatch.setattr(failover, "_wait_for_overload_retry", delay)
    monkeypatch.setattr(failover, "_try_channel", attempt)
    rid = f"audit-queue-{queued}-{retry_saturated}"
    body = {"model": "test-model", "input": "hi"}
    log_db.insert_pending(rid, "127.0.0.1", "ws-key", "test-model", False, 1, 0, {}, body, ingress_protocol="responses")
    route = scheduler.ScheduleResult([] if queued else [(ch, "test-model")], None, False,
                                     saturated=[(ch, "test-model")] if queued else [])
    response = await failover.run_failover(route, body, rid, "ws-key", "127.0.0.1", False, time.time(), ingress_protocol="responses")
    assert response.status_code == 200
    assert len(attempts) == len(releases) == 2
    assert not held
    assert len(queue_calls) == int(queued) + int(retry_saturated)


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("failure,disabled", [("timeout", False), ("429", False), ("503", False),
                                            ("disk", False), ("invalid_grant", True)])
async def test_refresh_classification_is_not_changed_by_queue(monkeypatch, queued, failure, disabled):
    setup(monkeypatch)
    monkeypatch.delenv("PARROT_NO_REFRESH", raising=False)
    ch = SimpleNamespace(key="oauth:claude:fixture", type="oauth", provider="claude", protocol="anthropic",
                         account_key="claude:fixture", email="fixture@example.test")
    monkeypatch.setattr(failover, "_pick_non_direct_proxy_name", lambda *a: None)
    monkeypatch.setattr(failover, "_should_use_responses_upstream_ws", lambda *a, **kw: False)
    monkeypatch.setattr(failover.oauth_manager, "account_generation_guard", lambda *a: nullcontext(True))
    toggles, refreshes = [], []
    monkeypatch.setattr(failover.oauth_manager, "set_enabled", lambda *a, **kw: toggles.append((a, kw)))
    monkeypatch.setattr(failover.notifier, "notify_event", lambda *a, **kw: None)

    async def acquire(*a):
        return True
    async def queue(candidates, timeout):
        return candidates[0]
    async def attempt(*a, **kw):
        return AttemptResult(outcome="http_error", http_status=401, error_detail="HTTP 401: expired access token")
    async def refresh(*a, **kw):
        refreshes.append(True)
        if failure == "timeout":
            raise httpx.ReadTimeout("fixture timeout")
        if failure == "disk":
            raise OSError("fixture persistence unavailable")
        status = 400 if failure == "invalid_grant" else int(failure)
        request = httpx.Request("POST", "https://example.invalid/token")
        response = httpx.Response(status, request=request, json={"error": failure})
        raise httpx.HTTPStatusError("fixture refresh failure", request=request, response=response)

    monkeypatch.setattr(failover.concurrency, "try_acquire", acquire)
    monkeypatch.setattr(failover.concurrency, "acquire_from_candidates", queue)
    monkeypatch.setattr(failover.concurrency, "release", lambda *a: None)
    monkeypatch.setattr(failover, "_try_channel", attempt)
    monkeypatch.setattr(failover.oauth_manager, "force_refresh", refresh)
    rid = f"audit-refresh-{queued}-{failure}"
    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    log_db.insert_pending(rid, "127.0.0.1", "ws-key", "test-model", False, 1, 0, {}, body, ingress_protocol="anthropic")
    route = scheduler.ScheduleResult([] if queued else [(ch, "test-model")], None, False,
                                     saturated=[(ch, "test-model")] if queued else [])
    response = await failover.run_failover(route, body, rid, "ws-key", "127.0.0.1", False, time.time(), ingress_protocol="anthropic")
    assert response.status_code == 401
    assert len(refreshes) == 1
    assert bool(toggles) is disabled
    if disabled:
        assert toggles == [(("claude:fixture", False), {"reason": "auth_error"})]


@pytest.mark.parametrize("envelope", ["data", "data.response"])
def test_failover_ws_counts_supported_nested_usage(envelope):
    usage = {"input_tokens": 100, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 20}}
    evt = {"type": "response.completed", "response": {"id": "fixture", "status": "completed", "output": []},
           "data": {"usage": usage} if envelope == "data" else {"response": {"usage": usage}}}
    tracker = failover._WsResponsesTracker()
    tracker.feed_text(json.dumps(evt))
    assert tracker.response_completed and tracker.usage_observed
    assert tracker.usage == {"input_tokens": 80, "output_tokens": 10, "cache_creation": 0, "cache_read": 20}
