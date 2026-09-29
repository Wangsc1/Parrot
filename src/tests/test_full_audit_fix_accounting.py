"""AO-04/05/07/08 correct-behavior regressions. All traffic/state is synthetic."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from src import cache_display, config, model_pricing, probe
from src.anthropic import rate_limit_headers as headers
from src.protocols.usage import select_openai_chat_usage, select_openai_responses_usage
from src.upstream import ResponsesSSEUsageTracker


def _responses_usage(inp=100, out=10, cached=20):
    return {"input_tokens": inp, "output_tokens": out,
            "input_tokens_details": {"cached_tokens": cached}}


def _sse(*events):
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events)


@pytest.mark.parametrize("cached", [0, 20, 50, 80, 100])
def test_accounting_modern_cache_rows_preserve_normalized_counters(cached):
    tracker = ResponsesSSEUsageTracker()
    event = {"type": "response.completed", "response": {"usage": _responses_usage(cached=cached)}}
    tracker.feed(("event: response.completed\n" + _sse(event)).encode())
    assert tracker.usage_observed
    row = {"upstream_protocol": "openai-responses", "usage_observed": True,
           "input_tokens": tracker.usage["input_tokens"],
           "cache_read_tokens": tracker.usage["cache_read"], "cache_creation_tokens": 0}
    assert cache_display.prompt_total_from_row(row) == 100
    assert f"({cached:.1f}%)" in cache_display.cache_read_phrase_from_row(row)


def test_accounting_cache_display_preserves_write_and_aggregate_semantics():
    assert cache_display.prompt_total_from_row({"input_tokens": 3, "cache_creation_tokens": 7, "cache_read_tokens": 10}) == 20
    assert cache_display.prompt_total_from_row({"total_prompt_tokens": 0, "total_input_tokens": 99}, aggregate=True) == 0
    assert cache_display.prompt_total_from_row({"total_input_tokens": 3, "total_cache_creation": 7, "total_cache_read": 10}, aggregate=True) == 20


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("missing", ["input", "output"])
@pytest.mark.parametrize("location", ["response", "data", "data.response"])
def test_accounting_partial_candidate_falls_through_to_complete_envelope(chat, missing, location):
    if chat:
        full = {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 20}}
        partial = {"completion_tokens" if missing == "input" else "prompt_tokens": 0}
        select = select_openai_chat_usage
    else:
        full = _responses_usage()
        partial = {"output_tokens" if missing == "input" else "input_tokens": 0}
        select = select_openai_responses_usage
    obj = {"usage": partial}
    container = obj
    for name in location.split("."):
        container = container.setdefault(name, {})
    container["usage"] = full
    strict = select(obj)
    billed = model_pricing.normalize_response_billing(obj)
    assert strict.usage_observed and billed.usage_observed and not billed.usage_invalid
    assert billed.usage_present
    assert (billed.input_tokens, billed.output_tokens, billed.cache_read_tokens) == (80, 10, 20)


@pytest.mark.parametrize("usage", [
    {"input_tokens": 0}, {"output_tokens": 0}, {"prompt_tokens": 0},
    {"completion_tokens": 0}, {"cache_read_tokens": 2},
    {"input_tokens": 0, "cache_creation_input_tokens": 0},
])
def test_accounting_incomplete_snapshot_is_not_observed_zero(usage):
    billed = model_pricing.normalize_response_billing({"usage": usage})
    assert billed.usage_present and not billed.usage_observed and billed.usage_invalid


@pytest.mark.parametrize("usage", [
    {"input_tokens": 7, "completion_tokens": 3},
    {"prompt_tokens": 7, "output_tokens": 3},
])
def test_accounting_complete_compatibility_aliases_keep_existing_semantics(usage):
    result = model_pricing.normalize_response_billing({"usage": usage})
    assert result.usage_observed and not result.usage_invalid
    assert (result.input_tokens, result.output_tokens) == (7, 3)


def test_accounting_never_fills_missing_dimensions_across_envelopes_or_events():
    a = {"usage": {"input_tokens": 1}, "response": {"usage": {"output_tokens": 2}}}
    b = _sse({"usage": {"input_tokens": 1}}, {"usage": {"output_tokens": 2}})
    for obj in (a, b):
        result = model_pricing.normalize_response_billing(obj)
        assert not result.usage_observed and result.usage_invalid


def test_accounting_complete_zero_and_later_snapshot_replace_not_sum():
    first = {"usage": _responses_usage()}
    second = {"usage": _responses_usage(0, 0, 0)}
    result = model_pricing.normalize_response_billing(_sse(first, first, second, second))
    assert result.usage_observed and not result.usage_invalid
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (0, 0, 0)
    result = model_pricing.normalize_response_billing(_sse({"usage": {"input_tokens": -1}}, first))
    assert result.usage_observed and not result.usage_invalid
    assert (result.input_tokens, result.output_tokens) == (80, 10)


def test_accounting_anthropic_partial_start_delta_accumulates_once_with_ttl_split():
    start = {"type": "message_start", "message": {"usage": {
        "input_tokens": 10, "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 5,
        "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 3},
    }}}
    delta = {"type": "message_delta", "usage": {"output_tokens": 4}}
    zeros = {"type": "message_delta", "usage": {
        "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }}
    assert not model_pricing.normalize_response_billing(start).usage_observed
    assert not model_pricing.normalize_response_billing(delta).usage_observed
    result = model_pricing.normalize_response_billing(_sse(start, delta, delta, zeros))
    assert result.usage_observed and not result.usage_invalid
    assert (result.input_tokens, result.output_tokens, result.cache_creation_tokens, result.cache_read_tokens) == (10, 4, 5, 20)
    assert (result.cache_creation_5m_tokens, result.cache_creation_1h_tokens) == (2, 3)


@pytest.mark.parametrize("location", ["root", "response", "data", "data.response", "message"])
@pytest.mark.parametrize("usage", [None, {}, [], {"input_tokens": 0}, {"input_tokens": 0, "output_tokens": 0}])
def test_accounting_usage_presence_is_independent_of_completeness(location, usage):
    obj = {}
    container = obj
    for key in ([] if location == "root" else location.split(".")):
        container = container.setdefault(key, {})
    assert not model_pricing.normalize_response_billing(obj).usage_present
    container["usage"] = usage
    result = model_pricing.normalize_response_billing(obj)
    assert result.usage_present
    assert result.usage_observed is (isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage)


def test_accounting_optional_details_and_actual_cost_preserve_contract():
    result = model_pricing.normalize_response_billing({"usage": {
        "input_tokens": 0, "output_tokens": 0, "input_tokens_details": None,
        "output_tokens_details": None, "cost_in_usd_ticks": 0}})
    assert result.usage_observed and result.actual_cost_ticks == 0
    result = model_pricing.normalize_response_billing({"usage": {"cost_in_usd_ticks": 42}})
    assert result.usage_present and not result.usage_observed and result.actual_cost_ticks == 42
    result = model_pricing.normalize_response_billing({
        "usage": {"input_tokens": 1, "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": -1}},
        "response": {"usage": _responses_usage(), "service_tier": "priority"},
    })
    assert result.usage_observed and (result.input_tokens, result.output_tokens) == (80, 10)
    assert result.service_tier == "priority"


@pytest.fixture
def fake_probe(monkeypatch):
    state = {"payload": {}, "protocol": "anthropic", "cleared": [], "calls": 0}

    class FakeChannel:
        type = "api"
        key = "api:accounting-fixture"
        enabled = True

        @property
        def protocol(self):
            return state["protocol"]

        async def build_upstream_request(self, body, model, **kwargs):
            assert body["stream"] is False
            return SimpleNamespace(url="https://fixture.invalid/completion", headers={}, body=json.dumps(body).encode())

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def post(self, *args, **kwargs):
            state["calls"] += 1
            return httpx.Response(200, json=state["payload"])

    channel = FakeChannel()
    state["channel"] = channel
    monkeypatch.setattr(config, "get", lambda: {"probe": {}, "cooldownRecovery": {"enabled": True}})
    monkeypatch.setattr(probe.network, "async_client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(probe.registry, "get_channel", lambda key: channel)
    monkeypatch.setattr(probe.cooldown, "active_entries", lambda: [{
        "channel_key": channel.key, "model": "fixture", "cooldown_until": None, "last_error_message": "HTTP 502",
    }])
    monkeypatch.setattr(probe.cooldown, "clear", lambda *args: state["cleared"].append(args))
    return state


@pytest.mark.parametrize("protocol", ["anthropic", "openai-chat", "openai-responses"])
@pytest.mark.parametrize("payload", [{}, [], {"status": "ok"}])
@pytest.mark.asyncio
async def test_accounting_invalid_probe_json_cannot_clear_cooldown(fake_probe, protocol, payload):
    fake_probe.update(protocol=protocol, payload=payload)
    ok, _, reason = await probe.probe_channel_model(fake_probe["channel"], "fixture")
    assert not ok and reason
    assert await probe.recovery_run_once() == 0
    assert fake_probe["cleared"] == []


@pytest.mark.parametrize("protocol,payload", [
    ("anthropic", {"type": "message", "content": [], "stop_reason": "end_turn"}),
    ("anthropic", {"type": "message", "content": [], "stop_reason": "max_tokens"}),
    ("anthropic", {"type": "message", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}], "stop_reason": "tool_use"}),
    ("openai-chat", {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}),
    ("openai-chat", {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}]}),
    ("openai-chat", {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}),
    ("openai-responses", {"id": "r", "object": "response", "status": "completed", "output": []}),
    ("openai-responses", {"id": "r", "object": "response", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}),
])
@pytest.mark.asyncio
async def test_accounting_valid_probe_terminals_allow_empty_tool_and_length(fake_probe, protocol, payload):
    fake_probe.update(protocol=protocol, payload=payload)
    ok, _, reason = await probe.probe_channel_model(fake_probe["channel"], "fixture")
    assert ok and reason is None
    assert await probe.recovery_run_once() == 1
    assert fake_probe["cleared"] == [(fake_probe["channel"].key, "fixture")]


@pytest.mark.parametrize("payload", [
    {"status": "queued"}, {"status": "in_progress"}, {"status": "failed"},
    {"status": "completed", "error": {"message": "busy"}},
    {"type": "error", "error": {"message": "busy"}},
])
@pytest.mark.asyncio
async def test_accounting_probe_rejects_async_and_failed_responses(fake_probe, payload):
    fake_probe.update(protocol="openai-responses", payload=payload)
    assert await probe.recovery_run_once() == 0
    assert not fake_probe["cleared"]


@pytest.mark.asyncio
async def test_accounting_oauth_probe_still_skips_network(fake_probe):
    fake_probe["channel"].type = "oauth"
    assert await probe.probe_channel_model(fake_probe["channel"], "fixture") == (False, 0, "oauth not probable")
    assert fake_probe["calls"] == 0


@pytest.mark.parametrize("bad", ["inf", "-inf", "NaN", "1e999", "-0.01", "1e308", 10**400])
def test_accounting_invalid_utilization_does_not_create_quota_hit(bad):
    raw = {headers.H_5H_UTIL: bad, headers.H_7D_UTIL: "1.05"}
    assert headers.parse_rate_limit_headers(raw) == {"seven_day_util": 105.0}
    assert not headers.is_window_exceeded(raw, "5h")
    assert headers.is_window_exceeded(raw, "7d")


@pytest.mark.parametrize("bad", ["inf", "-inf", "NaN", "1e999", "-1", 10**400])
def test_accounting_invalid_reset_does_not_discard_other_valid_fields(bad):
    raw = {headers.H_5H_UTIL: "0.5", headers.H_5H_RESET: bad, headers.H_7D_RESET: "1700000000000"}
    assert headers.parse_rate_limit_headers(raw) == {
        "five_hour_util": 50.0, "seven_day_reset": "2023-11-14T22:13:20Z",
    }


@pytest.mark.parametrize("value", ["0", "0.5", "1", "1.05", "2"])
def test_accounting_fraction_zero_and_real_overquota_remain_valid(value):
    raw = {headers.H_5H_UTIL: value}
    assert headers.parse_rate_limit_headers(raw) == {"five_hour_util": float(value) * 100}
    assert headers.is_window_exceeded(raw, "5h") is (float(value) >= 1)
    assert headers.is_window_exceeded({headers.H_5H_SURPASS: "true", headers.H_5H_UTIL: "nan"}, "5h")
