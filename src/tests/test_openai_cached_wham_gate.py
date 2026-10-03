"""A cached WHAM gate must not override newer complete low Codex windows."""
from __future__ import annotations

import json
import time

import pytest

from src.tests.test_openai_oauth_quota import (
    _import_modules, _MockResp, _UiRecorder, _preheat_oauth_menu_windows,
)
from src.tests.test_codex_quota_fragments import _account, _save_headers


_LOW_HEADERS = {
    "x-codex-primary-used-percent": "48",
    "x-codex-primary-window-minutes": "300",
    "x-codex-primary-reset-after-seconds": "15780",
    "x-codex-secondary-used-percent": "8",
    "x-codex-secondary-window-minutes": "10080",
    "x-codex-secondary-reset-after-seconds": "601200",
}


def _wham(m, key, observed_at, *, hard=None, monthly=False, no_windows=False):
    rate = {"allowed": False, "limit_reached": True}
    if not no_windows:
        rate.update(
            primary_window={
                "used_percent": 100,
                "limit_window_seconds": 2592000 if monthly else 18000,
                "reset_at": int(time.time()) + 3600,
            },
            secondary_window={
                "used_percent": 8, "limit_window_seconds": 604800,
                "reset_at": int(time.time()) + 601200,
            },
        )
    payload = {"plan_type": "plus", "rate_limit": rate}
    if hard == "spend":
        payload["spend_control"] = {"reached": True}
    elif hard == "overage":
        payload["credits"] = {"overage_limit_reached": True}
    elif hard:
        payload["rate_limit_reached_type"] = hard
    usage = m["openai_provider"].normalize_wham_usage(payload)
    m["state_db"].quota_save(key, {
        **m["oauth_manager"].flatten_usage(usage), "fetched_at": observed_at,
    })
    return usage


def _cached(m, key, *, direct=False, threshold=95):
    manager = m["oauth_manager"]
    if direct:
        usage = manager.usage_from_quota_row(m["state_db"].quota_load(key))
        return manager.evaluate_and_toggle_by_usage(key, usage, threshold=threshold, fresh=False)
    return manager.evaluate_and_toggle_by_cached_quota(key, threshold=threshold)


def test_reopening_menu_does_not_redisable_recovered_account(m, monkeypatch):
    key, channel = _account(m, "menu-gate")
    _wham(m, key, int(time.time() * 1000) - 60000)
    # Use the ordinary response sampler, not a hand-built menu snapshot.
    m["failover"]._maybe_record_codex_snapshot(channel, _MockResp(_LOW_HEADERS))
    _preheat_oauth_menu_windows(m)
    recorder = _UiRecorder()
    monkeypatch.setattr(m["ui"], "api", recorder)
    for callback in ("open", "reopen", "reopen-again"):
        m["oauth_menu"].show(42, 100, callback)
        account = m["oauth_manager"].get_account(key)
        assert account["enabled"] is True
        assert account.get("disabled_reason") is None
        text = recorder.last("editMessageText")["text"]
        assert "配额禁用" not in text
        assert "48%" in text and "8%" in text
    # Preserve the original WHAM facts, rather than rewriting stored truth.
    original = json.loads(m["state_db"].quota_load(key)["raw_data"])
    assert original["openai"]["allowed"] is False
    assert original["openai"]["limit_reached"] is True


@pytest.mark.parametrize("direct", [False, True])
def test_complete_new_low_windows_supersede_cached_subscription_gate(m, direct):
    key, _ = _account(m, "complete-gate")
    now = int(time.time() * 1000)
    _wham(m, key, now - 1000)
    _save_headers(m, key, _LOW_HEADERS, now)
    result = _cached(m, key, direct=direct)
    assert result["action"] == ("kept_enabled" if direct else "cached_below_threshold"), result
    assert m["oauth_manager"].get_account(key)["enabled"] is True


@pytest.mark.parametrize("fragment", [
    "primary-only", "secondary-only", "old-primary", "credits-only", "reserve-only",
    "same-clock", "expired", "monthly-missing", "unknown-active-clock", "no-wham-windows",
])
@pytest.mark.parametrize("direct", [False, True])
def test_unrelated_partial_or_ambiguous_samples_do_not_clear_wham_gate(m, fragment, direct):
    key, _ = _account(m, f"partial-{fragment}")
    now = int(time.time() * 1000)
    active_at = 0 if fragment == "unknown-active-clock" else now - 1000
    _wham(m, key, active_at, monthly=fragment == "monthly-missing",
          no_windows=fragment == "no-wham-windows")
    if fragment in {"credits-only", "reserve-only", "old-primary"}:
        _save_headers(m, key, _LOW_HEADERS, active_at - 1000)
    if fragment in {"primary-only", "secondary-only", "old-primary"}:
        prefix = "primary" if fragment == "primary-only" else "secondary"
        headers = {k: v for k, v in _LOW_HEADERS.items() if f"-{prefix}-" in k}
    elif fragment == "credits-only":
        headers = {"x-codex-credits-balance": "9"}
    elif fragment == "reserve-only":
        headers = {"x-gpt-reserve-primary-used-percent": "1",
                   "x-gpt-reserve-primary-window-minutes": "60"}
    else:
        headers = dict(_LOW_HEADERS)
    if fragment == "expired":
        for prefix in ("primary", "secondary"):
            headers.pop(f"x-codex-{prefix}-reset-after-seconds")
            headers[f"x-codex-{prefix}-reset-at"] = str(int(time.time()) - 1)
    _save_headers(m, key, headers, active_at if fragment == "same-clock" else now)
    result = _cached(m, key, direct=direct)
    assert result["action"] == "wham_limit_disabled", result
    assert m["oauth_manager"].get_account(key)["disabled_reason"] == "quota"


@pytest.mark.parametrize("hard", [
    "spend", "overage", "workspace_owner_credits_depleted", "workspace_member_credits_depleted",
    "workspace_owner_usage_limit_reached", "workspace_member_usage_limit_reached",
])
@pytest.mark.parametrize("direct", [False, True])
def test_new_low_windows_never_clear_spend_or_credit_hard_limits(m, hard, direct):
    key, _ = _account(m, "hard-gate")
    now = int(time.time() * 1000)
    _wham(m, key, now - 1000, hard=hard)
    _save_headers(m, key, _LOW_HEADERS, now)
    result = _cached(m, key, direct=direct)
    assert result["action"] == "wham_limit_disabled", result


def test_new_active_wham_limit_remains_authoritative(m):
    key, _ = _account(m, "new-active-gate")
    now = int(time.time() * 1000)
    _wham(m, key, now - 1000)
    _save_headers(m, key, _LOW_HEADERS, now)
    fresh = _wham(m, key, now + 1)
    assert _cached(m, key)["action"] == "wham_limit_disabled"
    m["oauth_manager"].set_enabled(key, True)
    result = m["oauth_manager"].evaluate_and_toggle_by_usage(key, fresh, fresh=True)
    assert result["action"] == "wham_limit_disabled", result


@pytest.mark.parametrize("direct", [False, True])
def test_cached_low_windows_do_not_auto_resume_existing_quota_pause(m, direct):
    key, _ = _account(m, "still-cached-gate")
    now = int(time.time() * 1000)
    _wham(m, key, now - 1000)
    _save_headers(m, key, _LOW_HEADERS, now)
    manager = m["oauth_manager"]
    manager.set_disabled_by_quota(key, None)
    result = _cached(m, key, direct=direct)
    assert result["action"] == ("quota_stale_keep_disabled" if direct else "cached_below_threshold"), result
    assert manager.get_account(key)["enabled"] is False


def test_configured_threshold_still_disables_new_over_threshold_windows(m):
    key, _ = _account(m, "configured-gate")
    now = int(time.time() * 1000)
    _wham(m, key, now - 1000)
    _save_headers(m, key, _LOW_HEADERS, now)
    result = _cached(m, key, threshold=40)
    assert result["any_over"] is True, result
    assert m["oauth_manager"].get_account(key)["enabled"] is False


@pytest.mark.parametrize("code,kind", [("usage_limit_reached", "usage_limit"), ("credits_exhausted", "quota")])
def test_newer_actual_refusal_is_not_overridden_by_low_windows(m, code, kind):
    from types import SimpleNamespace
    from src.openai.recovery import ResponseErrorAdvice

    key, channel = _account(m, "actual-refusal-gate")
    now = int(time.time() * 1000)
    _wham(m, key, now - 1000)
    _save_headers(m, key, _LOW_HEADERS, now - 1)
    result = SimpleNamespace(error_advice=ResponseErrorAdvice(
        code=code, kind=kind, active_limit="codex", observed_at=time.time(),
        cooldown_until=now + 600000,
    ))
    assert m["failover"]._apply_codex_error_policy(channel, "gpt-5.5", result)
    decision = _cached(m, key, direct=True)
    assert decision["action"] == "still_over_quota", decision
    assert m["oauth_manager"].get_account(key)["enabled"] is False
