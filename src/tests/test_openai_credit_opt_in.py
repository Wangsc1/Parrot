"""Account-local Codex credit consent: TG toggle, quota gates and recovery."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from src.tests.test_openai_oauth_quota import (
    _import_modules, _UiRecorder, _MockResp, _preheat_oauth_menu_windows,
)
from src.tests.test_codex_quota_fragments import _account, _save_headers


def _allow(m, key, enabled=True):
    m["config"].update(lambda cfg: next(
        acc for acc in cfg["oauthAccounts"] if m["oauth_manager"]._account_key(acc) == key
    ).update(allowCredits=enabled))


def _payload(*, credits=None, no_plan=False, hard=None):
    rate = {"allowed": False, "limit_reached": True}
    if not no_plan:
        rate["secondary_window"] = {
            "used_percent": 100, "limit_window_seconds": 604800,
            "reset_at": int(time.time()) + 86400,
        }
    value = {
        "plan_type": "free" if no_plan else "pro", "rate_limit": rate,
        "credits": credits if credits is not None else {
            "has_credits": True, "unlimited": False, "balance": "62500",
        },
    }
    if hard == "spend":
        value["spend_control"] = {"reached": True}
    elif hard == "overage":
        value["credits"]["overage_limit_reached"] = True
    elif hard:
        value["rate_limit_reached_type"] = hard
    return value


def _usage(m, key, **kwargs):
    usage = m["openai_provider"].normalize_wham_usage(_payload(**kwargs))
    m["state_db"].quota_save(key, m["oauth_manager"].flatten_usage(usage))
    return usage


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("no_plan", [False, True])
def test_credit_consent_gates_weekly_and_no_subscription(m, enabled, no_plan):
    key, _ = _account(m, "consent")
    if enabled is not None:
        _allow(m, key, enabled)
    # Expired subscription metadata is not an auth-token expiry or a credit gate.
    m["config"].update(lambda c: c["oauthAccounts"][0].update(
        plan_type="free" if no_plan else "pro", subscription_expires_at="2020-01-01T00:00:00Z",
    ))
    usage = _usage(m, key, no_plan=no_plan)
    result = m["oauth_manager"].evaluate_and_toggle_by_usage(key, usage, threshold=100)
    assert result["action"] == ("kept_enabled" if enabled else "wham_limit_disabled"), result
    assert m["oauth_manager"].get_account(key)["enabled"] is bool(enabled)


@pytest.mark.parametrize("credits,usable", [
    ({"has_credits": True, "balance": "62500"}, True),
    ({"unlimited": True, "has_credits": False, "balance": "0"}, True),
    ({"has_credits": True}, True),
    ({"balance": "62500"}, True),
    ({"has_credits": False, "balance": "62500"}, False),
    ({"has_credits": True, "balance": "0"}, False),
    ({"balance": "-1"}, False),
    ({"balance": "NaN"}, False),
    ({}, False),
])
def test_credit_availability_is_not_just_a_display_balance(m, credits, usable):
    key, _ = _account(m, "availability")
    _allow(m, key)
    result = m["oauth_manager"].evaluate_and_toggle_by_usage(key, _usage(m, key, credits=credits))
    assert (result["action"] == "kept_enabled") is usable


@pytest.mark.parametrize("hard", [
    "spend", "overage", "workspace_owner_credits_depleted", "workspace_member_credits_depleted",
    "workspace_owner_usage_limit_reached", "workspace_member_usage_limit_reached",
])
def test_credit_consent_never_overrides_hard_limits(m, hard):
    key, _ = _account(m, "hard-limit")
    _allow(m, key)
    result = m["oauth_manager"].evaluate_and_toggle_by_usage(key, _usage(m, key, hard=hard))
    assert result["action"] == "wham_limit_disabled"
    cached = m["oauth_manager"].usage_from_quota_row(m["state_db"].quota_load(key))
    assert not m["oauth_manager"].openai_credits_usable(key, cached)


@pytest.mark.parametrize("no_plan", [False, True])
def test_monitor_recovers_old_window_pause_without_erasing_model_cooldown(m, monkeypatch, no_plan):
    key, _ = _account(m, "recover")
    manager = m["oauth_manager"]
    snap = _save_headers(m, key, {
        "x-codex-secondary-used-percent": "100", "x-codex-secondary-window-minutes": "10080",
        "x-codex-secondary-reset-after-seconds": "86400",
    }, int(time.time() * 1000))
    manager.set_disabled_by_quota(key, "2099-01-01T00:00:00Z", observation=manager.codex_quota_observation(snap))
    from src import cooldown
    cooldown.record_error(f"oauth:{key}", "restricted-model", "model_not_entitled",
                          cooldown_until=int((time.time() + 3600) * 1000))
    _allow(m, key)
    monkeypatch.setattr(m["openai_provider"], "_mock_wham_payload", lambda: _payload(no_plan=no_plan))
    asyncio.run(manager.quota_monitor_once())
    acc = manager.get_account(key)
    assert acc["enabled"] and acc.get("disabled_reason") is None
    assert cooldown.get_state(f"oauth:{key}", "restricted-model")
    assert not m["registry"].get_channel(f"oauth:{key}").disabled_reason


@pytest.mark.parametrize("reason", ["user", "auth_error"])
def test_opt_in_does_not_resume_manual_or_auth_disabled_accounts(m, reason):
    key, _ = _account(m, "disabled")
    _allow(m, key)
    manager = m["oauth_manager"]
    manager.set_enabled(key, False, reason=reason)
    result = manager.evaluate_and_toggle_by_usage(key, _usage(m, key))
    assert result["action"] == f"noop_{reason}"
    assert manager.get_account(key)["disabled_reason"] == reason


def test_cached_positive_credit_evidence_never_recovers_account(m):
    key, _ = _account(m, "stale")
    _allow(m, key)
    _usage(m, key)
    manager = m["oauth_manager"]
    manager.set_disabled_by_quota(key, "2099-01-01T00:00:00Z")
    result = manager.evaluate_and_toggle_by_cached_quota(key)
    assert result["action"] == "quota_stale_keep_disabled"


def test_fresh_missing_credits_does_not_recover_from_old_positive_balance(m):
    key, _ = _account(m, "missing")
    _allow(m, key)
    _usage(m, key)
    manager = m["oauth_manager"]
    manager.set_disabled_by_quota(key, "2099-01-01T00:00:00Z")
    missing = m["openai_provider"].normalize_wham_usage(_payload(credits={}))
    result = manager.evaluate_and_toggle_by_usage(key, missing)
    assert result["action"] == "wham_limit_keep_disabled"


def test_headers_do_not_disable_credit_funded_requests_and_depletion_reapplies_gate(m):
    key, channel = _account(m, "headers")
    _allow(m, key)
    _usage(m, key)
    headers = {
        "x-codex-secondary-used-percent": "100", "x-codex-secondary-window-minutes": "10080",
        "x-codex-secondary-reset-after-seconds": "86400",
        "x-codex-credits-has-credits": "true", "x-codex-credits-balance": "62499",
    }
    m["failover"]._maybe_record_codex_snapshot(channel, _MockResp(headers))
    assert m["oauth_manager"].get_account(key)["enabled"]
    # No quota window in this fragment, and the persistence throttle is active.
    m["failover"]._maybe_record_codex_snapshot(channel, _MockResp({
        "x-codex-credits-has-credits": "false", "x-codex-credits-balance": "0",
    }))
    assert m["oauth_manager"].get_account(key)["disabled_reason"] == "quota"
    row = m["state_db"].quota_load(key)
    assert row["codex_credits_has_credits"] is False
    assert row["codex_credits_balance"] == "0"


def test_sparse_header_can_use_cached_credits_and_newer_false_wins(m):
    key, _ = _account(m, "sparse")
    _allow(m, key)
    usage = _usage(m, key)
    manager = m["oauth_manager"]
    assert manager.openai_credits_usable(key, snapshot={"primary_used_pct": 100})
    now = int(time.time() * 1000)
    usage["openai"]["credits_observed_at"] = dict.fromkeys(("has_credits", "balance", "unlimited"), now - 1000)
    _save_headers(m, key, {"x-codex-credits-has-credits": "false", "x-codex-credits-balance": "0"}, now)
    assert not manager.openai_credits_usable(key, usage)


@pytest.mark.parametrize("code,kind", [("usage_limit_reached", "usage_limit"), ("credits_exhausted", "quota")])
def test_actual_upstream_refusal_is_not_overridden_by_credit_balance(m, code, kind):
    key, channel = _account(m, "refusal")
    _allow(m, key)
    usage = _usage(m, key)
    from src.openai.recovery import ResponseErrorAdvice
    result = SimpleNamespace(error_advice=ResponseErrorAdvice(
        code=code, kind=kind, active_limit="codex", observed_at=time.time(),
        cooldown_until=int((time.time() + 600) * 1000),
    ))
    assert m["failover"]._apply_codex_error_policy(channel, "gpt-5.5", result)
    decision = m["oauth_manager"].evaluate_and_toggle_by_usage(key, usage)
    assert decision["action"] == "still_over_quota"
    assert m["oauth_manager"].get_account(key)["disabled_reason"] == "quota"


def test_tg_toggle_placement_persistence_recovery_and_off(m, monkeypatch):
    key, _ = _account(m, "menu")
    menu = m["oauth_menu"]
    manager = m["oauth_manager"]
    usage = _usage(m, key)
    manager.evaluate_and_toggle_by_usage(key, usage)
    _preheat_oauth_menu_windows(m)
    recorder = _UiRecorder()
    monkeypatch.setattr(m["ui"], "api", recorder)
    monkeypatch.setattr(m["openai_provider"], "_mock_wham_payload", _payload)
    _, keyboard = menu._detail_text_and_kb(key, page=2, filter_key="openai")
    row = next(row for row in keyboard["inline_keyboard"] if "重置额度" in row[0]["text"])
    assert len(row) == 2 and row[1]["text"] == "⬜ 允许使用积分"
    callback = row[1]["callback_data"]
    assert len(callback.encode()) <= 64
    assert menu.handle_callback(42, 100, "cb", callback)
    account = manager.get_account(key)
    assert account["allowCredits"] is True and account["enabled"]
    # Same email in another workspace must retain its own default-off policy.
    manager.add_account({"email": account["email"], "provider": "openai",
                         "access_token": "at-other", "refresh_token": "rt-other",
                         "chatgpt_account_id": "other-workspace"})
    other = next(a for a in manager.list_accounts() if a.get("chatgpt_account_id") == "other-workspace")
    assert other.get("allowCredits", False) is False
    other_key = manager._account_key(other)
    _, other_kb = menu._detail_text_and_kb(other_key)
    assert any(b["text"] == "⬜ 允许使用积分" for row in other_kb["inline_keyboard"] for b in row)
    edited = recorder.last("editMessageText")
    assert any(b["text"] == "✅ 允许使用积分" for row in edited["reply_markup"]["inline_keyboard"] for b in row)
    # Persisted account setting; no global opt-in and no subscription mutation.
    on_disk = json.loads(m["config"].CONFIG_PATH.read_text()) if hasattr(m["config"].CONFIG_PATH, "read_text") else json.load(open(m["config"].CONFIG_PATH))
    assert on_disk["oauthAccounts"][0]["allowCredits"] is True
    assert menu.handle_callback(42, 100, "cb2", callback)
    account = manager.get_account(key)
    assert account["allowCredits"] is False and account["disabled_reason"] == "quota"


def test_toggle_saves_consent_but_refresh_failure_does_not_restore(m, monkeypatch):
    key, _ = _account(m, "refresh-failure")
    manager = m["oauth_manager"]
    manager.evaluate_and_toggle_by_usage(key, _usage(m, key))
    async def failed(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(m["openai_provider"], "fetch_wham_usage", failed)
    control = m["oauth_menu"].oauth_control
    context = m["oauth_menu"]._management_context(42)
    result = control.toggle_openai_credits(context, key)
    assert result["allowed"] is True and result.get("error")
    assert manager.get_account(key)["disabled_reason"] == "quota"


def test_exact_relogin_preserves_local_credit_consent(m):
    key, _ = _account(m, "relogin")
    _allow(m, key)
    manager = m["oauth_manager"]
    entry = dict(manager.get_account(key), access_token="new-token", allowCredits=False)
    result = manager.replace_exact_identity(key, entry)
    assert result["status"] == "replaced"
    assert manager.get_account(key)["allowCredits"] is True
