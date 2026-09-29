"""Codex credits visibility, freshness and shared login/refresh acquisition."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.tests.test_openai_oauth_quota import (
    _import_modules, _setup, _UiRecorder, _preheat_oauth_menu_windows,
)
from src.tests.test_codex_quota_fragments import _account, _save_headers


def _payload(credits):
    result = {
        "plan_type": "pro",
        "rate_limit": {"allowed": True, "limit_reached": False,
                       "primary_window": {"used_percent": 12, "limit_window_seconds": 18000}},
        "rate_limit_reset_credits": {"available_count": 0},
    }
    if credits is not None:
        result["credits"] = credits
    return result


def _save(m, key, credits, at=1790696400000):
    usage = m["openai_provider"].normalize_wham_usage(_payload(credits))
    m["state_db"].quota_save(key, {**m["oauth_manager"].flatten_usage(usage), "fetched_at": at})
    return m["state_db"].quota_load(key)


@pytest.mark.parametrize("credits,expected", [
    (None, None), ({}, None),
    ({"has_credits": False, "unlimited": False}, None),
    ({"has_credits": False, "balance": "0"}, None),
    ({"has_credits": True, "balance": "0"}, None),
    ({"balance": "-1"}, None), ({"balance": "NaN"}, None),
    ({"balance": "Infinity"}, None), ({"balance": True}, None),
    ({"balance": "25000.00005"}, "余额 25,000.00005 credits"),
    ({"has_credits": True, "balance": "25000.5000"}, "余额 25,000.5 credits"),
    ({"unlimited": True, "has_credits": False, "balance": "9"}, "不限量"),
    ({"has_credits": True}, "可用 · 余额未返回"),
    ({"has_credits": True, "balance": None}, "可用 · 余额未返回"),
    ({"has_credits": False, "balance": "25"}, "余额 25 credits"),
])
def test_credits_visibility_on_list_and_detail(m, credits, expected):
    key, _ = _account(m, "visibility")
    _save(m, key, credits)
    menu = m["oauth_menu"]
    account = m["oauth_manager"].get_account(key)
    for text in (menu._format_account_block(account), menu._format_usage_block(key)):
        rows = [line for line in text.splitlines() if "Credits:" in line]
        if expected is None:
            assert rows == []
        else:
            assert len(rows) == 1
            # The availability warning is part of the same credit value.
            assert expected in rows[0]
            assert "$" not in rows[0]
            if credits.get("has_credits") is False and credits.get("balance") == "25":
                assert "上游标记不可用" in rows[0]
        assert "5h" in text
    if expected:
        detail = menu._format_usage_block(key)
        assert "观测于 09-29 23:40:00 · 官方用量接口" in detail
        assert detail.index("Credits:") < detail.index("5h")


def test_missing_credits_stays_unknown_in_normalizer(m):
    credits = m["openai_provider"].normalize_wham_usage(_payload(None))["openai"]["credits"]
    assert credits == {"has_credits": None, "unlimited": None, "balance": None}


@pytest.mark.parametrize("path", ["manual", "manual_progress", "monitor", "access", "startup", "token_refresh"])
def test_all_refresh_paths_fetch_and_save_credits(m, monkeypatch, path):
    key, _ = _account(m, f"refresh-{path}")
    manager = m["oauth_manager"]
    calls = []

    def payload():
        calls.append("wham")
        return _payload({"has_credits": True, "unlimited": False, "balance": "25000.75"})

    monkeypatch.setattr(m["openai_provider"], "_mock_wham_payload", payload)
    if path in {"manual", "manual_progress"}:
        result = m["oauth_menu"]._fetch_and_save_usage_result_sync(
            key, chat_id=42, on_stage=(lambda *args: None) if path == "manual_progress" else None,
        )
        assert not result.get("error"), result
    elif path == "monitor":
        asyncio.run(manager.quota_monitor_once())
    elif path == "access":
        monkeypatch.setattr(manager, "_should_skip_access_refresh", lambda: False)
        assert asyncio.run(manager.ensure_quota_fresh(key))
    elif path == "startup":
        assert asyncio.run(manager.preload_openai_reset_credit_details_once())[key] == "refreshed"
    else:
        async def token(*args, **kwargs):
            return "at-x"
        async def metadata(*args, **kwargs):
            return False
        monkeypatch.setattr(manager, "force_refresh", token)
        monkeypatch.setattr(manager, "ensure_valid_token", token)
        monkeypatch.setattr(manager, "ensure_openai_metadata_fresh", metadata)
        monkeypatch.setattr(manager, "_token_expiry", lambda acc: datetime.now(timezone.utc) - timedelta(seconds=1))
        result = asyncio.run(manager.proactive_refresh_once())
        assert "refreshed" in result.values(), result
    assert calls == ["wham"]
    row = m["state_db"].quota_load(key)
    assert manager.usage_from_quota_row(row)["openai"]["credits"]["balance"] == "25000.75"
    assert "25,000.75 credits" in m["oauth_menu"]._format_usage_block(key)
    assert row.get("codex_usage_failed_at") is None


def test_pkce_login_fetches_credits_before_showing_account(m, monkeypatch):
    _setup(m)
    menu = m["oauth_menu"]
    recorder = _UiRecorder()
    monkeypatch.setattr(m["ui"], "api", recorder)
    calls = []

    def payload():
        calls.append("wham")
        return _payload({"has_credits": True, "balance": "45678.125"})

    monkeypatch.setattr(m["openai_provider"], "_mock_wham_payload", payload)
    menu.on_login_openai_start(42, 100, "cb")
    state = m["states"].get_state(42)["data"]["state"]
    menu.on_login_openai_code_input(42, f"http://localhost:1455/auth/callback?code=mock_credits&state={state}")
    accounts = [acc for acc in m["oauth_manager"].list_accounts() if acc.get("provider") == "openai"]
    assert len(accounts) == 1
    key = m["oauth_manager"]._account_key(accounts[0])
    row = m["state_db"].quota_load(key)
    assert row and calls == ["wham"]
    assert m["oauth_manager"].usage_from_quota_row(row)["openai"]["credits"]["balance"] == "45678.125"
    assert "45,678.125 credits" in menu._format_account_block(accounts[0])


@pytest.mark.parametrize("timeout", [False, True])
def test_failed_refresh_preserves_balance_and_marks_old_then_recovers(m, monkeypatch, timeout):
    key, _ = _account(m, f"failure-{timeout}")
    old = _save(m, key, {"has_credits": True, "balance": "25000.5"})
    clock = [old["fetched_at"] + 1000]
    monkeypatch.setattr(m["state_db"], "now_ms", lambda: clock[0])

    async def failure(*args, **kwargs):
        if timeout:
            await asyncio.sleep(10)
        raise RuntimeError("synthetic refresh failure")

    monkeypatch.setattr(m["openai_provider"], "fetch_wham_usage", failure)
    with pytest.raises((RuntimeError, asyncio.TimeoutError)):
        asyncio.run(m["oauth_manager"].fetch_usage_snapshot(key, usage_timeout_s=0.001 if timeout else None))
    row = m["state_db"].quota_load(key)
    assert row["raw_data"] == old["raw_data"] and row["fetched_at"] == old["fetched_at"]
    assert row["codex_usage_failed_at"] == clock[0]
    assert "刷新失败，旧数据" in m["oauth_menu"]._format_usage_block(key)
    assert "25,000.5 credits" in m["oauth_menu"]._format_usage_block(key)
    assert "synthetic" not in m["oauth_menu"]._format_usage_block(key)
    clock[0] += 1000
    _save(m, key, {"has_credits": True, "balance": "24000"}, at=clock[0])
    assert "旧数据" not in m["oauth_menu"]._format_usage_block(key)
    assert "24,000 credits" in m["oauth_menu"]._format_usage_block(key)


def test_credit_observation_clock_is_not_quota_or_other_credit_field_clock(m):
    key, _ = _account(m, "clock")
    base = 1790696400000
    _save_headers(m, key, {"x-codex-credits-balance": "25000"}, base)
    expected_key = m["oauth_manager"].account_state_key(m["oauth_manager"].get_account(key))
    m["state_db"].quota_record_openai_usage_failure(key, started_at=base + 1000, expected_state_key=expected_key)
    _save_headers(m, key, {"x-codex-credits-has-credits": "true", "x-codex-primary-used-percent": "15"}, base + 2000)
    detail = m["oauth_menu"]._format_usage_block(key)
    assert "刷新失败，旧数据" in detail
    assert "观测于 09-29 23:40:00 · 上游响应采样" in detail
    _save_headers(m, key, {"x-codex-credits-balance": "24000"}, base + 3000)
    detail = m["oauth_menu"]._format_usage_block(key)
    assert "旧数据" not in detail and "24,000 credits" in detail


def test_old_failure_does_not_mark_new_success_as_stale(m):
    key, _ = _account(m, "newer-success")
    base = 1790696400000
    _save(m, key, {"balance": "20000"}, at=base + 2000)
    expected_key = m["oauth_manager"].account_state_key(m["oauth_manager"].get_account(key))
    m["state_db"].quota_record_openai_usage_failure(key, started_at=base + 1000, expected_state_key=expected_key)
    assert m["state_db"].quota_load(key).get("codex_usage_failed_at") is None


@pytest.mark.parametrize("credits,visible", [
    ({"has_credits": True, "balance": "25000"}, True),
    ({"has_credits": False, "balance": "0"}, False),
    (None, False),
])
def test_manual_refresh_failure_immediately_redraws_cached_credit_status(m, monkeypatch, credits, visible):
    key, _ = _account(m, "manual-failure")
    old = _save(m, key, credits)
    monkeypatch.setattr(m["state_db"], "now_ms", lambda: old["fetched_at"] + 1000)
    _preheat_oauth_menu_windows(m)
    menu = m["oauth_menu"]
    _, original_keyboard = menu._detail_text_and_kb(key)
    recorder = _UiRecorder()
    monkeypatch.setattr(m["ui"], "api", recorder)
    calls = []

    async def failure(*args, **kwargs):
        calls.append("wham")
        raise RuntimeError("synthetic refresh failure")

    monkeypatch.setattr(m["openai_provider"], "fetch_wham_usage", failure)
    menu.on_refresh_usage(42, 100, "cb", m["ui"].register_code(key))
    message = recorder.last("editMessageText")
    assert message and "用量刷新失败" in message["text"]
    assert ("Credits:" in message["text"]) is visible
    if visible:
        assert "25,000 credits" in message["text"] and "刷新失败，旧数据" in message["text"]
    assert message["reply_markup"] == original_keyboard
    assert calls == ["wham"]


def test_later_card_save_cannot_retimestamp_credits_or_override_newer_sampling(m, monkeypatch):
    key, _ = _account(m, "save-clock")
    manager = m["oauth_manager"]
    base = 1790696400000
    monkeypatch.setattr(m["state_db"], "now_ms", lambda: base)
    monkeypatch.setattr(m["openai_provider"], "_mock_wham_payload", lambda: _payload({"balance": "25000"}))
    usage = asyncio.run(manager.fetch_usage(key))
    assert usage["openai"]["credits_observed_at"]["balance"] == base
    # A later reset-card result saves the same usage again; it is not a new balance read.
    m["state_db"].quota_save(key, {**manager.flatten_usage(usage), "fetched_at": base + 5000})
    detail = m["oauth_menu"]._format_usage_block(key)
    assert "观测于 09-29 23:40:00" in detail
    _save_headers(m, key, {"x-codex-credits-balance": "24000"}, base + 2000)
    assert "24,000 credits" in m["oauth_menu"]._format_usage_block(key)
    assert manager.usage_from_quota_row(m["state_db"].quota_load(key))["openai"]["credits_sources"]["balance"] == "response"
