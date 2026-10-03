"""A recovered WHAM snapshot must not revive an older Codex refusal."""
from __future__ import annotations

import json
import time

import pytest

from src.tests.test_openai_oauth_quota import _import_modules, _low_wham
from src.tests.test_codex_quota_fragments import _account, _save_headers


_OLD_REFUSAL = "workspace_member_credits_depleted"


def _row(*, active=2000, refusal=1000, explicit=True, active_refusal=None):
    usage = _low_wham()
    if explicit:
        usage["openai"]["rate_limit_reached_type"] = active_refusal
        usage["openai"]["rate_limit_reached_type_explicit_null"] = active_refusal is None
    return {
        "five_hour_util": 10.0,
        "seven_day_util": 20.0,
        "codex_active_observed_at": active,
        "fetched_at": active,
        # New unrelated response fragments must not rejuvenate the refusal.
        "last_passive_update_at": 3000,
        "codex_rate_limit_reached_at": refusal,
        "codex_rate_limit_reached_type": _OLD_REFUSAL,
        "raw_data": json.dumps(usage),
    }


@pytest.mark.parametrize("active,refusal,explicit,active_refusal,expected", [
    (2000, 1000, True, None, None),
    (2000, 1000, True, "workspace_owner_usage_limit_reached", "workspace_owner_usage_limit_reached"),
    (1000, 2000, True, None, _OLD_REFUSAL),
    (2000, 2000, True, None, _OLD_REFUSAL),
    (2000, 1000, False, None, _OLD_REFUSAL),
])
def test_refusal_merge_respects_explicit_newer_wham(
    m, active, refusal, explicit, active_refusal, expected,
):
    row = _row(active=active, refusal=refusal, explicit=explicit,
               active_refusal=active_refusal)
    usage = m["oauth_manager"].usage_from_quota_row(row)
    assert usage["openai"]["rate_limit_reached_type"] == expected


@pytest.mark.parametrize("allow_credits", [False, True])
def test_recovered_cache_and_credit_fragment_do_not_disable(m, monkeypatch, allow_credits):
    key, _ = _account(m, f"recovered-refusal-{allow_credits}")
    manager = m["oauth_manager"]
    manager.get_account(key)["allowCredits"] = allow_credits
    row = _row()
    monkeypatch.setattr(m["state_db"], "quota_load", lambda _key: row)
    result = manager.evaluate_and_toggle_by_cached_quota(key)
    assert result["action"] == "cached_below_threshold"
    assert manager.get_account(key)["enabled"] is True
    # A successful response with no purchased credits is not a refusal.
    m["failover"]._maybe_auto_disable_by_codex_snapshot(
        key, manager.account_key_to_email(key),
        {"credits": {"has_credits": False}, "fetched_at": 4000},
    )
    assert manager.get_account(key)["enabled"] is True
    assert manager.get_account(key).get("disabled_reason") is None


_MISSING = object()
_NEW_REFUSAL = "workspace_owner_usage_limit_reached"


def _store_wham(m, key, now, raw_refusal, *, legacy=False, hard=None):
    payload = m["openai_provider"]._mock_wham_payload()
    payload.pop("rate_limit_reached_type", None)
    payload["credits"] = {"has_credits": True, "unlimited": False, "balance": "9"}
    if raw_refusal is not _MISSING:
        payload["rate_limit_reached_type"] = raw_refusal
    if hard == "spend":
        payload["spend_control"] = {"reached": True}
    elif hard == "overage":
        payload["credits"]["overage_limit_reached"] = True
    usage = m["openai_provider"].normalize_wham_usage(payload)
    if legacy:
        # Already persisted snapshots have no provenance for a normalized null.
        usage["openai"].pop("rate_limit_reached_type_explicit_null", None)
    m["state_db"].quota_save(key, {
        **m["oauth_manager"].flatten_usage(usage), "fetched_at": now,
    })
    return usage


@pytest.mark.parametrize("raw_refusal,expected", [
    pytest.param(_MISSING, _OLD_REFUSAL, id="missing"),
    pytest.param(None, None, id="explicit-null"),
    pytest.param("", _OLD_REFUSAL, id="empty-string"),
    pytest.param("  ", _OLD_REFUSAL, id="blank-string"),
    pytest.param({}, _OLD_REFUSAL, id="empty-object"),
    pytest.param({"type": None}, _OLD_REFUSAL, id="null-nested-type"),
    pytest.param({"type": ""}, _OLD_REFUSAL, id="empty-nested-type"),
    pytest.param(False, _OLD_REFUSAL, id="false"),
    pytest.param(0, _OLD_REFUSAL, id="zero"),
    pytest.param(_NEW_REFUSAL, _NEW_REFUSAL, id="new-refusal"),
    pytest.param({"type": _NEW_REFUSAL}, _NEW_REFUSAL, id="new-object-refusal"),
])
@pytest.mark.parametrize("allow_credits", [False, True])
def test_real_wham_parse_and_store_require_explicit_null(m, raw_refusal, expected, allow_credits):
    key, _ = _account(m, "real-refusal-provenance")
    manager = m["oauth_manager"]
    manager.get_account(key)["allowCredits"] = allow_credits
    now = int(time.time() * 1000)
    _save_headers(m, key, {"x-codex-rate-limit-reached-type": _OLD_REFUSAL}, now - 2000)
    parsed = _store_wham(m, key, now - 1000, raw_refusal)
    assert parsed["openai"]["rate_limit_reached_type_explicit_null"] is (raw_refusal is None)
    # A later unrelated credit fragment must not update the refusal's clock.
    _save_headers(m, key, {"x-codex-credits-balance": "9"}, now)
    row = m["state_db"].quota_load(key)
    assert row["codex_rate_limit_reached_type"] == _OLD_REFUSAL
    assert row["codex_rate_limit_reached_at"] == now - 2000
    usage = manager.usage_from_quota_row(row)
    assert usage["openai"]["rate_limit_reached_type"] == expected
    assert manager.openai_credits_usable(key, usage) is (allow_credits and expected is None)
    result = manager.evaluate_and_toggle_by_cached_quota(key)
    assert result["action"] == ("cached_below_threshold" if expected is None else "wham_limit_disabled")
    assert manager.get_account(key)["enabled"] is (expected is None)


@pytest.mark.parametrize("delta", [0, 1000])
@pytest.mark.parametrize("allow_credits", [False, True])
def test_explicit_null_cannot_clear_same_time_or_newer_response_refusal(m, delta, allow_credits):
    key, _ = _account(m, "newer-response-refusal")
    manager = m["oauth_manager"]
    manager.get_account(key)["allowCredits"] = allow_credits
    now = int(time.time() * 1000)
    _store_wham(m, key, now, None)
    _save_headers(m, key, {"x-codex-rate-limit-reached-type": _OLD_REFUSAL}, now + delta)
    usage = manager.usage_from_quota_row(m["state_db"].quota_load(key))
    assert usage["openai"]["rate_limit_reached_type"] == _OLD_REFUSAL
    assert not manager.openai_credits_usable(key, usage)
    assert manager.evaluate_and_toggle_by_cached_quota(key)["action"] == "wham_limit_disabled"


@pytest.mark.parametrize("raw_refusal,expected", [(None, _OLD_REFUSAL), (_NEW_REFUSAL, _NEW_REFUSAL)])
def test_legacy_cache_without_null_provenance_keeps_refusal(m, raw_refusal, expected):
    key, _ = _account(m, "legacy-refusal-provenance")
    manager = m["oauth_manager"]
    now = int(time.time() * 1000)
    _save_headers(m, key, {"x-codex-rate-limit-reached-type": _OLD_REFUSAL}, now - 1000)
    _store_wham(m, key, now, raw_refusal, legacy=True)
    usage = manager.usage_from_quota_row(m["state_db"].quota_load(key))
    assert usage["openai"]["rate_limit_reached_type"] == expected
    assert manager.evaluate_and_toggle_by_cached_quota(key)["action"] == "wham_limit_disabled"


@pytest.mark.parametrize("hard", ["spend", "overage"])
def test_explicit_null_does_not_clear_other_hard_limits(m, hard):
    key, _ = _account(m, "independent-hard-limit")
    manager = m["oauth_manager"]
    manager.get_account(key)["allowCredits"] = True
    now = int(time.time() * 1000)
    _save_headers(m, key, {"x-codex-rate-limit-reached-type": _OLD_REFUSAL}, now - 1000)
    _store_wham(m, key, now, None, hard=hard)
    usage = manager.usage_from_quota_row(m["state_db"].quota_load(key))
    assert usage["openai"]["rate_limit_reached_type"] is None
    assert not manager.openai_credits_usable(key, usage)
    assert manager.evaluate_and_toggle_by_cached_quota(key)["action"] == "wham_limit_disabled"
