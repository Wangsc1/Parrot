"""A recovered WHAM snapshot must not revive an older Codex refusal."""
from __future__ import annotations

import json

import pytest

from src.tests.test_openai_oauth_quota import _import_modules, _low_wham
from src.tests.test_codex_quota_fragments import _account


_OLD_REFUSAL = "workspace_member_credits_depleted"


def _row(*, active=2000, refusal=1000, explicit=True, active_refusal=None):
    usage = _low_wham()
    if explicit:
        usage["openai"]["rate_limit_reached_type"] = active_refusal
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
