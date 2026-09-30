"""Subscription display labels preserve provider boundaries and raw account data."""
from __future__ import annotations

from copy import deepcopy

import pytest

from src.oauth.plan_labels import openai_plan_label, xai_plan_label


OPENAI_PLANS = [
    ("free", "Free"),
    ("go", "Go"),
    ("plus", "Plus"),
    ("prolite", "Pro 100"),
    ("pro", "Pro 200"),
    ("promax", "Pro 500"),
    ("chatgpt_pro", "Pro 200"),
    ("team", "Business"),
    ("self_serve_business_usage_based", "Business"),
    ("self_serve_business_prolite", "Business Premium"),
    ("business", "Enterprise"),
    ("enterprise", "Enterprise"),
    ("ent26", "Enterprise"),
    ("enterprise_cbp_usage_based", "Enterprise"),
    ("enterprise_cbp_automation", "Enterprise (Automation)"),
    ("edu", "Edu"),
    ("edu_plus", "Edu Plus"),
    ("edu_pro", "Edu Pro"),
    ("unknown", "Unknown"),
]


@pytest.mark.parametrize("raw,expected", OPENAI_PLANS)
def test_openai_subscription_labels(raw, expected):
    assert openai_plan_label(raw) == expected
    assert openai_plan_label(f"  {raw.upper()}  ") == expected
    assert openai_plan_label(raw.replace("_", "-")) == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_empty_subscription_label(raw):
    assert openai_plan_label(raw) == ""
    assert xai_plan_label(raw) == ""


@pytest.mark.parametrize("raw", ["future_sku", "Future-SKU", "<new_plan>"])
def test_unknown_subscription_codes_remain_readable(raw):
    assert openai_plan_label(raw) == raw
    assert xai_plan_label(raw) == raw


@pytest.mark.parametrize("raw,expected", [
    ("free", "Grok Free"),
    ("BASIC", "Grok Free"),
    ("pro", "Pro"),
    ("plus", "Plus"),
    ("team", "Team"),
    ("supergrok", "SuperGrok"),
    ("super_grok_lite", "SuperGrok Lite"),
    ("supergrokplus", "SuperGrok Plus"),
    ("super-grok-heavy", "SuperGrok Heavy"),
    ("heavy", "Heavy"),
    ("x_basic", "X Basic"),
])
def test_grok_labels_do_not_use_openai_names(raw, expected):
    assert xai_plan_label(raw) == expected


def _env(monkeypatch, account):
    from src.tests.test_tg_contract_oauth_support import FakeEnv
    return FakeEnv({
        "initialConfig": {"oauthAccounts": [account]},
        "initialRuntime": {},
    }, monkeypatch)


@pytest.mark.parametrize("raw,expected", OPENAI_PLANS + [("<future_sku>", "&lt;future_sku&gt;")])
def test_openai_list_detail_and_overwrite_display_only(monkeypatch, raw, expected):
    from src import oauth_manager
    from src.telegram.menus import oauth_menu

    account = {
        "provider": "openai", "email": "labels@fake.invalid",
        "workspace_id": "workspace-labels", "chatgpt_account_id": "workspace-labels",
        "workspace_name": "Personal", "plan_type": raw, "enabled": True,
        "models": [], "expired": "2030-01-01T00:00:00Z",
    }
    env = _env(monkeypatch, account)
    original = deepcopy(env.cfg)
    key = oauth_manager.get_account_key(account)

    listing = oauth_menu._format_account_block(account, month_snapshot={})
    detail, _ = oauth_menu._detail_text_and_kb(key, month_snapshot={}, model_stats=[])
    overwrite = oauth_menu._overwrite_summary(account)
    assert f"套餐: <code>{expected}</code>" in listing
    assert f"套餐: <code>{expected}</code>" in detail
    assert f"套餐: <code>{expected}</code>" in overwrite
    assert env.cfg == original
    assert account["plan_type"] == raw


@pytest.mark.parametrize("raw,expected", OPENAI_PLANS)
def test_openai_notification_labels_preserve_workspace_and_sku(raw, expected):
    from src import oauth_manager

    account = {"plan_type": raw, "workspace_name": "AU", "workspace_id": "private-id"}
    original = deepcopy(account)
    assert oauth_manager.openai_plan_workspace_label(account) == f"OpenAI · {expected}（AU）"
    assert account == original


def test_grok_code_and_display_name_are_deduplicated(monkeypatch):
    from src.telegram.menus import oauth_menu

    value = {
        "user": {"subscription_tier": "supergrokheavy"},
        "settings": {"subscription_tier_display": "SuperGrok Heavy"},
    }
    original = deepcopy(value)
    assert oauth_menu._xai_tier_label(value) == "SuperGrok Heavy"
    assert value == original


@pytest.mark.parametrize("tier,expected", [
    ("default_claude_ai", "Free"),
    ("default_claude_pro", "Pro"),
    ("default_claude_max_5x", "Max 5x"),
    ("default_claude_max_20x", "Max 20x"),
])
def test_claude_existing_tiers_keep_their_own_labels(tier, expected):
    from src import oauth_manager
    assert oauth_manager.claude_plan_label({"rate_limit_tier": tier}) == expected


@pytest.mark.parametrize("tier,expected", [
    ("g1-pro-tier", "Google AI Pro"),
    ("g1-ultra-tier", "Google AI Ultra"),
    ("standard-tier", "Antigravity"),
    ("free-tier", "免费档"),
])
def test_google_existing_tiers_prefer_official_name(tier, expected):
    from src.oauth import antigravity
    assert antigravity.credits_tier_label({"tier": tier}) == expected
    assert antigravity.credits_tier_label({"tier": tier, "tier_name": "Official plan name"}) == "Official plan name"
