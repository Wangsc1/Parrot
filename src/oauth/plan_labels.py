"""Provider-specific subscription labels; wire values remain unchanged."""
from __future__ import annotations

import re


_OPENAI_PLAN_LABELS = {
    "free": "Free",
    "go": "Go",
    "plus": "Plus",
    "prolite": "Pro 100",
    "pro": "Pro 200",
    "chatgptpro": "Pro 200",
    "promax": "Pro 500",
    "team": "Business",
    "selfservebusinessusagebased": "Business",
    "selfservebusinessprolite": "Business Premium",
    "business": "Enterprise",
    "enterprise": "Enterprise",
    "ent26": "Enterprise",
    "enterprisecbpusagebased": "Enterprise",
    "enterprisecbpautomation": "Enterprise (Automation)",
    "edu": "Edu",
    "eduplus": "Edu Plus",
    "edupro": "Edu Pro",
    "unknown": "Unknown",
}

_XAI_PLAN_LABELS = {
    "free": "Grok Free",
    "basic": "Grok Free",
    "pro": "Pro",
    "plus": "Plus",
    "team": "Team",
    "supergrok": "SuperGrok",
    "supergroklite": "SuperGrok Lite",
    "supergrokplus": "SuperGrok Plus",
    "supergrokheavy": "SuperGrok Heavy",
    "heavy": "Heavy",
    "xbasic": "X Basic",
}


def _label(value: object, labels: dict[str, str]) -> str:
    raw = str(value or "").strip()
    key = re.sub(r"[\s_-]+", "", raw.casefold())
    return labels.get(key, raw)


def openai_plan_label(value: object) -> str:
    """Display an OpenAI subscription SKU without merging its stored identity."""
    return _label(value, _OPENAI_PLAN_LABELS)


def xai_plan_label(value: object) -> str:
    """Display a Grok subscription code independently of OpenAI naming."""
    return _label(value, _XAI_PLAN_LABELS)
