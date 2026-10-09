"""Status-page fetches use real route selection with isolated fake wire I/O."""
from __future__ import annotations

import copy

import httpx
import pytest

from src import status_monitor
from src.proxy import manager as pm
from src.proxy.connector import ProxyStats


@pytest.fixture
def status_routes(monkeypatch):
    cfg = {
        "channels": [],
        "network": {
            "proxies": {name: {} for name in (
                "default-route", "claude-route", "openai-route", "legacy-route",
            )},
            "routing": {
                "default": "default-route",
                "directFallback": False,
                "providers": {"claude": "claude-route", "openai": "openai-route"},
            },
        },
    }
    calls = []
    failures = set()

    class WireConnector:
        type = "ss2022"

        def __init__(self, name):
            self.name = name
            self.stats = ProxyStats()

        def create_sync_httpx_client(self, **kwargs):
            def wire(request):
                calls.append((self.name, str(request.url)))
                if self.name in failures:
                    raise httpx.ConnectError("fixture proxy unavailable", request=request)
                return httpx.Response(200, json={"incidents": [{"id": self.name}]})

            return httpx.Client(
                transport=httpx.MockTransport(wire), trust_env=False, **kwargs,
            )

    monkeypatch.setattr(pm.config, "get", lambda: cfg)
    monkeypatch.setattr(pm.config, "on_reload", lambda _fn: None)
    monkeypatch.setattr(pm, "connector_from_config", lambda name, _cfg: WireConnector(name))
    monkeypatch.setattr(pm, "_snapshot", pm._EMPTY_SNAPSHOT)
    monkeypatch.setattr(pm, "_config_generation", None)
    monkeypatch.setattr(pm, "_callback_registered", False)
    monkeypatch.setattr(pm, "_initialized", False)
    pm.init()
    return cfg, calls, failures


@pytest.mark.parametrize("provider", ["claude", "openai"])
def test_status_uses_provider_route_not_default(status_routes, provider):
    _cfg, calls, _failures = status_routes
    route = provider + "-route"
    url = status_monitor.TARGETS[provider]["base"] + "/api/v2/incidents.json"
    assert status_monitor._fetch_incidents(provider) == [{"id": route}]
    assert calls == [(route, url)]


@pytest.mark.parametrize("provider,purpose", [
    ("claude", "oauth_anthropic"), ("openai", "oauth_openai"),
])
@pytest.mark.parametrize("legacy", [False, True])
def test_missing_provider_route_preserves_legacy_then_default(status_routes, provider, purpose, legacy):
    cfg, calls, _failures = status_routes
    routing = cfg["network"]["routing"]
    del routing["providers"][provider]
    if legacy:
        routing[purpose] = "legacy-route"
    pm._install_config_locked(copy.deepcopy(cfg))
    expected = "legacy-route" if legacy else "default-route"
    assert status_monitor._fetch_incidents(provider) == [{"id": expected}]
    assert calls[0][0] == expected


def test_cloudflare_status_keeps_existing_default_route(status_routes):
    _cfg, calls, _failures = status_routes
    assert status_monitor._fetch_incidents("cloudflare") == [{"id": "default-route"}]
    assert calls[0][0] == "default-route"


@pytest.mark.parametrize("provider", ["claude", "openai"])
def test_proxy_failure_does_not_try_default_or_direct(status_routes, provider):
    _cfg, calls, failures = status_routes
    route = provider + "-route"
    failures.add(route)
    assert status_monitor._fetch_incidents(provider) is None
    assert calls == [(route, status_monitor.TARGETS[provider]["base"] + "/api/v2/incidents.json")]
