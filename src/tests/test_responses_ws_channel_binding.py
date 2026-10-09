"""Live API-key source grants on sequential Responses WebSocket requests."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from src import auth, config, scheduler
from src.channel import registry
from src.management_auth import Capability
from src.management_control.apikey import ApiKeyControl
from src.tests import test_openai_responses_ws as ws_helpers
from src.tests.test_management_apikey_control import FakeLimiter, FakeStats, context
from src.tests.test_openai_responses_ws import _isolate_ws_config


def _done(response_id):
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.completed", "response": {
            "id": response_id, "output": [],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["native-ws", "codex-http-fallback"])
@pytest.mark.parametrize("initial,change,change_event,denied", [
    pytest.param("missing", "bind-b", "response.completed", True, id="enable-after-shared"),
    pytest.param("bound-a", "bind-b", "response.completed", True, id="rebind-after-bound"),
    pytest.param("bound-a", "bind-b", "response.created", True, id="keep-inflight-request"),
    pytest.param("missing", None, "response.completed", False, id="legacy-missing"),
    pytest.param("empty", None, "response.completed", False, id="legacy-empty"),
    pytest.param("bound-a", "disable-b", "response.completed", False, id="disable-retains-selection"),
    pytest.param("bound-a", None, "response.completed", False, id="still-authorized"),
])
async def test_sequential_turn_checks_current_channel_binding(
    monkeypatch, transport, initial, change, change_event, denied,
):
    m = ws_helpers._import_modules()
    cfg = ws_helpers._setup(m)
    cfg["channelSelection"] = "order"
    cfg["apiKeyConcurrency"] = {"enabled": False}
    cfg["concurrency"]["enabled"] = False
    if transport == "native-ws":
        a = ws_helpers._make_channel(m, extra={"name": "a"})
        b = ws_helpers._make_channel(m, extra={"name": "b"})
    else:
        a = ws_helpers._make_oauth_channel_for_failover(m, name="a@example.test")
        account_a = cfg["oauthAccounts"][0]
        b = ws_helpers._make_oauth_channel_for_failover(m, name="b@example.test")
        cfg["oauthAccounts"].append(account_a)
    monkeypatch.setattr(registry, "_channels", {a.key: a, b.key: b})
    entry = cfg["apiKeys"]["ws-key"]
    if initial == "bound-a":
        entry.update(allowedChannels=[a.key], channelBindingEnabled=True)
    elif initial == "empty":
        entry["allowedChannels"] = []
    control = ApiKeyControl(
        config_store=config, statistics=FakeStats(), limiter=FakeLimiter(),
    )
    ctx = context(Capability.READ, Capability.WRITE)
    changed = False

    class ReconfiguringWebSocket(ws_helpers.SequentialFakeWebSocket):
        async def send_text(self, text):
            nonlocal changed
            await super().send_text(text)
            if change and not changed and json.loads(text).get("type") == change_event:
                # Exercise the real control/persistence path, not a cached mock
                # authorization result. The next create has not been sent yet.
                control.update_api_key(ctx, "ws-key", changes={
                    "allowed_channels": [b.key],
                    "channel_binding_enabled": change == "bind-b",
                })
                changed = True
                assert auth.channel_allowed("ws-key", a.key) is not denied
                fresh = scheduler.schedule(
                    {"model": "test-model", "input": "new"}, "ws-key", "1.2.3.4",
                    ingress_protocol="responses",
                )
                assert [ch.key for ch, _ in fresh.candidates] == (
                    [b.key] if denied else [a.key, b.key]
                )

    ws = ReconfiguringWebSocket(
        {"type": "response.create", "model": "test-model", "input": "first"},
        {"type": "response.create", "model": "test-model", "input": "second"},
    )
    connects = []
    http_requests = []
    if transport == "native-ws":
        upstream = ws_helpers.FakeUpstreamWebSocket(_done("first") + _done("second"))

        async def connect(url, **kwargs):
            connects.append(url)
            return upstream

        monkeypatch.setattr(m["responses_ws"], "_connect_upstream_ws", connect)
    else:
        from websockets.exceptions import InvalidStatus
        from src.transports.http_runtime import OpenedHttpResponse

        class Stream(httpx.AsyncByteStream):
            def __init__(self, events):
                self.events = events

            async def __aiter__(self):
                for event in self.events:
                    yield (f"event: {event['type']}\ndata: {json.dumps(event)}\n\n").encode()

        class Context:
            async def __aexit__(self, *args):
                return None

        async def connect(url, **kwargs):
            connects.append(url)
            raise InvalidStatus(SimpleNamespace(
                status_code=426, headers={}, body=b"upgrade unavailable",
            ))

        async def opened(**kwargs):
            raw = kwargs["upstream_req"].body
            http_requests.append(json.loads(raw) if isinstance(raw, (str, bytes)) else raw)
            response_id = "first" if len(http_requests) == 1 else "second"
            response = httpx.Response(
                200, stream=Stream(_done(response_id)),
                request=httpx.Request("POST", "https://example.test/responses"),
            )
            return OpenedHttpResponse(ctx=Context(), response=response, connect_ms=1)

        monkeypatch.setattr(m["responses_ws"], "_connect_upstream_ws", connect)
        monkeypatch.setattr(m["responses_ws"], "open_response_with_proxy_chain", opened)

    conn = m["log_db"]._get_conn()
    before = conn.execute("SELECT COALESCE(MAX(id), 0) FROM request_log").fetchone()[0]
    await m["responses_ws"].handle_responses_ws(ws)
    rows = conn.execute(
        "SELECT status, final_channel_key, input_tokens, output_tokens "
        "FROM request_log WHERE id>? ORDER BY id", (before,),
    ).fetchall()
    expected_turns = 1 if denied else 2
    wire = [json.loads(raw) for raw in upstream.sent] if transport == "native-ws" else http_requests
    assert len(wire) == expected_turns
    assert "first" in json.dumps(wire[0]["input"])
    if not denied:
        assert "second" in json.dumps(wire[1]["input"])
    # Never reconnect or migrate the fixed session to B, even though a fresh
    # request could use B. Rejected creates must not create paid attempt rows.
    assert len(connects) == 1
    assert len(rows) == expected_turns
    assert all(row["status"] == "success" and row["final_channel_key"] == a.key for row in rows)
    assert [(row["input_tokens"], row["output_tokens"]) for row in rows] == [(2, 1)] * expected_turns
    events = [json.loads(text) for text in ws.sent_texts]
    completed = [e for e in events if e.get("type") == "response.completed"]
    assert len(completed) == expected_turns
    assert completed[0]["response"]["id"] == "first"
    errors = [e for e in events if e.get("type") == "error"]
    if denied:
        assert len(errors) == 1
        assert errors[0]["status"] == 403
        assert errors[0]["error"]["code"] == "permission_denied"
        assert "no longer allowed" in errors[0]["error"]["message"]
    else:
        assert errors == []
    assert not ws.close_calls
    assert changed is bool(change)
    if change == "disable-b":
        assert config.get()["apiKeys"]["ws-key"]["allowedChannels"] == [b.key]
        assert auth.allowed_channels("ws-key") is None
