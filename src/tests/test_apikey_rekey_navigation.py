"""Custom rekey cancellation and real Telegram navigation routing."""

from __future__ import annotations

import copy

import pytest

from src.telegram import bot, states, ui
from src.telegram.menus import apikey_menu
from src.tests.test_apikey_custom_key import _import_modules, _install_recorder, _setup


@pytest.fixture
def rekey(monkeypatch):
    modules = _import_modules()
    _setup(modules)
    recorder = _install_recorder(modules)
    modules["config"].update(lambda c: c.__setitem__("apiKeys", {
        "alpha": {"key": "old-alpha-key", "allowedModels": ["m1"], "allowImages": True},
    }))
    monkeypatch.setattr(ui, "is_admin", lambda chat_id: True)
    views = []
    monkeypatch.setattr(apikey_menu, "_edit_cached_detail", lambda *args, **kwargs: views.append((args, kwargs)) or True)
    short = apikey_menu._short_of("alpha")
    apikey_menu.on_rekey_enter(42, 100, "enter", short, page=2)
    before = copy.deepcopy(modules["config"].get()["apiKeys"])
    return modules, recorder, short, views, before


def callback(data, chat_id=42):
    bot._handle_callback({
        "id": "callback", "data": data,
        "from": {"id": chat_id},
        "message": {"message_id": 100, "chat": {"id": chat_id}},
    })


def message(text):
    bot._handle_message({"chat": {"id": 42}, "text": text})


def test_actual_cancel_button_exits_and_next_command_routes(rekey, monkeypatch):
    modules, recorder, short, views, before = rekey
    prompt = recorder.last("editMessageText")
    cancel = prompt["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    callback(cancel)
    assert states.get_state(42) is None
    assert views[-1] == ((42, 100, "alpha", 2), {"start_view": True})
    routed = []
    monkeypatch.setattr(bot.main_menu, "on_menu_command", lambda chat_id: routed.append(chat_id))
    message("/menu")
    assert routed == [42]
    assert modules["config"].get()["apiKeys"] == before
    assert not any("key 太短" in call.get("text", "") for call in recorder.by("sendMessage"))


def test_direct_apikey_callback_also_clears_rekey(rekey):
    modules, recorder, short, views, before = rekey
    assert apikey_menu.handle_callback(42, 100, "cancel", f"ak:view:{short}:2")
    assert states.get_state(42) is None
    assert modules["config"].get()["apiKeys"] == before


@pytest.mark.parametrize("command, target", [
    ("/menu", "menu"), ("/start", "start"), ("/help", "help"),
    ("/settings", "settings"), ("/menu@TestBot", "menu"),
])
def test_navigation_commands_never_become_new_key(rekey, monkeypatch, command, target):
    modules, recorder, short, views, before = rekey
    routed = []
    if target == "menu":
        monkeypatch.setattr(bot.main_menu, "on_menu_command", lambda chat_id: routed.append(chat_id))
    elif target == "start":
        monkeypatch.setattr(bot.main_menu, "on_start_command", lambda chat_id: routed.append(chat_id))
    elif target == "help":
        monkeypatch.setattr(bot.help_menu, "send_new", lambda chat_id: routed.append(chat_id))
    else:
        monkeypatch.setattr(bot.system_menu, "send_new", lambda chat_id: routed.append(chat_id))
    message(command)
    assert states.get_state(42) is None
    assert routed == [42]
    assert modules["config"].get()["apiKeys"] == before


@pytest.mark.parametrize("text", ["/cancel", "/cancel@TestBot"])
def test_cancel_command_clears_without_writing_key(rekey, text):
    modules, recorder, short, views, before = rekey
    message(text)
    assert states.get_state(42) is None
    assert modules["config"].get()["apiKeys"] == before
    assert "已取消" in recorder.last("sendMessage")["text"]


def test_main_menu_callback_exits_before_early_return(rekey, monkeypatch):
    modules, recorder, short, views, before = rekey
    routed = []
    monkeypatch.setattr(bot.main_menu, "handle_back", lambda *args: routed.append(args))
    callback("menu:main")
    assert routed == [(42, 100, "callback")]
    assert states.get_state(42) is None
    assert modules["config"].get()["apiKeys"] == before


def test_rejected_plain_key_retains_retry_state(rekey):
    modules, recorder, short, views, before = rekey
    message("short")
    assert states.get_state(42)["action"] == "ak_rekey_input"
    assert "key 太短" in recorder.last("sendMessage")["text"]
    assert modules["config"].get()["apiKeys"] == before
    message("new-alpha-key")
    assert states.get_state(42) is None
    assert modules["config"].get()["apiKeys"]["alpha"]["key"] == "new-alpha-key"
    assert modules["config"].get()["apiKeys"]["alpha"]["allowedModels"] == ["m1"]


@pytest.mark.parametrize("command", [
    "/unknown_command", "/unknown_command@TestBot", "/unknown_command argument",
])
def test_unknown_long_command_is_not_saved_as_key(rekey, command):
    modules, recorder, short, views, before = rekey
    message(command)
    assert states.get_state(42) is None
    assert modules["config"].get()["apiKeys"] == before


@pytest.mark.parametrize("api_key", [
    "/abc+/=123", "/settings+/=123", "/abc-def123", "/abc.def123", "/abc/def123",
])
def test_slash_prefixed_non_command_key_is_saved(rekey, api_key):
    modules, recorder, short, views, before = rekey
    assert apikey_menu._validate_custom_key(api_key, []) is None
    message(api_key)
    assert states.get_state(42) is None
    entry = modules["config"].get()["apiKeys"]["alpha"]
    for field, value in before["alpha"].items():
        assert entry[field] == (api_key if field == "key" else value)


@pytest.mark.parametrize("action", ["ak_add_key_input", "ak_sort", "ak_perm", "other_input"])
def test_hooks_do_not_clear_other_flows(rekey, action):
    states.set_state(42, action, {"name": "draft"})
    before = states.get_state(42)
    apikey_menu.before_callback(42, "menu:main")
    apikey_menu.before_command(42, "/menu")
    assert states.get_state(42) == before


def test_hooks_only_clear_target_chat(rekey):
    states.set_state(43, "ak_rekey_input", {"name": "other"})
    before = states.get_state(43)
    apikey_menu.before_command(42, "/menu")
    assert states.get_state(42) is None
    assert states.get_state(43) == before


def test_stale_cancel_does_not_clear_another_menu_state(rekey):
    modules, recorder, short, views, before = rekey
    states.set_state(42, "other_input", {"draft": "keep"})
    state = states.get_state(42)
    callback(f"ak:view:{short}:2")
    assert states.get_state(42) == state
    assert modules["config"].get()["apiKeys"] == before


def test_grant_notice_close_is_not_navigation(rekey, monkeypatch):
    state = states.get_state(42)
    monkeypatch.setattr(bot.zhipu_oauth_menu, "handle_callback", lambda *args: True)
    callback("oa:zh:notice_close")
    assert states.get_state(42) == state
