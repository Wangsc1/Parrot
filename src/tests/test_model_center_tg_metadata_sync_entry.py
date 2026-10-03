"""Regression checks for the metadata-sync entry and its CAS domain."""
from __future__ import annotations

from src.tests import _isolation
_isolation.isolate()

from unittest.mock import Mock

import pytest

from src.management_control import ManagementError
from src.management_control.errors import ManagementErrorCode
from src.management_control.mapping.control import MappingControl
from src.telegram import ui
from src.telegram.menus import model_center_menu as menu
from src.tests.test_model_center_tg_settings import env, _button


@pytest.fixture
def sync_env(env, monkeypatch):
    control, edits, answers, sends = env
    control.mapping.metadata_revision = "metadata-1"
    control.mapping.get_metadata_revision = lambda ctx: control.mapping.metadata_revision
    requests = []

    def api(method, data):
        requests.append((method, data))
        return {"ok": True, "result": {"message_id": 11}}

    monkeypatch.setattr(ui, "api", api)
    return control, edits, answers, requests


def test_full_sync_button_uses_metadata_not_catalog_revision(sync_env):
    control, _, _, _ = sync_env
    text, kb = menu._metadata_sync_render(7, "mc:list")
    action = menu._thaw(7, _button(kb, "同步全部元数据")["callback_data"].split(":")[-1])
    assert action.data["revision"] == "metadata-1"
    assert action.data["revision"] != control.mapping.catalog_revision
    assert "cat1" in text  # Catalog label remains a catalog label.
    assert _button(kb, "返回模型列表")["callback_data"] == "mc:list"


def test_entry_sends_plaintext_new_page_without_editing_list(sync_env):
    _, edits, answers, requests = sync_env
    _, kb = menu.render(7)
    menu.handle_callback(7, 10, "open", _button(kb, "同步元数据")["callback_data"])
    assert len(requests) == 1
    method, data = requests[0]
    assert method == "sendMessage" and data["chat_id"] == 7
    assert "parse_mode" not in data
    assert "元数据同步" in data["text"] and "人工匹配" in data["text"]
    assert "<b>" not in data["text"] and "<code>" not in data["text"]
    assert len(data["reply_markup"]["inline_keyboard"]) == 2
    assert edits == [] and answers == [("open", None, False)]


def test_new_page_button_starts_full_sync_with_frozen_metadata_revision(sync_env):
    control, _, _, requests = sync_env
    callback = menu._freeze(7, "metadata_sync", back_callback="mc:list")
    menu.handle_callback(7, 10, "open", callback)
    kb = requests[-1][1]["reply_markup"]
    menu.handle_callback(7, 11, "start", _button(kb, "同步全部元数据")["callback_data"])
    call = control.mapping.sync_calls[-1]
    assert call["mode"].value == "full" and call["targets"] == ()
    assert call["source"] is None and call["refresh_catalog"] is True
    assert call["expected_revision"] == "metadata-1"


def test_stale_new_page_keeps_revision_conflict_protection(sync_env, monkeypatch):
    control, _, answers, _ = sync_env
    _, kb = menu._metadata_sync_render(7)
    control.mapping.metadata_revision = "metadata-2"

    def start(ctx, **kwargs):
        MappingControl._check_revision(kwargs["expected_revision"], control.mapping.metadata_revision)
        pytest.fail("stale page must not start a worker")

    monkeypatch.setattr(control.mapping, "start_metadata_sync", start)
    menu.handle_callback(7, 11, "stale", _button(kb, "同步全部元数据")["callback_data"])
    assert any(alert and "页面版本已变化" in (text or "") for _, text, alert in answers)


def test_metadata_revision_reader_authorizes_and_uses_sync_domain(monkeypatch):
    control = MappingControl()
    context = object()
    read = Mock(return_value=context)
    revision = Mock(return_value="sync-domain")
    monkeypatch.setattr(control, "_read", read)
    monkeypatch.setattr(control, "_metadata_revision", revision)
    assert control.get_metadata_revision(context) == "sync-domain"
    read.assert_called_once_with(context)
    revision.assert_called_once_with()


def test_metadata_revision_reader_does_not_read_after_permission_denial(monkeypatch):
    control = MappingControl()
    revision = Mock()
    monkeypatch.setattr(control, "_read", Mock(side_effect=ManagementError(ManagementErrorCode.CAPABILITY_DENIED)))
    monkeypatch.setattr(control, "_metadata_revision", revision)
    with pytest.raises(ManagementError):
        control.get_metadata_revision(object())
    revision.assert_not_called()
