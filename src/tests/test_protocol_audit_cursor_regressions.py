"""C-01/C-02 local adapter regressions, NOT official Cursor wire fixtures.

Exercise CursorClient's session selection and inspect the actual serialized Run
request. Scripted events only replace upstream I/O; no real account is used.
"""
from __future__ import annotations

import json
from collections import deque

import pytest

from src.cursor_bridge import agent_pb2
from src.cursor_bridge.client import ConversationState, CursorClient, _final_usage
from src.cursor_bridge.constants import MAX_EFFECTIVE_PROMPT_BYTES
from src.cursor_bridge.runtime import CursorBridgeRuntime
from src.cursor_bridge.session import CursorSession, SessionEvent
from src.cursor_bridge.tool_dispatch import PendingExec


HISTORY = [
    {"role": "system", "content": "SYSTEM_ONE"},
    {"role": "developer", "content": "DEVELOPER_TWO"},
    {"role": "system", "content": "SYSTEM_THREE"},
    {"role": "user", "content": "QUESTION_ONE"},
    {"role": "assistant", "content": "CALL_TEXT", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "Read", "arguments": '{"path":"x"}'}}
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "TOOL_RESULT_UNIQUE"},
    {"role": "assistant", "content": "SECOND_ASSISTANT"},
    {"role": "user", "content": "QUESTION_TWO"},
]


def pending(call_id="c1"):
    return PendingExec(exec_id="exec-" + call_id, exec_msg_id=1,
                       tool_call_id=call_id, tool_name="Read", decoded_args='{"path":"x"}')


class LocalSession:
    def __init__(self, events=(), **kwargs):
        self.kwargs = kwargs
        self.events = deque(events)
        self.alive = True
        self.pending_execs = []
        self.results = []

    def next(self, timeout=None):
        event = self.events.popleft()
        if event.exec is not None:
            self.pending_execs.append(event.exec)
        return event

    def send_tool_results(self, results):
        self.results.append(results)
        matched = {item["tool_call_id"] for item in results}
        self.pending_execs = [item for item in self.pending_execs if item.tool_call_id not in matched]

    def close(self):
        self.alive = False


def local_client(*event_batches):
    sessions = []
    batches = deque(event_batches)

    def factory(**kwargs):
        session = LocalSession(batches.popleft() if batches else (), **kwargs)
        sessions.append(session)
        return session

    return CursorClient("local-not-a-token", session_factory=factory, sleeper=lambda _: None), sessions


def request(session):
    msg = agent_pb2.AgentClientMessage()
    msg.ParseFromString(session.kwargs["request_bytes"])
    return msg.run_request


def context(session):
    return request(session).action.user_message_action.request_context.cloud_rule


def action_text(session):
    return request(session).action.user_message_action.user_message.text


def json_messages(text):
    # The bridge's documented textual compatibility representation, not a
    # claim about an upstream native history/checkpoint schema.
    return json.loads(text[text.index("\n[") + 1:])


def test_c01_full_history_reaches_final_request_in_order():
    client, sessions = local_client()
    try:
        client.chat_completions(model="local-model", messages=HISTORY)
        raw = sessions[0].kwargs["request_bytes"]
        for marker in ("SYSTEM_ONE", "DEVELOPER_TWO", "SYSTEM_THREE", "QUESTION_ONE",
                       "CALL_TEXT", "TOOL_RESULT_UNIQUE", "SECOND_ASSISTANT", "QUESTION_TWO"):
            assert marker.encode() in raw, marker
        assert json_messages(context(sessions[0])) == HISTORY[:-1]
        assert action_text(sessions[0]) == "QUESTION_TWO"
        assert "QUESTION_TWO" not in context(sessions[0])
        assert sessions[0].kwargs["cloud_rule"] == context(sessions[0])
    finally:
        client.close()


def test_c01_rebuild_preserves_consecutive_roles_empty_content_and_tool_arguments():
    messages = [
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "one", "type": "function", "function": {"name": "Read", "arguments": '{ "path": "a" }'}},
            {"id": "two", "type": "function", "function": {"name": "Read", "arguments": '{"path":"b"}'}}
        ]},
        {"role": "tool", "tool_call_id": "two", "content": ""},
        {"role": "tool", "tool_call_id": "one", "content": "User: not a role\nAssistant: not a role"},
        {"role": "assistant", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
        {"role": "assistant", "content": "another"},
        {"role": "user", "content": "continue"},
    ]
    client, sessions = local_client()
    try:
        client.chat_completions(model="local-model", messages=messages)
        assert json_messages(context(sessions[0])) == messages[:-1]
        assert action_text(sessions[0]) == "continue"
    finally:
        client.close()


def test_c01_long_rebuild_does_not_silently_truncate_oldest_or_compaction():
    messages = [
        {"role": "system", "content": "EARLY_RULE"},
        {"role": "user", "content": "EARLY_USER" + "x" * MAX_EFFECTIVE_PROMPT_BYTES},
        {"role": "assistant", "content": "EARLY_ANSWER"},
        {"role": "user", "content": "The conversation history before this point was compacted into the following summary:\nSUMMARY"},
        {"role": "assistant", "content": "SUMMARY_ACK"},
        {"role": "user", "content": "last"},
    ]
    client, sessions = local_client()
    try:
        client.chat_completions(model="local-model", messages=messages)
        assert json_messages(context(sessions[0])) == messages[:-1]
    finally:
        client.close()


@pytest.mark.parametrize("checkpoint", [None, b"\xff", agent_pb2.ConversationStateStructure(turns=[b"opaque"]).SerializeToString()])
def test_c01_missing_or_invalid_checkpoint_rebuilds_tool_ended_request(checkpoint):
    client, sessions = local_client()
    messages = HISTORY[:6]
    try:
        client.chat_completions(model="local-model", messages=[{"role": "user", "content": "initial"}], session_id="stable")
        state = client._conversations["stable"]
        sessions[0].close()
        state.checkpoint = checkpoint
        client.chat_completions(model="local-model", messages=messages, session_id="stable")
        assert len(sessions) == 2
        assert json_messages(context(sessions[-1])) == messages[:-1]
        assert json_messages(action_text(sessions[-1])) == messages[-1:]
        assert "QUESTION_ONE" not in action_text(sessions[-1])
    finally:
        client.close()


def test_c01_checkpoint_continuation_sends_only_current_action():
    client, sessions = local_client([SessionEvent(type="text", text="answer"), SessionEvent(type="done")])
    try:
        client.chat_completions(model="local-model", messages=HISTORY, session_id="stable", stream=False)
        checkpoint = agent_pb2.ConversationStateStructure(turns=[b"opaque-upstream-blob-reference"])
        sessions[0].kwargs["on_checkpoint"](checkpoint.SerializeToString())
        messages = HISTORY + [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "NEW_USER"}]
        client.chat_completions(model="local-model", messages=messages, session_id="stable")
        rebuilt = request(sessions[-1])
        assert rebuilt.conversation_state == checkpoint
        assert action_text(sessions[-1]) == "NEW_USER"
        assert "QUESTION_ONE" not in context(sessions[-1])
        assert "TOOL_RESULT_UNIQUE" not in context(sessions[-1])
        assert "DEVELOPER_TWO" in context(sessions[-1])
        assert "SYSTEM_THREE" in context(sessions[-1])
    finally:
        client.close()


@pytest.mark.parametrize("stream", [False, True])
def test_c01_live_tool_result_resumes_same_session_without_replaying_history(stream):
    events = [SessionEvent(type="toolCall", exec=pending()), SessionEvent(type="batchReady"),
              SessionEvent(type="text", text="resumed"), SessionEvent(type="done")]
    client, sessions = local_client(events)
    try:
        first = client.chat_completions(model="local-model", messages=HISTORY[:4], session_id="stable", stream=stream)
        if stream:
            list(first)
        second = client.chat_completions(model="local-model", messages=HISTORY[:6], session_id="stable", stream=stream)
        if stream:
            list(second)
        assert len(sessions) == 1
        assert sessions[0].results == [[{"tool_call_id": "c1", "content": "TOOL_RESULT_UNIQUE", "is_error": False}]]
    finally:
        client.close()


def test_c01_old_tool_results_do_not_resume_a_live_session_for_new_user():
    client, sessions = local_client()
    try:
        client.chat_completions(model="local-model", messages=HISTORY[:4], session_id="stable")
        sessions[0].pending_execs = [pending()]
        client.chat_completions(model="local-model", messages=HISTORY, session_id="stable")
        assert len(sessions) == 2
        assert sessions[0].results == []
        assert action_text(sessions[-1]) == "QUESTION_TWO"
    finally:
        client.close()


def test_c01_runtime_pins_only_terminal_tool_results_and_scopes_account():
    runtime = CursorBridgeRuntime()
    runtime.register_tool_call("account", "paused", "c1")
    assert runtime.session_for("account", {"messages": HISTORY[:6]}) == "paused"
    assert runtime.session_for("account", {"messages": HISTORY}) != "paused"
    assert runtime.session_for("other", {"messages": HISTORY[:6]}) != "paused"


def test_c01_blob_not_found_retry_rebuilds_full_history():
    client, sessions = local_client([SessionEvent(type="done", error="blob not found")],
                                    [SessionEvent(type="text", text="ok"), SessionEvent(type="done")])
    try:
        checkpoint = agent_pb2.ConversationStateStructure(turns=[b"lost-upstream-blob"])
        client._conversations["stable"] = ConversationState(checkpoint=checkpoint.SerializeToString())
        result = client.chat_completions(model="local-model", messages=HISTORY, session_id="stable", stream=False)
        assert request(sessions[0]).conversation_state == checkpoint
        assert "QUESTION_ONE" not in context(sessions[0])
        assert result["choices"][0]["message"]["content"] == "ok"
        assert len(sessions) == 2
        assert request(sessions[0]).conversation_id != request(sessions[1]).conversation_id
        assert json_messages(context(sessions[1])) == HISTORY[:-1]
    finally:
        client.close()


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
])
def test_c02_reported_usage_is_not_overwritten_by_text_estimate(usage):
    original = dict(usage)
    assert _final_usage(usage, ["x" * 600]) == original
    assert usage == original


@pytest.mark.parametrize(("usage", "expected"), [
    (None, None), ({}, None),
    ({"completion_tokens": 7}, {"completion_tokens": 7}),
    ({"total_tokens": 100}, {"total_tokens": 100}),
    ({"prompt_tokens": 80, "completion_tokens": 20}, {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}),
    ({"total_tokens": 100, "completion_tokens": 20}, {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}),
])
def test_c02_missing_partial_usage_never_invents_exact_counts(usage, expected):
    assert _final_usage(usage, ["x" * 600]) == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reported", [True, False])
def test_c02_final_response_usage_matches_observed_events(stream, reported):
    events = [SessionEvent(type="text", text="x" * 600)]
    if reported:
        events.append(SessionEvent(type="usage", output_tokens=20, total_tokens=100))
    events.append(SessionEvent(type="done"))
    client, _ = local_client(events)
    try:
        result = client.chat_completions(model="local-model", messages=[{"role": "user", "content": "hi"}], stream=stream)
        usages = [chunk["usage"] for chunk in result if "usage" in chunk] if stream else ([result["usage"]] if "usage" in result else [])
        assert usages == ([{"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}] if reported else [])
    finally:
        client.close()


def test_c01_simple_system_user_control_keeps_existing_wire_text():
    client, sessions = local_client()
    try:
        client.chat_completions(model="local-model", messages=[
            {"role": "system", "content": "rules"}, {"role": "user", "content": "question"}])
        assert context(sessions[0]) == "rules"
        assert action_text(sessions[0]) == "question"
    finally:
        client.close()


def test_c01_unknown_terminal_result_is_rebuilt_instead_of_silently_ignored():
    client, sessions = local_client([
        SessionEvent(type="toolCall", exec=pending()), SessionEvent(type="batchReady"),
    ])
    try:
        # Consume the first HTTP turn to its real pause boundary before replying.
        list(client.chat_completions(model="local-model", messages=HISTORY[:4], session_id="stable"))
        messages = HISTORY[:6] + [{"role": "tool", "tool_call_id": "other", "content": "UNKNOWN_RESULT"}]
        client.chat_completions(model="local-model", messages=messages, session_id="stable")
        assert len(sessions) == 2
        assert sessions[0].results == []
        assert json_messages(action_text(sessions[-1])) == messages[-2:]
    finally:
        client.close()


def test_c01_parallel_results_can_arrive_in_parts_without_resending_answered_ids():
    client, sessions = local_client([
        SessionEvent(type="toolCall", exec=pending()),
        SessionEvent(type="toolCall", exec=pending("c2")), SessionEvent(type="batchReady"),
        SessionEvent(type="batchReady"),
    ])
    try:
        list(client.chat_completions(model="local-model", messages=HISTORY[:4], session_id="stable"))
        list(client.chat_completions(model="local-model", messages=HISTORY[:6], session_id="stable"))
        messages = HISTORY[:6] + [{"role": "tool", "tool_call_id": "c2", "content": "SECOND_RESULT"}]
        client.chat_completions(model="local-model", messages=messages, session_id="stable")
        assert len(sessions) == 1
        assert [[item["tool_call_id"] for item in batch] for batch in sessions[0].results] == [["c1"], ["c2"]]
    finally:
        client.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("output", "total", "expected"), [
    (None, None, []),
    (7, None, [{"completion_tokens": 7}]),
    (None, 100, [{"total_tokens": 100}]),
    (20, 100, [{"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}]),
    (0, 0, [{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}]),
])
def test_c02_wire_usage_presence_survives_real_session_decoder(monkeypatch, stream, output, total, expected):
    from src import network
    from src.cursor_bridge.connect import frame_connect_message
    from src.cursor_bridge.constants import CONNECT_END_STREAM_FLAG
    from .test_cursor_proxy_lifecycle import ScriptedH2Stream

    frames = [agent_pb2.AgentServerMessage(interaction_update=agent_pb2.InteractionUpdate(
        text_delta=agent_pb2.TextDeltaUpdate(text="x" * 600)))]
    if output is not None:
        frames.append(agent_pb2.AgentServerMessage(interaction_update=agent_pb2.InteractionUpdate(
            token_delta=agent_pb2.TokenDeltaUpdate(tokens=output))))
    if total is not None:
        frames.append(agent_pb2.AgentServerMessage(conversation_checkpoint_update=agent_pb2.ConversationStateStructure(
            token_details=agent_pb2.ConversationTokenDetails(used_tokens=total))))
    wire = b"".join(frame_connect_message(frame.SerializeToString()) for frame in frames)
    wire += frame_connect_message(b"{}", flags=CONNECT_END_STREAM_FLAG)
    routed = ScriptedH2Stream(mode="session", response=wire)
    monkeypatch.setattr(network, "open_sync_stream", lambda *args, **kwargs: routed)
    client = CursorClient("local-not-a-token", max_retries=0, request_timeout_s=2)
    try:
        result = client.chat_completions(model="local-model", messages=[{"role": "user", "content": "hi"}], stream=stream)
        usages = [chunk["usage"] for chunk in result if "usage" in chunk] if stream else ([result["usage"]] if "usage" in result else [])
        assert usages == expected
    finally:
        client.close()


def test_c01_live_tool_reply_uses_original_exec_wire_ids_and_content():
    from queue import Queue
    from types import SimpleNamespace
    from src.cursor_bridge.connect import ConnectFrameParser

    # Bypass only transport startup; invoke production send_tool_results.
    session = object.__new__(CursorSession)
    sent = []
    session._stream = SimpleNamespace(write=sent.append)
    session.events = Queue()
    session.pending_execs = [pending()]
    session._timer_phase = "streaming"
    session.send_tool_results([{"tool_call_id": "c1", "content": "TOOL_RESULT_UNIQUE", "is_error": False}])
    decoded = []
    parser = ConnectFrameParser(lambda raw: decoded.append(agent_pb2.AgentClientMessage.FromString(raw)), lambda _: None)
    for frame in sent:
        parser.feed(frame)
    assert len(decoded) == 2
    reply = decoded[0].exec_client_message
    assert reply.id == 1 and reply.exec_id == "exec-c1"
    assert reply.mcp_result.success.content[0].text.text == "TOOL_RESULT_UNIQUE"
    assert not reply.mcp_result.success.is_error
    assert decoded[1].exec_client_control_message.stream_close.id == 1
    assert session.pending_execs == []
