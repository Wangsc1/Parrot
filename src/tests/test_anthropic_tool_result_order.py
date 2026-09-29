"""Compaction summaries must not precede tool results in an Anthropic user turn."""
from __future__ import annotations

import copy

import pytest

from src.openai.transform import chat_to_anthropic, responses_to_anthropic
from src.transform.cc_mimicry import transform_request


@pytest.fixture(params=["responses", "chat"])
def protocol(request):
    return request.param


def _calls(protocol, *ids):
    if protocol == "responses":
        return [{"type": "function_call", "call_id": call_id, "name": "shell_execute",
                 "arguments": '{"command":"echo ok"}'} for call_id in ids]
    return [{"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {
            "name": "shell_execute", "arguments": '{"command":"echo ok"}',
        }} for call_id in ids
    ]}]


def _result(protocol, call_id, output):
    if protocol == "responses":
        return {"type": "function_call_output", "call_id": call_id, "output": output}
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def _translate(protocol, items):
    tool = {"name": "shell_execute", "parameters": {"type": "object"}}
    body = {"model": "claude-opus-5-5"}
    if protocol == "responses":
        body.update(input=items, tools=[{"type": "function", **tool}])
        translate = responses_to_anthropic.translate_request
    else:
        body.update(messages=items, tools=[{"type": "function", "function": tool}])
        translate = chat_to_anthropic.translate_request
    original = copy.deepcopy(body)
    payload = translate(body)
    assert body == original
    return payload


@pytest.mark.parametrize("summary_length", [7, 23000])
def test_compaction_summary_follows_result_through_oauth_payload(protocol, summary_length):
    summary = "<context-summary>" + "摘" * summary_length + "</context-summary>"
    payload = _translate(protocol, [
        {"role": "user", "content": "Run a check"},
        *_calls(protocol, "call_a"),
        {"role": "user", "content": summary},
        _result(protocol, "call_a", "ok"),
    ])
    expected = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_a", "content": "ok"},
        {"type": "text", "text": summary},
    ]}
    assert len(payload["messages"]) == 3
    assert payload["messages"][2] == expected
    assert payload["messages"][1]["content"][0]["id"] == "call_a"
    # Check the actual Claude OAuth body-shaping stage, not only the bridge.
    wire, _ = transform_request(
        {**payload, "model": "claude-opus-5-5"},
        auth_mode="oauth", session_id="tool-result-order-test",
    )
    # OAuth may add cache hints; compare the complete ordered content without
    # treating that existing transport metadata as a tool-history change.
    assert wire["messages"][2]["role"] == expected["role"]
    assert [{k: v for k, v in block.items() if k != "cache_control"}
            for block in wire["messages"][2]["content"]] == expected["content"]


def test_parallel_results_and_other_blocks_keep_their_relative_order(protocol):
    image = ({"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
             if protocol == "responses" else
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})
    text_type = "input_text" if protocol == "responses" else "text"
    output = [{"type": text_type, "text": "result b"}, image]
    payload = _translate(protocol, [
        {"role": "user", "content": "start"},
        *_calls(protocol, "call_a", "call_b"),
        {"role": "user", "content": "summary before results"},
        _result(protocol, "call_b", output),
        {"role": "user", "content": [image, {"type": text_type, "text": "between results"}]},
        _result(protocol, "call_a", "result a"),
        {"role": "user", "content": "after results"},
    ])
    image_block = {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "AAAA",
    }}
    assert len(payload["messages"]) == 3
    assert payload["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_b", "content": [
            {"type": "text", "text": "result b"}, image_block,
        ]},
        {"type": "tool_result", "tool_use_id": "call_a", "content": "result a"},
        {"type": "text", "text": "summary before results"},
        image_block,
        {"type": "text", "text": "between results"},
        {"type": "text", "text": "after results"},
    ]
    assert [b["id"] for b in payload["messages"][1]["content"]] == ["call_a", "call_b"]


def test_already_valid_result_then_text_is_unchanged(protocol):
    payload = _translate(protocol, [
        {"role": "user", "content": "start"},
        *_calls(protocol, "call_a"),
        _result(protocol, "call_a", "ok"),
        {"role": "user", "content": "continue"},
    ])
    assert payload["messages"][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_a", "content": "ok"},
        {"type": "text", "text": "continue"},
    ]


def test_does_not_move_results_across_assistant_turns_or_invent_missing_results(protocol):
    payload = _translate(protocol, [
        {"role": "user", "content": "start"},
        *_calls(protocol, "call_a", "call_missing"),
        {"role": "user", "content": "first user turn"},
        {"role": "assistant", "content": "intervening assistant"},
        {"role": "user", "content": "second user turn"},
        _result(protocol, "call_a", "ok"),
    ])
    messages = payload["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == [{"type": "text", "text": "first user turn"}]
    assert messages[3]["content"] == [{"type": "text", "text": "intervening assistant"}]
    assert messages[4]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_a", "content": "ok"},
        {"type": "text", "text": "second user turn"},
    ]
    assert [b["tool_use_id"] for m in messages for b in m["content"]
            if b["type"] == "tool_result"] == ["call_a"]
