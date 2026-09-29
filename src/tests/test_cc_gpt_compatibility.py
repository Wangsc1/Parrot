"""CC interoperability regressions: convert compatible data, not spurious 400s."""
from __future__ import annotations
import asyncio
import json
import pytest

from src.openai.channel.api_channel import OpenAIApiChannel
from src.openai.transform import anthropic_to_chat as chat, anthropic_to_responses as responses
from src.openai.transform.anthropic_compat import StopMatcher, apply_stop_sequences
from src.openai.transform.stream_chat_to_anthropic import StreamTranslator as ChatStream
from src.openai.transform.stream_responses_to_anthropic import StreamTranslator as ResponsesStream
from src.protocols.commit_gate import SseCommitGate
from src.protocols.matrix import DEFAULT_MATRIX, extract_request_features
from src.protocols.runtime import make_stream_translator, prepare_non_stream_response
from src.upstream import ResponsesSSEUsageTracker

BASE = {"model": "cc", "max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}]}
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}

def frame(name, data, nl="\n"):
    return ((f"event: {name}{nl}" if name else "") + "data: " + (data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)) + nl + nl).encode()

def decoded(chunks):
    return [json.loads(next(line[5:].strip() for line in raw.decode().splitlines() if line.startswith("data:"))) for raw in chunks]

def channel(protocol="openai-responses", model="gpt-5"):
    return OpenAIApiChannel({"name": "compat-test", "baseUrl": "https://api.example.test", "apiKey": "test", "protocol": protocol, "models": [{"alias": "cc", "real": model}]})

@pytest.mark.parametrize("protocol,bridge", [("openai-chat", chat), ("openai-responses", responses)])
@pytest.mark.parametrize("effort", [None, "high"])
def test_structured_output_format_only_and_with_effort(protocol, bridge, effort):
    cfg = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if effort:
        cfg["effort"] = effort
    body = {**BASE, "output_config": cfg}
    DEFAULT_MATRIX.plan("anthropic", protocol, extract_request_features("anthropic", body))
    payload = bridge.translate_request(body, target_model="gpt-5")
    fmt = payload["response_format"]["json_schema"] if protocol == "openai-chat" else payload["text"]["format"]
    assert fmt["schema"] == SCHEMA and fmt["strict"] is True
    assert cfg == body["output_config"]


def test_optional_output_properties_are_not_made_required():
    schema = {"type": "object", "properties": {"note": {"type": "string"}}, "additionalProperties": False}
    payload = responses.translate_request({**BASE, "output_config": {"format": {"type": "json_schema", "schema": schema}}})
    assert payload["text"]["format"]["schema"] == schema
    assert payload["text"]["format"]["strict"] is False

@pytest.mark.parametrize("bridge", [chat, responses])
@pytest.mark.parametrize("strict", [None, False, True])
def test_explicit_and_default_function_strict(bridge, strict):
    tool = {"name": "Read", "input_schema": {"type": "object", "properties": {"offset": {"type": "integer"}}}}
    if strict is not None:
        tool["strict"] = strict
    result = bridge.translate_request({**BASE, "tools": [tool]})["tools"][0]
    fn = result.get("function", result)
    assert fn["strict"] is (strict if strict is not None else False)
    assert fn["parameters"] == tool["input_schema"]

@pytest.mark.parametrize("protocol,bridge", [("openai-chat", chat), ("openai-responses", responses)])
def test_switch_from_claude_drops_only_foreign_signed_thinking(protocol, bridge):
    body = {**BASE, "messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "private reasoning", "signature": "foreign"}, {"type": "text", "text": "answer"}, {"type": "tool_use", "id": "call_a", "name": "Read", "input": {}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_a", "content": "ENOENT", "is_error": True}]}]}
    DEFAULT_MATRIX.plan("anthropic", protocol, extract_request_features("anthropic", body))
    payload = bridge.translate_request(body)
    wire = json.dumps(payload)
    assert "private reasoning" not in wire and "foreign" not in wire
    assert "answer" in wire and "call_a" in wire and "ENOENT" in wire and "Tool execution failed" in wire


def test_chat_tool_attachments_after_all_parallel_results():
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    doc = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}}
    body = {**BASE, "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": "Read", "input": {}} for cid in ("a", "b")]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": [image]}, {"type": "tool_result", "tool_use_id": "b", "content": [doc]}, {"type": "text", "text": "continue"}]}]}
    DEFAULT_MATRIX.plan("anthropic", "openai-chat", extract_request_features("anthropic", body))
    msgs = chat.translate_request(body)["messages"]
    assert [m["role"] for m in msgs] == ["assistant", "tool", "tool", "user"]
    wire = json.dumps(msgs[-1])
    assert "data:image/png;base64,AAAA" in wire and "data:application/pdf;base64,JVBERi0=" in wire
    assert "Attachment from tool result a" in wire and "Attachment from tool result b" in wire

@pytest.mark.parametrize("protocol", ["openai-chat", "openai-responses"])
def test_final_reasoning_model_payload_is_compatible(protocol):
    body = {**BASE, "temperature": 1, "top_p": 0.9, "thinking": {"type": "enabled", "budget_tokens": 2048}, "stop_sequences": ["END"]}
    ch = channel(protocol, "o3")
    req = asyncio.run(ch.build_upstream_request(body, "o3", ingress_protocol="anthropic"))
    wire = json.loads(req.body)
    field = "max_completion_tokens" if protocol == "openai-chat" else "max_output_tokens"
    assert wire[field] == 4096
    assert not {"max_tokens", "temperature", "top_p", "stop"} & wire.keys()
    assert req.translator_ctx["request_body"]["stop_sequences"] == ["END"]


def test_effort_adapts_only_to_advertised_levels_at_cross_protocol_boundary(monkeypatch):
    from src import model_metadata
    monkeypatch.setattr(model_metadata, "get_metadata", lambda *a, **k: {"reasoningEfforts": ["low", "medium", "high"]})
    ch = channel("openai-responses", "gpt-5.1")
    body = {**BASE, "output_config": {"effort": "max"}}
    wire = json.loads(asyncio.run(ch.build_upstream_request(body, "gpt-5.1", ingress_protocol="anthropic")).body)
    assert wire["reasoning"]["effort"] == "high"
    assert body["output_config"]["effort"] == "max"
    native = {"input": "hi", "reasoning": {"effort": "max"}}
    wire = json.loads(asyncio.run(ch.build_upstream_request(native, "gpt-5.1", ingress_protocol="responses")).body)
    assert wire["reasoning"]["effort"] == "max"


def test_nonreasoning_gpt_ignores_unavailable_thinking_instead_of_sending_invalid_field():
    ch = channel("openai-chat", "gpt-4.1")
    body = {**BASE, "thinking": {"type": "enabled", "budget_tokens": 2048}}
    wire = json.loads(asyncio.run(ch.build_upstream_request(body, "gpt-4.1", ingress_protocol="anthropic")).body)
    assert "reasoning_effort" not in wire and wire["messages"][0]["content"] == "hi"


def test_nonreasoning_and_native_payload_sampling_unchanged():
    body = {**BASE, "temperature": 0.2}
    ch = channel("openai-chat", "gpt-4.1")
    wire = json.loads(asyncio.run(ch.build_upstream_request(body, "gpt-4.1", ingress_protocol="anthropic")).body)
    assert wire["max_tokens"] == 4096 and wire["temperature"] == 0.2
    wire = json.loads(asyncio.run(ch.build_upstream_request(body, "gpt-5", ingress_protocol="chat")).body)
    assert wire["temperature"] == 0.2

@pytest.mark.parametrize("cls,is_responses", [(ChatStream, False), (ResponsesStream, True)])
def test_crlf_after_commit_and_split_utf8(cls, is_responses):
    tr = cls(model="gpt")
    gate = SseCommitGate(protocol="anthropic", stream_translator=tr)
    if is_responses:
        first = frame("response.output_text.delta", {"type": "response.output_text.delta", "output_index": 0, "delta": "A"}, "\r\n")
        tail = frame("response.output_text.delta", {"type": "response.output_text.delta", "output_index": 0, "delta": "中文B"}, "\r\n") + frame("response.completed", {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 3}}}}, "\r\n")
    else:
        first = frame("", {"choices": [{"delta": {"content": "A"}}]}, "\r\n")
        tail = frame("", {"choices": [{"delta": {"content": "中文B"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 3}}}, "\r\n") + frame("", "[DONE]", "\r\n")
    out = gate.feed(first).downstream_chunks
    for value in tail:
        out.extend(tr.feed(bytes([value])))
    out.extend(tr.close())
    events = decoded(out)
    assert "".join(e.get("delta", {}).get("text", "") for e in events) == "A中文B"
    terminal = next(e for e in events if e["type"] == "message_delta")
    assert terminal["usage"]["input_tokens"] == 7 and terminal["usage"]["cache_read_input_tokens"] == 3


def test_chat_multiline_sse_data_preserves_full_json():
    tr = ChatStream(model="gpt")
    events = decoded(tr.feed(b'data: {"choices": [\r\ndata: {"delta": {"content": "ok"}, "finish_reason": "stop"}]}\r\n\r\n'))
    assert "".join(e.get("delta", {}).get("text", "") for e in events) == "ok"


def test_late_tool_metadata_buffered_until_real_name_id():
    tr = ChatStream(model="gpt")
    a = frame("", {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}}]})
    b = frame("", {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "real_id", "function": {"name": "Read", "arguments": "1}"}}]}, "finish_reason": "tool_calls"}]})
    initial = decoded(tr.feed(a))
    assert not any(e["type"] == "content_block_start" for e in initial)
    events = decoded(list(tr.feed(b)) + list(tr.close()))
    block = next(e["content_block"] for e in events if e["type"] == "content_block_start")
    assert block["id"] == "real_id" and block["name"] == "Read"
    assert json.loads("".join(e.get("delta", {}).get("partial_json", "") for e in events)) == {"a": 1}

@pytest.mark.parametrize("reason,stop", [("max_output_tokens", "max_tokens"), ("content_filter", "refusal")])
def test_incomplete_is_protocol_terminal_not_fake_context_error(reason, stop):
    tr = ResponsesStream(model="gpt")
    gate = SseCommitGate(protocol="anthropic", stream_translator=tr)
    resp = {"status": "incomplete", "incomplete_details": {"reason": reason}, "output": [], "usage": {"input_tokens": 5, "output_tokens": 4}}
    event = frame("response.incomplete", {"type": "response.incomplete", "response": resp})
    step = gate.feed(event)
    assert step.error_event is None and step.downstream_chunks
    events = decoded(step.downstream_chunks + list(tr.close()))
    assert not any(e["type"] == "error" for e in events)
    assert next(e for e in events if e["type"] == "message_delta")["delta"]["stop_reason"] == stop
    assert responses.translate_response(resp)["stop_reason"] == stop
    tracker = ResponsesSSEUsageTracker()
    tracker.preserve_incomplete = True
    tracker.feed(event)
    assert tracker.saw_stream_end and not tracker.saw_stream_error


def test_nonstream_output_budget_retains_incomplete_for_native_and_translated_routes():
    raw = json.dumps({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [], "usage": {"input_tokens": 5, "output_tokens": 4}}).encode()
    kwargs = dict(dynamic_map=None, connect_ms=1, total_ms=2)
    translated = asyncio.run(prepare_non_stream_response(channel(), raw, translator_ctx={"response_translator": "anthropic_to_responses"}, **kwargs))
    native = asyncio.run(prepare_non_stream_response(channel(), raw, **kwargs))
    assert translated.ok and native.ok
    assert translated.obj["status"] == native.obj["status"] == "incomplete"
    assert translated.obj["incomplete_details"] == native.obj["incomplete_details"] == {"reason": "max_output_tokens"}


def test_split_stop_sequence_never_leaks_and_suppresses_later_tools():
    ctx = {"response_translator": "anthropic_to_responses", "model_for_response": "gpt", "request_body": {"stop_sequences": ["</block>"]}}
    tr = make_stream_translator(ctx)
    out = []
    for text in ("hello</bl", "ock>hidden"):
        out += list(tr.feed(frame("response.output_text.delta", {"type": "response.output_text.delta", "output_index": 0, "delta": text})))
    out += list(tr.feed(frame("response.output_item.added", {"type": "response.output_item.added", "output_index": 1, "item": {"id": "fc_a", "call_id": "call_a", "type": "function_call", "name": "Delete", "arguments": "{}"}})))
    out += list(tr.feed(frame("response.completed", {"type": "response.completed", "response": {"status": "completed", "output": [], "usage": {"input_tokens": 10, "output_tokens": 8}}})))
    out += list(tr.close())
    events = decoded(out)
    assert "".join(e.get("delta", {}).get("text", "") for e in events) == "hello"
    assert not any(e.get("content_block", {}).get("type") == "tool_use" for e in events)
    terminal = next(e for e in events if e["type"] == "message_delta")
    assert terminal["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "</block>"}
    assert terminal["usage"]["output_tokens"] == 8
    assert tr.get_downstream_anthropic_assistant()["content"] == [{"type": "text", "text": "hello"}]


def test_stop_prefix_flushed_when_no_match_and_nonstream_parity():
    matcher = StopMatcher(("</block>",))
    assert matcher.feed("answer</bl") == "answer"
    assert matcher.feed("", final=True) == "</bl"
    message = {"content": [{"type": "text", "text": "aENDhidden"}, {"type": "tool_use", "id": "b"}], "usage": {"output_tokens": 8}}
    cut = apply_stop_sequences(message, {"stop_sequences": ["END"]})
    assert cut["content"] == [{"type": "text", "text": "a"}] and cut["stop_reason"] == "stop_sequence"
    assert message["content"][0]["text"] == "aENDhidden" and cut["usage"] == message["usage"]
