"""SSE translator: OpenAI Responses stream → Anthropic Messages stream.

Used by Phase 8 Anthropic ingress → OpenAI Responses upstream.  Metadata events
(response.created/in_progress) are consumed for state but do not emit downstream
bytes; the first Anthropic bytes are emitted only when a visible text/tool event
arrives, preserving the existing Responses failover commit boundary.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ...protocols import errors as protocol_errors
from ...protocols.sse import split_sse_events
from . import common
from ._stream_response_text_order import TextOutputOrder
from ...protocols.usage import legacy_usage_from_openai_responses_json


def _gen_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


def _emit(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")


def _parse_event_block(block: str) -> tuple[Optional[str], Optional[dict]]:
    event_name: Optional[str] = None
    data_lines: list[str] = []
    for line in block.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            event_name = line[6:].strip() or None
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return event_name, None
    data_str = "\n".join(data_lines)
    if data_str == "[DONE]":
        return event_name, None
    try:
        obj = json.loads(data_str)
    except Exception:
        return event_name, None
    return event_name, obj if isinstance(obj, dict) else None


def _error_type_from_code_or_message(code: Any, message: Any) -> str:
    low = f"{code or ''} {message or ''}".lower()
    if protocol_errors.is_context_length_code_or_message(code, message):
        return "invalid_request_error"
    if "rate_limit" in low or "rate limit" in low:
        return "rate_limit_error"
    if "permission" in low or "forbidden" in low:
        return "permission_error"
    if "auth" in low or "api key" in low:
        return "authentication_error"
    return "api_error"


def _normalize_error_for_anthropic(err: dict[str, Any]) -> dict[str, Any]:
    out = dict(err or {})
    message = out.get("message") or out.get("reason") or "upstream response failed"
    code = out.get("code") or out.get("error_type")
    if protocol_errors.is_context_length_code_or_message(code, message):
        out["type"] = "invalid_request_error"
        out["code"] = protocol_errors.CONTEXT_LENGTH_EXCEEDED_CODE
        out["message"] = protocol_errors.context_length_error_message_for_claude_code(message)
        return out
    if not out.get("message"):
        out["message"] = str(message)
    if not out.get("type"):
        out["type"] = _error_type_from_code_or_message(code, message)
    return out


def _response_from_event(data: dict) -> dict:
    resp = data.get("response")
    return resp if isinstance(resp, dict) else data


def _anthropic_usage_from_responses_usage(usage: Optional[dict]) -> dict[str, int]:
    legacy = legacy_usage_from_openai_responses_json({"usage": usage or {}})
    return {
        "input_tokens": legacy["input_tokens"],
        "output_tokens": legacy["output_tokens"],
        "cache_creation_input_tokens": legacy["cache_creation"],
        "cache_read_input_tokens": legacy["cache_read"],
    }


def _stop_reason(status: Optional[str], incomplete_reason: Optional[str], *, saw_tool: bool) -> str:
    if status == "incomplete":
        if incomplete_reason in ("max_output_tokens", "max_tokens"):
            return "max_tokens"
        if incomplete_reason == "content_filter":
            return "refusal"
        return "pause_turn"
    return "tool_use" if saw_tool else "end_turn"


@dataclass
class _ReasoningState:
    key: str
    block_index: int
    thinking: str = ""
    signature: str = ""
    started: bool = False
    stopped: bool = False


@dataclass
class _ToolState:
    key: str
    block_index: int
    id: str = ""
    name: str = ""
    args: str = ""
    started: bool = False
    stopped: bool = False
    args_emitted: bool = False


@dataclass
class _State:
    message_id: str
    model: str
    created_ts: int
    message_started: bool = False
    text_started: bool = False
    text_stopped: bool = False
    text_index: int = -1
    next_block_index: int = 0
    text_blocks: dict[int, list[str]] = field(default_factory=dict)
    tools: dict[str, _ToolState] = field(default_factory=dict)
    reasoning: dict[str, _ReasoningState] = field(default_factory=dict)
    output_index_to_reasoning_key: dict[int, str] = field(default_factory=dict)
    item_id_to_reasoning_key: dict[str, str] = field(default_factory=dict)
    active_reasoning_key: Optional[str] = None
    seen_reasoning_signatures: set[str] = field(default_factory=set)
    output_index_to_key: dict[int, str] = field(default_factory=dict)
    item_id_to_key: dict[str, str] = field(default_factory=dict)
    last_tool_key: Optional[str] = None
    status: Optional[str] = None
    incomplete_reason: Optional[str] = None
    usage: Optional[dict] = None
    terminal_emitted: bool = False

    def alloc_index(self) -> int:
        idx = self.next_block_index
        self.next_block_index += 1
        return idx


class StreamTranslator:
    """OpenAI Responses SSE → Anthropic SSE."""

    preserves_incomplete = True

    def __init__(
        self,
        *,
        model: str,
        created_ts: Optional[int] = None,
        optional_empty_string_fields_by_tool: dict[str, set[str]] | None = None,
        request_body: dict[str, Any] | None = None,
        allow_reasoning_bridge: bool = False,
    ):
        self.state = _State(
            message_id=_gen_id("msg_"),
            model=model,
            created_ts=int(created_ts or time.time()),
        )
        self.optional_empty_string_fields_by_tool = dict(optional_empty_string_fields_by_tool or {})
        if request_body is not None:
            self.optional_empty_string_fields_by_tool.update(
                common.optional_empty_string_fields_by_tool_from_anthropic_tools(
                    request_body.get("tools") if isinstance(request_body, dict) else None
                )
            )
        self.allow_reasoning_bridge = bool(allow_reasoning_bridge)
        self._buf = b""
        self._hosted_seen: set[str] = set()
        self._hosted_blocks: list[tuple[int, dict]] = []
        self._pending_hosted: tuple[str, int | None] | None = None
        self._deferred_events: list[tuple[str, dict]] = []
        self._pending_reasoning: tuple[str, int | None] | None = None
        self._deferred_reasoning: list[tuple[str, dict]] = []
        self._item_ids_by_output_index: dict[int, str] = {}
        self._active_text_item: str | None = None
        self._text_emitted_by_part: dict[tuple[str, int], str] = {}
        self._text_order = TextOutputOrder(max_output_is_error=False)

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        if not chunk:
            return
        self._buf, blocks = split_sse_events(self._buf + chunk)
        for block_bytes in blocks:
            block = block_bytes.decode("utf-8", errors="replace")
            if not block.strip():
                continue
            event_name, data = _parse_event_block(block)
            if event_name is None and data is None:
                continue
            for ready_name, ready_data in self._text_order.feed(event_name or "", data or {}):
                yield from self._handle_event(ready_name, ready_data)

    def close(self) -> Iterator[bytes]:
        if self.state.terminal_emitted:
            return
        self.state.terminal_emitted = True
        if not self.state.message_started:
            yield from self._emit_message_start()
        yield from self._stop_text_if_needed()
        yield from self._stop_all_reasoning()
        yield from self._stop_all_tools()
        yield _emit("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": _stop_reason(
                    self.state.status, self.state.incomplete_reason,
                    saw_tool=any(t.started for t in self.state.tools.values()),
                ),
                "stop_sequence": None,
            },
            "usage": _anthropic_usage_from_responses_usage(self.state.usage),
        })
        yield _emit("message_stop", {"type": "message_stop"})

    # ─── event handling ──────────────────────────────────────────

    def _handle_event(self, event_name: str, data: dict) -> Iterator[bytes]:
        # The final reasoning summary determines thinking vs redacted_thinking.
        # Do not commit later text/tool blocks until that earlier item is done.
        if self._pending_reasoning is not None:
            pending_id, pending_index = self._pending_reasoning
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            oi = data.get("output_index")
            same_item = (str(item.get("id") or data.get("item_id") or "") == pending_id
                         or (isinstance(oi, int) and oi == pending_index))
            if event_name in ("error", "response.failed") or data.get("type") == "error":
                self._pending_reasoning = None
                self._deferred_reasoning.clear()
            elif event_name == "response.output_item.done" and item.get("type") == "reasoning" and same_item:
                self._pending_reasoning = None
                yield from self._on_output_item_done(data, item)
                deferred, self._deferred_reasoning = self._deferred_reasoning, []
                for name, frame in deferred:
                    yield from self._handle_event(name, frame)
                return
            elif event_name in ("response.completed", "response.incomplete"):
                resp = _response_from_event(data)
                snapshot = next((i for i in resp.get("output") or [] if isinstance(i, dict)
                                 and i.get("type") == "reasoning" and str(i.get("id") or "") == pending_id), None)
                if snapshot is None:
                    self._deferred_reasoning.append((event_name, data))
                    return
                self._pending_reasoning = None
                yield from self._on_output_item_done({"output_index": pending_index, "item": snapshot}, snapshot)
                deferred, self._deferred_reasoning = self._deferred_reasoning, []
                for name, frame in deferred:
                    yield from self._handle_event(name, frame)
            elif ((isinstance(oi, int) and isinstance(pending_index, int) and oi < pending_index)
                  or (event_name == "response.reasoning_summary_text.delta" and same_item)):
                self._pending_reasoning = None
                try:
                    yield from self._handle_event(event_name, data)
                finally:
                    self._pending_reasoning = (pending_id, pending_index)
                return
            elif event_name != "keepalive":
                self._deferred_reasoning.append((event_name, data))
                return
        # A hosted call's final evidence can arrive after later items. Defer
        # those items until its done event so their Anthropic blocks cannot be
        # emitted ahead of the hosted result.
        if self._pending_hosted is not None and (event_name in ("error", "response.failed") or data.get("type") == "error"):
            self._pending_hosted = None
            self._deferred_events.clear()
        if self._pending_hosted is not None:
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            pending_id, pending_index = self._pending_hosted
            # A preceding output item may itself finish after hosted.added.
            # Its missing suffix still belongs before the hosted blocks.
            prior_index = data.get("output_index")
            if isinstance(prior_index, int) and isinstance(pending_index, int) and prior_index < pending_index:
                self._pending_hosted = None
                try:
                    yield from self._handle_event(event_name, data)
                finally:
                    self._pending_hosted = (pending_id, pending_index)
                return
            if (event_name == "response.output_item.done" and item.get("type") == "web_search_call"
                    and (str(item.get("id") or "") == pending_id or data.get("output_index") == pending_index)):
                self._pending_hosted = None
                yield from self._on_output_item_done(data, item)
                deferred, self._deferred_events = self._deferred_events, []
                for name, frame in deferred:
                    yield from self._handle_event(name, frame)
                return
            if event_name in ("response.completed", "response.incomplete"):
                resp = _response_from_event(data)
                hosted = next((i for i in resp.get("output") or [] if isinstance(i, dict)
                               and i.get("type") == "web_search_call" and str(i.get("id") or "") == pending_id), None)
                if hosted is not None:
                    self._pending_hosted = None
                    yield from self._emit_hosted_search(hosted)
                    deferred, self._deferred_events = self._deferred_events, []
                    for name, frame in deferred:
                        yield from self._handle_event(name, frame)
                else:
                    self._deferred_events.append((event_name, data))
                    return
            elif event_name != "keepalive":
                self._deferred_events.append((event_name, data))
                return
        if event_name == "error" or data.get("type") == "error" or isinstance(data.get("error"), dict):
            err = data.get("error") if isinstance(data.get("error"), dict) else data
            yield _emit("error", {"type": "error", "error": _normalize_error_for_anthropic(err)})
            self.state.terminal_emitted = True
            return

        if event_name in ("response.created", "response.in_progress"):
            self._capture_response_metadata(_response_from_event(data))
            return

        if event_name == "keepalive" or data.get("type") == "keepalive":
            yield from self._emit_ping()
            return

        if event_name == "response.output_item.added":
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            if isinstance(data.get("output_index"), int) and isinstance(item.get("id"), str):
                self._item_ids_by_output_index[data["output_index"]] = item["id"]
            if item.get("type") == "web_search_call" and item.get("id"):
                self._pending_hosted = (str(item["id"]), data.get("output_index"))
                return
            if item.get("type") == "reasoning" and self.allow_reasoning_bridge and item.get("id"):
                self._pending_reasoning = (str(item["id"]), data.get("output_index"))
            yield from self._on_output_item_added(data, item)
            return

        if event_name == "response.output_item.done":
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            yield from self._on_output_item_done(data, item)
            return

        if event_name == "response.content_part.added":
            part = data.get("part") if isinstance(data.get("part"), dict) else {}
            if part.get("type") in ("output_text", "refusal"):
                yield from self._ensure_text_for_item(data)
            return

        if event_name in ("response.output_text.done", "response.refusal.done", "response.content_part.done"):
            part = data.get("part") if isinstance(data.get("part"), dict) else {}
            text = (part.get("text") or part.get("refusal") if event_name == "response.content_part.done"
                    else data.get("text") or data.get("refusal"))
            if event_name != "response.content_part.done" or part.get("type") in ("output_text", "refusal"):
                yield from self._emit_text_tail(data, text)
            return

        if event_name == "response.reasoning_summary_text.delta":
            if self.allow_reasoning_bridge:
                delta = data.get("delta")
                if isinstance(delta, str) and delta:
                    yield from self._emit_reasoning_delta(self._reasoning_for_event(data), delta)
            return

        if event_name in ("response.output_text.delta", "response.refusal.delta"):
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                yield from self._emit_text_for_item(data, delta)
            return

        if event_name == "response.function_call_arguments.delta":
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                st = self._tool_for_delta_event(data)
                if not st.stopped:
                    yield from self._emit_tool_args_delta(st, delta)
            return

        if event_name in ("response.completed", "response.incomplete", "response.failed"):
            resp = _response_from_event(data)
            self._capture_response_metadata(resp)
            if event_name != "response.failed":
                for index, item in enumerate(resp.get("output") or []):
                    if not isinstance(item, dict):
                        continue
                    output_index = next((oi for oi, item_id in self._item_ids_by_output_index.items()
                                         if item_id == item.get("id")), index)
                    yield from self._on_output_item_done({"output_index": output_index, "item": item}, item)
            self.state.status = str(resp.get("status") or event_name.removeprefix("response."))
            details = resp.get("incomplete_details") if isinstance(resp.get("incomplete_details"), dict) else {}
            self.state.incomplete_reason = details.get("reason") if isinstance(details, dict) else None
            usage = resp.get("usage")
            if isinstance(usage, dict):
                self.state.usage = usage
            if event_name != "response.failed" and not self.state.message_started:
                yield from self._emit_message_start()
            if event_name == "response.failed":
                err = (
                    resp.get("error")
                    if isinstance(resp.get("error"), dict)
                    else {"message": "upstream response failed"}
                )
                yield _emit("error", {"type": "error", "error": _normalize_error_for_anthropic(err)})
                self.state.terminal_emitted = True
            return

    def _capture_response_metadata(self, resp: dict) -> None:
        if isinstance(resp.get("id"), str) and resp.get("id"):
            self.state.message_id = resp["id"]
        if isinstance(resp.get("model"), str) and resp.get("model"):
            self.state.model = resp["model"]
        if isinstance(resp.get("usage"), dict):
            self.state.usage = resp["usage"]
        if isinstance(resp.get("status"), str):
            self.state.status = resp["status"]

    def _on_output_item_added(self, data: dict, item: dict) -> Iterator[bytes]:
        item_type = item.get("type")
        if item_type == "reasoning" and self.allow_reasoning_bridge:
            st = self._reasoning_for_item(data, item)
            st.signature = str(item.get("encrypted_content") or item.get("thoughtSignature") or "").strip()
            # Delay block_start until readable thinking arrives.  If the item is
            # signature-only, output_item.done must use redacted_thinking rather
            # than an empty ordinary thinking block.
            return
        if item_type != "function_call":
            return
        signature = str(item.get("encrypted_content") or item.get("thoughtSignature") or "").strip()
        if self.allow_reasoning_bridge and signature and signature not in self.state.seen_reasoning_signatures:
            yield from self._emit_redacted_reasoning(signature)
        key = self._key_from_item_event(data, item)
        st = self._tool_state(key)
        self._update_tool_metadata(st, data, item, key)
        yield from self._start_tool_if_needed(st)

    def _on_output_item_done(self, data: dict, item: dict) -> Iterator[bytes]:
        item_type = item.get("type")
        if isinstance(item.get("id"), str) and isinstance(data.get("output_index"), int):
            self._item_ids_by_output_index[data["output_index"]] = item["id"]
        if item_type == "message":
            for content_index, part in enumerate(item.get("content") or []):
                if isinstance(part, dict) and part.get("type") in ("output_text", "refusal"):
                    value = part.get("text") if part.get("type") == "output_text" else part.get("refusal")
                    yield from self._emit_text_tail({**data, "item_id": item.get("id"), "content_index": content_index}, value)
            return
        if item_type == "web_search_call":
            yield from self._emit_hosted_search(item)
            return
        if item_type == "reasoning" and self.allow_reasoning_bridge:
            st = self._reasoning_for_item(data, item)
            summary = item.get("summary") if isinstance(item.get("summary"), list) else []
            final_text = "".join(str(p.get("text") or "") for p in summary if isinstance(p, dict))
            if final_text.startswith(st.thinking):
                yield from self._emit_reasoning_delta(st, final_text[len(st.thinking):])
            st.signature = str(item.get("encrypted_content") or item.get("thoughtSignature") or st.signature).strip()
            if not st.thinking and st.signature:
                st.stopped = True
                self.state.active_reasoning_key = None
                yield from self._emit_redacted_reasoning(st.signature, block_index=st.block_index)
            else:
                yield from self._stop_reasoning(st)
            return
        if item_type != "function_call":
            return
        signature = str(item.get("encrypted_content") or item.get("thoughtSignature") or "").strip()
        if self.allow_reasoning_bridge and signature and signature not in self.state.seen_reasoning_signatures:
            yield from self._emit_redacted_reasoning(signature)
        key = self._key_from_item_event(data, item)
        st = self._tool_state(key)
        if st.stopped:
            return  # done + terminal is one completion, not a second JSON value.
        self._update_tool_metadata(st, data, item, key)
        done_args = item.get("arguments")
        if self._should_buffer_tool_args(st) and isinstance(done_args, str):
            st.args = done_args
        elif isinstance(done_args, str) and done_args != st.args:
            # Responses usually emits argument deltas before output_item.done,
            # but some providers only include the final arguments on the done
            # item.  Emit only the missing suffix when the final value extends
            # the streamed buffer; otherwise avoid duplicating/mangling JSON.
            if done_args.startswith(st.args):
                missing = done_args[len(st.args):]
                if missing:
                    yield from self._emit_tool_args_delta(st, missing)
            elif not st.args:
                yield from self._emit_tool_args_delta(st, done_args)
        if self._should_buffer_tool_args(st):
            yield from self._flush_buffered_tool_args_if_needed(st)
        else:
            yield from self._start_tool_if_needed(st)
        if not st.stopped:
            st.stopped = True
            yield _emit("content_block_stop", {"type": "content_block_stop", "index": st.block_index})

    def _emit_hosted_search(self, item: dict) -> Iterator[bytes]:
        from ... import search_hosted_codec
        key = str(item.get("id") or item.get("call_id") or "")
        if not key or key in self._hosted_seen:
            return
        self._hosted_seen.add(key)
        yield from self._emit_message_start()
        yield from self._stop_text_if_needed()
        for block in search_hosted_codec.responses_to_anthropic(item):
            index = self.state.alloc_index()
            self._hosted_blocks.append((index, block))
            yield _emit("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})
            yield _emit("content_block_stop", {"type": "content_block_stop", "index": index})

    def _update_tool_metadata(self, st: _ToolState, data: dict, item: dict, key: str) -> None:
        call_id = item.get("call_id") or item.get("id")
        if isinstance(call_id, str) and call_id:
            st.id = call_id
        name = item.get("name")
        if isinstance(name, str) and name:
            st.name = name
        self._remember_tool_key(data, item, key)

    # ─── Anthropic emit helpers ──────────────────────────────────

    def _emit_message_start(self) -> Iterator[bytes]:
        if self.state.message_started:
            return
        self.state.message_started = True
        yield _emit("message_start", {
            "type": "message_start",
            "message": {
                "id": self.state.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.state.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def _ensure_text_block(self) -> Iterator[bytes]:
        if not self.state.message_started:
            yield from self._emit_message_start()
        yield from self._stop_active_reasoning()
        if self.state.text_started and not self.state.text_stopped:
            return
        self.state.text_index = self.state.alloc_index()
        self.state.text_blocks[self.state.text_index] = []
        self.state.text_started = True
        self.state.text_stopped = False
        yield _emit("content_block_start", {
            "type": "content_block_start",
            "index": self.state.text_index,
            "content_block": {"type": "text", "text": ""},
        })

    def _text_key(self, data: dict) -> tuple[str, int]:
        oi = data.get("output_index")
        item_id = data.get("item_id") or (self._item_ids_by_output_index.get(oi) if isinstance(oi, int) else None)
        return str(item_id or f"index:{oi}"), int(data.get("content_index", 0) or 0)

    def _ensure_text_for_item(self, data: dict) -> Iterator[bytes]:
        item_key = self._text_key(data)[0]
        if self._active_text_item is not None and self._active_text_item != item_key:
            yield from self._stop_text_if_needed()
        yield from self._ensure_text_block()
        self._active_text_item = item_key

    def _emit_text_for_item(self, data: dict, text: str) -> Iterator[bytes]:
        yield from self._ensure_text_for_item(data)
        key = self._text_key(data)
        self._text_emitted_by_part[key] = self._text_emitted_by_part.get(key, "") + text
        yield from self._emit_text_delta(text)

    def _emit_text_tail(self, data: dict, value: Any) -> Iterator[bytes]:
        if not isinstance(value, str) or not value:
            return
        key = self._text_key(data)
        existing = self._text_emitted_by_part.get(key, "")
        if value.startswith(existing) and len(value) > len(existing):
            yield from self._emit_text_for_item(data, value[len(existing):])

    def _emit_text_delta(self, text: str) -> Iterator[bytes]:
        yield from self._ensure_text_block()
        self.state.text_blocks[self.state.text_index].append(text)
        yield _emit("content_block_delta", {
            "type": "content_block_delta",
            "index": self.state.text_index,
            "delta": {"type": "text_delta", "text": text},
        })

    def _emit_ping(self) -> Iterator[bytes]:
        # Anthropic streams may contain ping events.  Forward OpenAI keepalives
        # as Anthropic pings so Claude Code does not see a dead connection while
        # Responses spends a long time in hidden/replayed reasoning.
        if not self.state.message_started:
            yield from self._emit_message_start()
        yield _emit("ping", {"type": "ping"})

    def _stop_text_if_needed(self) -> Iterator[bytes]:
        if self.state.text_started and not self.state.text_stopped:
            self.state.text_stopped = True
            self._active_text_item = None
            yield _emit("content_block_stop", {"type": "content_block_stop", "index": self.state.text_index})

    def _reasoning_for_item(self, data: dict, item: dict) -> _ReasoningState:
        oi = data.get("output_index")
        item_id = item.get("id") or data.get("item_id")
        if isinstance(oi, int):
            key = self.state.output_index_to_reasoning_key.get(oi) or f"oi:{oi}"
            self.state.output_index_to_reasoning_key[oi] = key
        elif isinstance(item_id, str) and item_id:
            key = self.state.item_id_to_reasoning_key.get(item_id) or f"id:{item_id}"
        else:
            key = self.state.active_reasoning_key or f"fallback:{len(self.state.reasoning)}"
        if isinstance(item_id, str) and item_id:
            self.state.item_id_to_reasoning_key[item_id] = key
        st = self.state.reasoning.get(key)
        if st is None:
            st = _ReasoningState(key=key, block_index=self.state.alloc_index())
            self.state.reasoning[key] = st
        if not st.stopped:
            self.state.active_reasoning_key = key
        return st

    def _reasoning_for_event(self, data: dict) -> _ReasoningState:
        return self._reasoning_for_item(data, {"id": data.get("item_id")})

    def _start_reasoning_if_needed(self, st: _ReasoningState) -> Iterator[bytes]:
        if st.started:
            return
        if not self.state.message_started:
            yield from self._emit_message_start()
        yield from self._stop_text_if_needed()
        st.started = True
        yield _emit("content_block_start", {
            "type": "content_block_start", "index": st.block_index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        })

    def _emit_reasoning_delta(self, st: _ReasoningState, delta: str) -> Iterator[bytes]:
        if not delta or st.stopped:
            return
        yield from self._start_reasoning_if_needed(st)
        st.thinking += delta
        yield _emit("content_block_delta", {
            "type": "content_block_delta", "index": st.block_index,
            "delta": {"type": "thinking_delta", "thinking": delta},
        })

    def _stop_reasoning(self, st: _ReasoningState) -> Iterator[bytes]:
        if st.stopped:
            return
        yield from self._start_reasoning_if_needed(st)
        if st.signature:
            self.state.seen_reasoning_signatures.add(st.signature)
            yield _emit("content_block_delta", {
                "type": "content_block_delta", "index": st.block_index,
                "delta": {"type": "signature_delta", "signature": st.signature},
            })
        st.stopped = True
        if self.state.active_reasoning_key == st.key:
            self.state.active_reasoning_key = None
        yield _emit("content_block_stop", {"type": "content_block_stop", "index": st.block_index})

    def _stop_active_reasoning(self) -> Iterator[bytes]:
        key = self.state.active_reasoning_key
        if key and key in self.state.reasoning:
            st = self.state.reasoning[key]
            if not st.thinking and st.signature and not st.started:
                st.stopped = True
                self.state.active_reasoning_key = None
                yield from self._emit_redacted_reasoning(st.signature, block_index=st.block_index)
            elif st.thinking or st.started:
                yield from self._stop_reasoning(st)

    def _stop_all_reasoning(self) -> Iterator[bytes]:
        for st in sorted(self.state.reasoning.values(), key=lambda value: value.block_index):
            if not st.stopped:
                if not st.thinking and st.signature and not st.started:
                    st.stopped = True
                    if self.state.active_reasoning_key == st.key:
                        self.state.active_reasoning_key = None
                    yield from self._emit_redacted_reasoning(st.signature, block_index=st.block_index)
                else:
                    yield from self._stop_reasoning(st)

    def _emit_redacted_reasoning(self, signature: str, *, block_index: int | None = None) -> Iterator[bytes]:
        if not signature or signature in self.state.seen_reasoning_signatures:
            return
        if not self.state.message_started:
            yield from self._emit_message_start()
        yield from self._stop_text_if_needed()
        yield from self._stop_active_reasoning()
        idx = self.state.alloc_index() if block_index is None else block_index
        self.state.seen_reasoning_signatures.add(signature)
        yield _emit("content_block_start", {
            "type": "content_block_start", "index": idx,
            "content_block": {"type": "redacted_thinking", "data": signature},
        })
        yield _emit("content_block_stop", {"type": "content_block_stop", "index": idx})

    def _tool_state(self, key: str) -> _ToolState:
        st = self.state.tools.get(key)
        if st is None:
            st = _ToolState(key=key, block_index=self.state.alloc_index())
            self.state.tools[key] = st
        self.state.last_tool_key = key
        return st

    def _start_tool_if_needed(self, st: _ToolState) -> Iterator[bytes]:
        if st.started:
            return
        if not self.state.message_started:
            yield from self._emit_message_start()
        yield from self._stop_text_if_needed()
        yield from self._stop_active_reasoning()
        st.id = st.id or _gen_id("call_")
        name = st.name or "tool"
        st.started = True
        yield _emit("content_block_start", {
            "type": "content_block_start",
            "index": st.block_index,
            "content_block": {"type": "tool_use", "id": st.id, "name": name, "input": {}},
        })

    def _should_buffer_tool_args(self, st: _ToolState) -> bool:
        return bool(self.optional_empty_string_fields_by_tool.get(st.name or ""))

    def _sanitized_tool_args_json(self, st: _ToolState) -> str:
        try:
            parsed = json.loads(st.args) if st.args else {}
        except Exception:
            return st.args
        if not isinstance(parsed, dict):
            return st.args
        normalized = common.normalize_tool_input_optional_empty_strings(
            st.name,
            parsed,
            self.optional_empty_string_fields_by_tool,
        )
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))

    def _emit_tool_args_delta(self, st: _ToolState, delta: str) -> Iterator[bytes]:
        if not delta:
            return
        st.args += delta
        if self._should_buffer_tool_args(st):
            return
        yield from self._start_tool_if_needed(st)
        yield _emit("content_block_delta", {
            "type": "content_block_delta",
            "index": st.block_index,
            "delta": {"type": "input_json_delta", "partial_json": delta},
        })

    def _flush_buffered_tool_args_if_needed(self, st: _ToolState) -> Iterator[bytes]:
        if not self._should_buffer_tool_args(st) or st.args_emitted or st.stopped:
            return
        args_json = self._sanitized_tool_args_json(st)
        yield from self._start_tool_if_needed(st)
        if args_json:
            st.args_emitted = True
            yield _emit("content_block_delta", {
                "type": "content_block_delta",
                "index": st.block_index,
                "delta": {"type": "input_json_delta", "partial_json": args_json},
            })

    def _stop_all_tools(self) -> Iterator[bytes]:
        for key in sorted(self.state.tools.keys(), key=lambda k: self.state.tools[k].block_index):
            st = self.state.tools[key]
            if not st.started:
                if self._should_buffer_tool_args(st):
                    yield from self._flush_buffered_tool_args_if_needed(st)
                else:
                    yield from self._start_tool_if_needed(st)
            if st.started and not st.stopped:
                st.stopped = True
                yield _emit("content_block_stop", {"type": "content_block_stop", "index": st.block_index})

    # ─── tool key helpers ────────────────────────────────────────

    def _key_from_item_event(self, data: dict, item: dict) -> str:
        item_id = item.get("id") or data.get("item_id")
        if item_id and item_id in self.state.item_id_to_key:
            return self.state.item_id_to_key[item_id]
        if item.get("call_id"):
            for key, state in self.state.tools.items():
                if state.id == item["call_id"]:
                    return key
        if isinstance(data.get("output_index"), int):
            return f"oi:{data['output_index']}"
        item_id = item.get("id") or data.get("item_id")
        if isinstance(item_id, str) and item_id:
            return f"id:{item_id}"
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id:
            return f"call:{call_id}"
        return f"fallback:{len(self.state.tools)}"

    def _remember_tool_key(self, data: dict, item: dict, key: str) -> None:
        oi = data.get("output_index")
        if isinstance(oi, int):
            self.state.output_index_to_key[oi] = key
        for raw in (item.get("id"), data.get("item_id"), item.get("call_id")):
            if isinstance(raw, str) and raw:
                self.state.item_id_to_key[raw] = key
        self.state.last_tool_key = key

    def _tool_for_delta_event(self, data: dict) -> _ToolState:
        oi = data.get("output_index")
        if isinstance(oi, int) and oi in self.state.output_index_to_key:
            return self._tool_state(self.state.output_index_to_key[oi])
        item_id = data.get("item_id")
        if isinstance(item_id, str) and item_id in self.state.item_id_to_key:
            return self._tool_state(self.state.item_id_to_key[item_id])
        if self.state.last_tool_key:
            return self._tool_state(self.state.last_tool_key)
        key = f"oi:{oi}" if isinstance(oi, int) else f"fallback:{len(self.state.tools)}"
        if isinstance(oi, int):
            self.state.output_index_to_key[oi] = key
        return self._tool_state(key)

    def get_downstream_anthropic_assistant(self) -> dict:
        indexed: list[tuple[int, dict[str, Any]]] = list(self._hosted_blocks)
        for index, parts in self.state.text_blocks.items():
            if parts:
                indexed.append((index, {"type": "text", "text": "".join(parts)}))
        for st in self.state.reasoning.values():
            if st.signature and st.thinking:
                indexed.append((st.block_index, {"type": "thinking", "thinking": st.thinking, "signature": st.signature}))
            elif st.signature:
                indexed.append((st.block_index, {"type": "redacted_thinking", "data": st.signature}))
        ordered = sorted(self.state.tools.values(), key=lambda t: t.block_index)
        for st in ordered:
            if st.started:
                try:
                    parsed = json.loads(st.args) if st.args else {}
                except Exception:
                    parsed = {"_raw": st.args}
                if not isinstance(parsed, dict):
                    parsed = {"_value": parsed}
                parsed = common.normalize_tool_input_optional_empty_strings(
                    st.name,
                    parsed,
                    self.optional_empty_string_fields_by_tool,
                )
                indexed.append((st.block_index, {"type": "tool_use", "id": st.id, "name": st.name or "tool", "input": parsed}))
        return {"role": "assistant", "content": [block for _, block in sorted(indexed, key=lambda pair: pair[0])]}
