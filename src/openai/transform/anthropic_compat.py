"""Small, loss-aware compatibility helpers for Anthropic ingress.

No provider I/O. Unknown signed reasoning is never re-labelled as another
provider's encrypted state. Stop matching is a downstream compatibility layer,
not a claim that the upstream stopped generation or billing at that point.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from .guard import GuardError


def output_format(body: dict, *, chat: bool = False) -> dict | None:
    config = body.get("output_config")
    if config is not None and (not isinstance(config, dict) or ("type" in config and "format" not in config)):
        raise GuardError(400, "invalid_request_error", "Structured output must use output_config.format", param="output_config")
    fmt = config.get("format") if isinstance(config, dict) else None
    if fmt is None:
        fmt = body.get("output_format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise GuardError(400, "invalid_request_error", "output_config.format must contain a JSON schema", param="output_config.format")
    # Preserve optional properties rather than making them required/null just
    # to satisfy OpenAI strict mode. Compatible schemas keep strict guarantees;
    # other valid Claude schemas retain their schema in non-strict JSON mode.
    value = {"name": "parrot_response", "schema": copy.deepcopy(fmt["schema"]), "strict": _strict_compatible(fmt["schema"])}
    if chat:
        return {"type": "json_schema", "json_schema": value}
    return {"type": "json_schema", **value}


def _strict_compatible(value: Any) -> bool:
    if isinstance(value, list):
        return all(_strict_compatible(v) for v in value)
    if not isinstance(value, dict):
        return True
    if value.get("type") == "object" or "properties" in value:
        props = value.get("properties") or {}
        if value.get("additionalProperties") is not False or set(value.get("required") or []) != set(props):
            return False
    if any(k in value for k in ("patternProperties", "unevaluatedProperties", "dependentSchemas", "if", "then", "else", "not", "allOf")):
        return False
    return all(_strict_compatible(v) for v in value.values())


def mark_tool_error(output: str | list[dict], block: dict) -> str | list[dict]:
    if block.get("is_error") is not True:
        return output
    marker = "[Tool execution failed]"
    if isinstance(output, list):
        return [{"type": "input_text", "text": marker}, *output]
    return marker + ("\n" + output if output else "")


def stop_sequences(body: dict | None) -> tuple[str, ...]:
    raw = (body or {}).get("stop_sequences")
    return tuple(dict.fromkeys(x for x in raw if isinstance(x, str) and x)) if isinstance(raw, list) else ()


class StopMatcher:
    """Hold only a possible stop prefix; never leak a split stop to the client."""
    def __init__(self, sequences: tuple[str, ...]):
        self.sequences = sequences
        self.pending = ""
        self.matched: str | None = None

    def feed(self, text: str, *, final: bool = False) -> str:
        if self.matched:
            return ""
        value = self.pending + text
        self.pending = ""
        hits = [(value.find(s), i, s) for i, s in enumerate(self.sequences) if s in value]
        if hits:
            at, _, self.matched = min(hits)
            return value[:at]
        hold = 0
        if not final:
            for seq in self.sequences:
                for size in range(min(len(seq) - 1, len(value)), 0, -1):
                    if value.endswith(seq[:size]):
                        hold = max(hold, size)
                        break
        if hold:
            self.pending = value[-hold:]
            return value[:-hold]
        return value


def apply_stop_sequences(message: dict, body: dict | None) -> dict:
    sequences = stop_sequences(body)
    if not sequences:
        return message
    out = dict(message)
    content = []
    for block in message.get("content") or []:
        if block.get("type") == "text":
            matcher = StopMatcher(sequences)
            text = matcher.feed(str(block.get("text") or ""), final=True)
            if text:
                content.append({**block, "text": text})
            if matcher.matched:
                out.update(stop_reason="stop_sequence", stop_sequence=matcher.matched, content=content)
                return out
        else:
            content.append(block)
    return message


class StopSequenceStream:
    """Filter already-translated Anthropic events while preserving final usage.

    The transport still drains the upstream terminal for honest accounting.
    Later tool blocks after a matched stop are not exposed or executed.
    """
    def __init__(self, inner, body: dict):
        self.inner = inner
        self.body = body
        self.matcher = StopMatcher(stop_sequences(body))
        self.active_text: int | None = None
        self.visible_blocks: set[int] = set()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @staticmethod
    def _frame(obj: dict) -> bytes:
        return ("event: " + obj["type"] + "\ndata: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode()

    def _filter(self, chunks):
        for raw in chunks:
            data = next((line[5:].strip() for line in raw.decode().splitlines() if line.startswith("data:")), "")
            try:
                obj = json.loads(data)
            except (ValueError, TypeError):
                yield raw
                continue
            typ, index = obj.get("type"), obj.get("index")
            if typ == "message_delta" and self.matcher.matched:
                obj = {**obj, "delta": {**obj.get("delta", {}), "stop_reason": "stop_sequence", "stop_sequence": self.matcher.matched}}
            if typ == "content_block_start":
                if self.matcher.matched:
                    continue
                self.visible_blocks.add(index)
                if obj.get("content_block", {}).get("type") == "text":
                    self.active_text = index
            elif typ == "content_block_delta":
                if index not in self.visible_blocks:
                    continue
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta":
                    if self.matcher.matched:
                        continue
                    text = self.matcher.feed(str(delta.get("text") or ""))
                    if not text:
                        continue
                    obj = {**obj, "delta": {**delta, "text": text}}
            elif typ == "content_block_stop":
                if index not in self.visible_blocks:
                    continue
                if index == self.active_text:
                    text = self.matcher.feed("", final=True)
                    if text:
                        yield self._frame({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}})
                    self.active_text = None
            yield self._frame(obj)

    def feed(self, chunk):
        yield from self._filter(self.inner.feed(chunk))

    def close(self):
        yield from self._filter(self.inner.close())

    def get_downstream_anthropic_assistant(self):
        return apply_stop_sequences(self.inner.get_downstream_anthropic_assistant(), self.body)
