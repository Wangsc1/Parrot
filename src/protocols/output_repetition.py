"""Bounded, content-independent detection of exact streaming output loops.

Protocol adapters supply generated text only. Signatures, event names, usage,
heartbeats and empty thinking are not input. No application state or I/O here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator


OUTPUT_REPETITION_CODE = "output_repetition_detected"
SAME_CHARACTER_LIMIT = 100
WHITESPACE_LIMIT = 100
MIN_REPEATED_CHARS = 1024
MIN_REPEATS = 8
MAX_PERIOD_CHARS = 512


@dataclass(frozen=True)
class RepetitionHit:
    rule: str
    position: int
    run_chars: int
    threshold: int
    period_chars: int = 1
    min_repeats: int = 1
    scope: tuple = ()

    @property
    def message(self) -> str:
        # Log enough to explain the cutoff without copying generated/private text.
        return (
            f"{OUTPUT_REPETITION_CODE}: 检测到模型输出连续重复，已中止生成; "
            f"rule={self.rule}; scope={':'.join(map(str, self.scope))}; "
            f"position_chars={self.position}; run_chars={self.run_chars}; "
            f"threshold_chars={self.threshold}; period_chars={self.period_chars}; "
            f"complete_repeats={self.run_chars // self.period_chars}; "
            f"min_repeats={self.min_repeats}"
        )

    def error_event(self) -> dict:
        return {"type": "error", "error": {
            "type": "api_error", "code": OUTPUT_REPETITION_CODE,
            "message": self.message,
        }}


class TextRepetitionDetector:
    """Incremental exact-period matching; no keywords or target text supplied.

    Only previous positions holding the same character can match a period.
    Tracking those positions avoids scanning all 512 periods for every normal
    output character. History and counters remain bounded by the period window.
    Positions are Unicode characters, independent of network chunk boundaries.
    """
    def __init__(self, *, max_period: int = MAX_PERIOD_CHARS):
        if max_period < 1:
            raise ValueError("max_period must be positive")
        self.max_period = max_period
        self.position = 0
        self._last_char: str | None = None
        self._same_run = self._whitespace_run = 0
        self._history: deque[tuple[str, int]] = deque()
        self._positions: dict[str, deque[int]] = {}
        self._matches = [0] * (max_period + 1)
        self._last_match = [0] * (max_period + 1)
        self.hit: RepetitionHit | None = None

    def feed(self, text: str) -> RepetitionHit | None:
        if self.hit is not None:
            return self.hit
        for char in text:
            self.position += 1
            self._same_run = self._same_run + 1 if char == self._last_char else 1
            self._last_char = char
            self._whitespace_run = self._whitespace_run + 1 if char.isspace() else 0
            if self._same_run >= SAME_CHARACTER_LIMIT:
                self.hit = RepetitionHit("identical_character", self.position,
                                         self._same_run, SAME_CHARACTER_LIMIT)
                return self.hit
            if self._whitespace_run >= WHITESPACE_LIMIT:
                self.hit = RepetitionHit("continuous_whitespace", self.position,
                                         self._whitespace_run, WHITESPACE_LIMIT)
                return self.hit

            for previous_position in reversed(self._positions.get(char, ())):
                period = self.position - previous_position
                self._matches[period] = (
                    self._matches[period] + 1
                    if self._last_match[period] == self.position - 1 else 1
                )
                self._last_match[period] = self.position
                run_chars = period + self._matches[period]
                if run_chars >= MIN_REPEATED_CHARS and run_chars // period >= MIN_REPEATS:
                    self.hit = RepetitionHit("exact_repetition", self.position,
                                             run_chars, MIN_REPEATED_CHARS,
                                             period, MIN_REPEATS)
                    return self.hit

            self._history.append((char, self.position))
            self._positions.setdefault(char, deque()).append(self.position)
            if len(self._history) > self.max_period:
                old_char, _ = self._history.popleft()
                old_positions = self._positions[old_char]
                old_positions.popleft()
                if not old_positions:
                    del self._positions[old_char]
        return None


def _index(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def generated_text_pieces(event: dict) -> Iterator[tuple[tuple, str]]:
    """Small field adapter, not semantic/event classification in the detector."""
    typ = event.get("type")
    response_fields = {
        "response.output_text.delta": "text",
        "response.refusal.delta": "refusal",
        "response.reasoning_text.delta": "reasoning",
        "response.reasoning_summary_text.delta": "reasoning_summary",
        "response.function_call_arguments.delta": "tool_arguments",
        "response.custom_tool_call_input.delta": "tool_input",
    }
    if typ in response_fields and isinstance(event.get("delta"), str):
        yield (("responses", response_fields[typ], _index(event.get("output_index")),
                _index(event.get("content_index", event.get("summary_index")))), event["delta"])
    elif typ in ("content_block_delta", "content_block_start"):
        index = _index(event.get("index"))
        block = event.get("delta" if typ == "content_block_delta" else "content_block")
        if not isinstance(block, dict):
            return
        fields = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json",
                  "text": "text", "thinking": "thinking"}
        field = fields.get(block.get("type"))
        if field and isinstance(block.get(field), str):
            # Initial text and later deltas belong to the same content block.
            yield (("anthropic", index, field), block[field])
    elif isinstance(event.get("choices"), list):
        for choice in event["choices"]:
            if not isinstance(choice, dict):
                continue
            index = _index(choice.get("index"))
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            for field in ("content", "reasoning_content", "reasoning"):
                if isinstance(delta.get(field), str):
                    yield (("chat", index, field), delta[field])
            for tool in delta.get("tool_calls") or []:
                if not isinstance(tool, dict):
                    continue
                function = tool.get("function")
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    yield (("chat", index, "tool_arguments", _index(tool.get("index"))), function["arguments"])
            function = delta.get("function_call")
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                yield (("chat", index, "function_arguments"), function["arguments"])


class OutputRepetitionGuard:
    def __init__(self):
        self._detectors: dict[tuple, TextRepetitionDetector] = {}
        self.hit: RepetitionHit | None = None

    def observe(self, event: dict) -> RepetitionHit | None:
        if self.hit is not None:
            return self.hit
        for scope, text in generated_text_pieces(event):
            if not text:
                continue
            detector = self._detectors.get(scope)
            if detector is None:
                detector = self._detectors[scope] = TextRepetitionDetector()
            hit = detector.feed(text)
            if hit is not None:
                self.hit = RepetitionHit(hit.rule, hit.position, hit.run_chars,
                                         hit.threshold, hit.period_chars, hit.min_repeats, scope)
                return self.hit
        return None


def observe_output_repetition(tracker, event: dict) -> bool:
    """Use the tracker's already-parsed event; preserve the cause on cancellation."""
    guard = getattr(tracker, "_output_repetition", None)
    if guard is None:
        tracker._output_repetition = guard = OutputRepetitionGuard()
    hit = guard.observe(event)
    if hit is None:
        return False
    tracker.saw_stream_error = True
    tracker.stream_error_code = OUTPUT_REPETITION_CODE
    tracker.stream_error_message = hit.message
    if hasattr(tracker, "response_failed"):
        tracker.response_failed = True
        tracker.last_event = hit.error_event()
    return True


def output_repetition_error_event(tracker) -> dict | None:
    guard = getattr(tracker, "_output_repetition", None)
    if guard is not None and guard.hit is not None:
        return guard.hit.error_event()
    return None
