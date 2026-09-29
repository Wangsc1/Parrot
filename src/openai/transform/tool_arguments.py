"""Conservative object-argument recovery at JSON-string → Anthropic boundaries.

Only transport wrappers and trailing separators are repaired. Missing values,
truncated JSON, non-object values and ambiguous syntax are not invented inputs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .guard import GuardError


class ToolArgumentsError(GuardError):
    def __init__(self, raw: Any, detail: str, *, tool_name: str = "", request: bool = False):
        self.raw_arguments = raw
        self.original_error = detail
        self.tool_name = tool_name
        super().__init__(
            400 if request else 502,
            "invalid_request_error" if request else "api_error",
            f"Invalid tool arguments for {tool_name or '<unnamed>'}: {detail}; original arguments: {raw!r}",
            param="messages" if request else "arguments",
            scope="request" if request else "candidate",
        )


def _without_trailing_commas(text: str) -> str:
    out = []
    quoted = escaped = False
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "," and text[i + 1:].lstrip().startswith(("}", "]")):
            # A leading/missing-value comma is not a trailing separator.
            before = text[:i].rstrip()
            if before and before[-1] not in "[{,:":
                continue
        out.append(char)
    return "".join(out)


def _reject_constant(value: str):
    raise ValueError(f"non-JSON constant {value}")


def parse_tool_arguments(raw: Any, *, tool_name: str = "", request: bool = False) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ToolArgumentsError(raw, "expected a JSON object", tool_name=tool_name, request=request)
    text = raw.strip().lstrip("\ufeff").strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    original_error = "expected a JSON object"
    # At most one JSON-string envelope; do not recursively interpret tool data.
    for depth in range(2):
        try:
            value = json.loads(text, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            original_error = str(exc)
            repaired = _without_trailing_commas(text)
            if repaired == text:
                break
            try:
                value = json.loads(repaired, parse_constant=_reject_constant)
            except (ValueError, RecursionError):
                break
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and depth == 0:
            text = value.strip()
            continue
        original_error = f"expected a JSON object, got {type(value).__name__}"
        break
    raise ToolArgumentsError(raw, original_error, tool_name=tool_name, request=request)
