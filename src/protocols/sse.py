"""Shared, byte-oriented Server-Sent Events framing helpers.

SSE permits LF, CRLF and bare CR line endings and blank-line event
separators.  HTTP clients expose wire bytes unchanged, so protocol adapters must
not assume that CRLF has already been normalized for them.
"""

from __future__ import annotations

import re


def _next_event_separator(buf: bytes) -> tuple[int, int]:
    """Return ``(offset, length)`` of the earliest complete SSE separator.

    ``(-1, 0)`` means that *buf* does not yet contain a complete event.  Keeping
    this byte-oriented avoids corrupting a partial UTF-8 sequence while callers
    buffer incremental network reads.
    """
    # Tokenize line endings before looking for adjacent ones. A backtracking
    # (CRLF|CR|LF){2} regex can incorrectly split a *single* CRLF into CR + LF.
    previous = None
    for ending in re.finditer(rb"\r\n|\r|\n", buf):
        if previous is not None and ending.start() == previous.end():
            return previous.start(), ending.end() - previous.start()
        previous = ending
    return -1, 0


def split_sse_events(buf: bytes) -> tuple[bytes, list[bytes]]:
    """Split all complete SSE event blocks from an incremental byte buffer.

    Returned blocks exclude their terminating blank line and retain their
    original internal line endings.  The remaining bytes are an incomplete tail
    to prepend to the next network chunk.
    """
    events: list[bytes] = []
    while True:
        # Empty events carry no data. This also consumes the optional LF when
        # the preceding chunk ended with a CR that dispatched a blank line.
        buf = buf.lstrip(b"\r\n")
        separator_at, separator_len = _next_event_separator(buf)
        if separator_at < 0:
            break
        events.append(buf[:separator_at])
        buf = buf[separator_at + separator_len:]
    return buf, events
