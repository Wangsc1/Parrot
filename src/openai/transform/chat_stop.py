"""Caller Chat stop compatibility for Responses upstreams (usage is not truncated)."""
from __future__ import annotations

import json
from .anthropic_compat import StopMatcher


def sequences(body: dict | None) -> tuple[str, ...]:
    raw = (body or {}).get('stop')
    if isinstance(raw, str):
        raw = [raw]
    return tuple(dict.fromkeys(s for s in raw if isinstance(s, str) and s)) if isinstance(raw, list) else ()


def apply_responses_stop(obj: dict, body: dict | None) -> dict:
    stops = sequences(body)
    if not stops:
        return obj
    output = []
    for item in obj.get('output') or []:
        if item.get('type') != 'message':
            output.append(item)
            continue
        parts = []
        for part in item.get('content') or []:
            if part.get('type') != 'output_text':
                parts.append(part)
                continue
            matcher = StopMatcher(stops)
            text = matcher.feed(str(part.get('text') or ''), final=True)
            parts.append({**part, 'text': text})
            if matcher.matched:
                output.append({**item, 'content': parts})
                return {**obj, 'output': output, 'status': 'completed', 'incomplete_details': None,
                        'output_text': ''.join(p.get('text', '') for i in output for p in i.get('content') or [] if p.get('type') == 'output_text')}
        output.append(item)
    return obj


class ChatStopStream:
    """Filter translated Chat chunks; drain upstream and keep errors/usage truthful."""
    def __init__(self, inner, body: dict):
        from ...upstream import ChatSSEAssistantBuilder
        self.inner = inner
        self.sequences = sequences(body)
        self.matchers = {}
        self.builder = ChatSSEAssistantBuilder()
        self.buffer = b''

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _filter(self, chunks):
        from ...protocols.sse import split_sse_events
        for raw in chunks:
            self.buffer, events = split_sse_events(self.buffer + raw)
            for event in events:
                data = b'\n'.join(line[5:].lstrip(b' ') for line in event.splitlines() if line.startswith(b'data:'))
                try:
                    obj = json.loads(data)
                except (ValueError, UnicodeError):
                    yield event + b'\n\n'
                    continue
                for choice in obj.get('choices') or []:
                    matcher = self.matchers.setdefault(choice.get('index', 0), StopMatcher(self.sequences))
                    delta = choice.get('delta') or {}
                    if matcher.matched:
                        delta = {}
                    elif isinstance(delta.get('content'), str):
                        text = matcher.feed(delta['content'])
                        delta = {**delta, 'content': text}
                        if matcher.matched:
                            delta = {k: v for k, v in delta.items() if k in ('role', 'content')}
                    if choice.get('finish_reason') is not None:
                        pending = matcher.feed('', final=True)
                        if pending:
                            delta = {**delta, 'content': delta.get('content', '') + pending}
                        if matcher.matched:
                            choice['finish_reason'] = 'stop'
                    choice['delta'] = delta
                    if matcher.matched:
                        choice.pop('logprobs', None)
                out = b'data: ' + json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode() + b'\n\n'
                self.builder.feed(out)
                yield out

    def feed(self, chunk):
        yield from self._filter(self.inner.feed(chunk))

    def close(self):
        yield from self._filter(self.inner.close())
        if self.buffer:
            yield self.buffer
            self.buffer = b''

    def get_downstream_chat_assistant(self):
        return self.builder.get_assistant()
