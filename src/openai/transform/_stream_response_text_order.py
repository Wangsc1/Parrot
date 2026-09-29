"""Order Responses text fallback frames without delaying a normal first delta.

An item/part which has started but lacks a full-text done snapshot may still grow.
Defer later output positions until that part is closed, then deliver their original
frames. At response completion the ordered output snapshot can close any remaining
part before its deferred successors. This does not invent missing content.
"""
from __future__ import annotations

from typing import Any


class TextOutputOrder:
    def __init__(self, *, max_output_is_error: bool = False) -> None:
        self.max_output_is_error = max_output_is_error
        self.open: set[tuple[int, int]] = set()
        self.deferred: list[tuple[str, dict]] = []
        self.indices: dict[str, int] = {}
        self.seen: set[int] = set()
        self.closed_parts: set[tuple[int, int]] = set()

    def _position(self, name: str, data: dict) -> tuple[int, int] | None:
        index = data.get('output_index')
        if type(index) is not int:
            item = data.get('item')
            item_id = (item.get('id') if isinstance(item, dict) else None) or data.get('item_id')
            index = self.indices.get(item_id)
        if type(index) is not int or index < 0:
            return None
        part = data.get('content_index', 0)
        return index, part if type(part) is int and part >= 0 else 0

    def _blocked(self, name: str, data: dict) -> bool:
        pos = self._position(name, data)
        if pos is None:
            return False
        if name == 'response.output_item.done' and any(i not in self.seen for i in range(pos[0])):
            return True  # A done-only later item must not jump ahead of unseen positions.
        if name in ('response.output_text.delta', 'response.refusal.delta', 'response.content_part.added',
                    'response.output_text.done', 'response.refusal.done', 'response.content_part.done'):
            if any((pos[0], earlier) not in self.closed_parts for earlier in range(pos[1])):
                return True
        if not self.open:
            return False
        first = min(self.open)
        # An added message is a placeholder (part -1), not a reason to hold
        # the first text delta of that very message.
        return pos[0] > first[0] or (pos[0] == first[0] and first[1] >= 0
                                      and pos[1] > first[1] and name != 'response.output_item.done')

    def _observe(self, name: str, data: dict) -> None:
        pos = self._position(name, data)
        if pos is None:
            return
        index, part = pos
        self.seen.add(index)
        item = data.get('item') if isinstance(data.get('item'), dict) else {}
        if isinstance(item.get('id'), str) and item['id']:
            self.indices[item['id']] = index
        if name == 'response.output_item.added' and item.get('type') == 'message':
            self.open.add((index, -1))
        elif name in ('response.output_text.delta', 'response.refusal.delta'):
            self.open.discard((index, -1))
            self.open.add((index, part))
        elif name == 'response.content_part.added' and isinstance(data.get('part'), dict) and data['part'].get('type') in ('output_text', 'refusal'):
            self.open.discard((index, -1))
            self.open.add((index, part))
        elif name in ('response.output_text.done', 'response.refusal.done', 'response.content_part.done'):
            self.closed_parts.add((index, part))
            self.open.discard((index, part))
            if part == 0:
                self.open.discard((index, -1))
        elif name == 'response.output_item.done' and item.get('type') == 'message':
            self.open = {p for p in self.open if p[0] != index}
            self.closed_parts.update((index, ci) for ci, _ in enumerate(item.get('content') or []))

    def _complete_parts(self, index: int, item: dict) -> list[tuple[str, dict]]:
        """Consume each part's queued deltas before its full-text snapshot.

        A whole message snapshot must not overtake the third (or later) part
        still in the queue. Other output items remain deferred until this
        message has been completed by the caller.
        """
        result = []
        part_events = {'response.output_text.delta', 'response.output_text.done',
                       'response.refusal.delta', 'response.refusal.done',
                       'response.content_part.added', 'response.content_part.done'}
        for ci, part in enumerate(item.get('content') or []):
            pending, self.deferred = self.deferred, []
            for frame in pending:
                if frame[0] in part_events and self._position(*frame) == (index, ci):
                    self._observe(*frame)
                    result.append(frame)
                else:
                    self.deferred.append(frame)
            if (index, ci) in self.closed_parts:
                continue
            part = part if isinstance(part, dict) else {}
            kind = part.get('type')
            if kind in ('output_text', 'refusal'):
                name = 'response.output_text.done' if kind == 'output_text' else 'response.refusal.done'
                field = 'text' if kind == 'output_text' else 'refusal'
                frame = (name, {'type': name, 'output_index': index,
                                'item_id': item.get('id'), 'content_index': ci, field: part.get(field, '')})
                self._observe(*frame)
                result.append(frame)
            else:
                self.closed_parts.add((index, ci))
                self.open.discard((index, ci))
        return result

    def _release(self) -> list[tuple[str, dict]]:
        result = []
        while self.deferred:
            # A later part can precede its predecessor's done in this queue.
            # Find the first eligible frame, not merely the queue head.
            eligible = next((i for i, frame in enumerate(self.deferred)
                             if not self._blocked(*frame)), None)
            if eligible is None:
                break
            frame = self.deferred.pop(eligible)
            name, data = frame
            item = data.get('item')
            if name == 'response.output_item.done' and isinstance(item, dict) and item.get('type') == 'message':
                pos = self._position(*frame)
                if pos:
                    result.extend(self._complete_parts(pos[0], item))
            self._observe(*frame)
            result.append(frame)
        return result

    def feed(self, name: str, data: dict) -> list[tuple[str, dict]]:
        item = data.get('item') if isinstance(data.get('item'), dict) else {}
        index = data.get('output_index')
        if type(index) is int and isinstance(item.get('id'), str) and item['id']:
            self.indices[item['id']] = index  # Id lookup must precede any deferred done.
        if name in ('response.failed', 'error') or data.get('type') == 'error':
            self.open.clear()
            self.deferred.clear()
            return [(name, data)]
        if name in ('response.completed', 'response.incomplete'):
            ready: list[tuple[str, dict]] = []
            response = data.get('response') if isinstance(data.get('response'), dict) else {}
            output = response.get('output') if isinstance(response.get('output'), list) else []
            details = response.get('incomplete_details')
            if (self.max_output_is_error and isinstance(details, dict)
                    and details.get('reason') == 'max_output_tokens'):
                self.deferred.clear()
                self.open.clear()
                return [(name, data)]
            while self.deferred:
                later = self._position(*self.deferred[0])
                missing = next((i for i in range(later[0]) if i not in self.seen), None) if later else None
                first_index = min(self.open)[0] if self.open else None
                prior_part = (later[0] if later and any((later[0], ci) not in self.closed_parts
                              for ci in range(later[1])) else None)
                blockers = [i for i in (missing, first_index, prior_part) if i is not None]
                if not blockers:
                    ready.extend(self._release())
                    break
                blocker = min(blockers)
                item: Any = output[blocker] if blocker < len(output) else None
                if isinstance(item, dict):
                    if item.get('type') == 'message':
                        ready.extend(self._complete_parts(blocker, item))
                    snapshot = ('response.output_item.done', {'type': 'response.output_item.done',
                                 'output_index': blocker, 'item': item})
                    self._observe(*snapshot)
                    ready.append(snapshot)
                    if item.get('type') != 'message':
                        self.open = {p for p in self.open if p[0] != blocker}
                else:
                    # An absent snapshot cannot supply any missing content.
                    self.open = {p for p in self.open if p[0] != blocker}
                    self.seen.add(blocker)
                    break  # No full snapshot exists to resolve this position.
                ready.extend(self._release())
            self.open.clear()
            ready.extend(self._release())
            if self.deferred:
                # No earlier item snapshot exists to supply the missing position.
                ready.extend(self.deferred)
                self.deferred.clear()
            ready.append((name, data))
            return ready
        if self._blocked(name, data):
            self.deferred.append((name, data))
            return []
        ready = []
        if name == 'response.output_item.done' and item.get('type') == 'message':
            pos = self._position(name, data)
            if pos:
                ready.extend(self._complete_parts(pos[0], item))
        self._observe(name, data)
        return [*ready, (name, data), *self._release()]
