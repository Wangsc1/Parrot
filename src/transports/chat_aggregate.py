"""Chat SSE -> Chat JSON without confusing wire usage with normalized billing."""
from __future__ import annotations

import copy
import time
import uuid

from ..upstream import ChatSSEAssistantBuilder


class ChatAggregateBuilder(ChatSSEAssistantBuilder):
    def __init__(self):
        super().__init__()
        self.metadata = {}
        self.raw_usage = None
        self.logprobs = {}
        self._choice_logprobs = {0: self.logprobs}

    @property
    def done_received(self):
        return self._done_received

    def _apply(self, event):
        super()._apply(event)
        for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if key not in self.metadata and event.get(key) is not None:
                self.metadata[key] = copy.deepcopy(event[key])
        if isinstance(event.get("usage"), dict):
            self.raw_usage = copy.deepcopy(event["usage"])
        for choice in event.get("choices") or []:
            values = choice.get("logprobs")
            if isinstance(values, dict):
                target = self._choice_logprobs.setdefault(int(choice.get("index") or 0), {})
                for key in ("content", "refusal"):
                    if isinstance(values.get(key), list):
                        target.setdefault(key, []).extend(copy.deepcopy(values[key]))

    def to_full_json(self, *, fallback_model=""):
        obj = super().to_full_json(
            id=self.metadata.get("id") or "chatcmpl-" + uuid.uuid4().hex,
            model=self.metadata.get("model") or fallback_model,
            created=self.metadata.get("created") or int(time.time()),
            system_fingerprint=self.metadata.get("system_fingerprint"), usage=self.raw_usage)
        for choice in obj["choices"]:
            values = self._choice_logprobs.get(choice["index"])
            if values:
                choice["logprobs"] = copy.deepcopy(values)
        if "service_tier" in self.metadata:
            obj["service_tier"] = self.metadata["service_tier"]
        return obj
