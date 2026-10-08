"""Prompt caching at the gateway.

* ``ExactCache``: key = sha256 of the canonical request (resolved route, messages,
  sampling params). LRU with TTL. Only deterministic requests (temperature 0)
  are cached unless the caller opts in.
* ``SemanticCache``: on an exact miss, embed the last user message and reuse a
  cached answer whose embedding has cosine similarity >= threshold, scoped to
  the same system prompt and route. Off by default because a wrong hit returns
  a confidently wrong answer.

Prefix (KV) caching inside the model server is separate: the Helm chart starts
vLLM with ``--enable-prefix-caching``.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class CachedAnswer:
    text: str
    backend: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    created: float


def request_key(scope: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    canon = json.dumps({"scope": scope, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


class ExactCache:
    def __init__(self, max_entries: int = 10_000, ttl_s: float = 3600, clock=time.monotonic):
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.clock = clock
        self._data: OrderedDict[str, tuple[float, CachedAnswer]] = OrderedDict()

    def get(self, key: str) -> CachedAnswer | None:
        item = self._data.get(key)
        if item is None:
            return None
        stored_at, value = item
        if self.clock() - stored_at > self.ttl_s:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: CachedAnswer) -> None:
        self._data[key] = (self.clock(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.9, max_entries: int = 5_000, ttl_s: float = 3600, clock=time.monotonic):
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.clock = clock
        self._vecs: dict[str, list[np.ndarray]] = {}
        self._vals: dict[str, list[tuple[float, CachedAnswer]]] = {}

    @staticmethod
    def scope_of(route_scope: str, messages: list[dict]) -> str:
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        return hashlib.sha256(f"{route_scope}|{system}".encode()).hexdigest()

    @staticmethod
    def query_text(messages: list[dict]) -> str:
        return next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")

    def lookup(self, route_scope: str, messages: list[dict]) -> tuple[CachedAnswer | None, float]:
        scope = self.scope_of(route_scope, messages)
        vecs = self._vecs.get(scope)
        if not vecs:
            return None, 0.0
        q = self.embedder.embed([self.query_text(messages)])[0]
        sims = np.stack(vecs) @ q
        i = int(np.argmax(sims))
        best = float(sims[i])
        stored_at, val = self._vals[scope][i]
        if best >= self.threshold and self.clock() - stored_at <= self.ttl_s:
            return val, best
        return None, best

    def put(self, route_scope: str, messages: list[dict], value: CachedAnswer) -> None:
        scope = self.scope_of(route_scope, messages)
        vec = self.embedder.embed([self.query_text(messages)])[0]
        self._vecs.setdefault(scope, []).append(vec)
        self._vals.setdefault(scope, []).append((self.clock(), value))
        if len(self._vecs[scope]) > self.max_entries:
            self._vecs[scope].pop(0)
            self._vals[scope].pop(0)

    def __len__(self) -> int:
        return sum(len(v) for v in self._vals.values())
