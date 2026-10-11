"""Model backends. vLLM (and KServe's HF runtime) expose the OpenAI API, so one client covers them."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .tokens import estimate_tokens, messages_tokens


class BackendError(Exception):
    def __init__(self, message: str, retryable: bool = True, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class BackendSpec:
    name: str
    model: str
    tier: int = 1                     # 1 = small/cheap, 2 = mid, 3 = large
    price_in_per_1k: float = 0.0      # USD per 1k prompt tokens (self-hosted: amortised GPU/CPU cost)
    price_out_per_1k: float = 0.0
    max_context: int = 4096
    kind: str = "openai"              # "openai" (vLLM, KServe, hosted) or "mock"
    base_url: str = ""
    api_key: str = ""
    timeout_s: float = 30.0
    extra: dict = field(default_factory=dict)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens / 1000 * self.price_in_per_1k + completion_tokens / 1000 * self.price_out_per_1k


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_s: float
    usage_source: str = "upstream"   # "upstream" (backend reported it) or "estimated" (gateway guessed)


# Stream events: ("delta", str) zero or more times, then exactly one ("done", Completion).
StreamEvent = tuple[str, object]


class OpenAICompatBackend:
    def __init__(self, spec: BackendSpec, transport: httpx.AsyncBaseTransport | None = None):
        self.spec = spec
        headers = {"authorization": f"Bearer {spec.api_key}"} if spec.api_key else {}
        self.client = httpx.AsyncClient(base_url=spec.base_url.rstrip("/"), timeout=spec.timeout_s, headers=headers, transport=transport)

    async def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        t0 = time.perf_counter()
        try:
            r = await self.client.post(
                "/chat/completions",
                json={"model": self.spec.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            )
        except httpx.TimeoutException as exc:
            raise BackendError(f"{self.spec.name}: timeout", retryable=True) from exc
        except httpx.TransportError as exc:
            raise BackendError(f"{self.spec.name}: {exc.__class__.__name__}", retryable=True) from exc
        if r.status_code == 429 or r.status_code >= 500:
            raise BackendError(f"{self.spec.name}: HTTP {r.status_code}", retryable=True, status=r.status_code)
        if r.status_code >= 400:
            raise BackendError(f"{self.spec.name}: HTTP {r.status_code} {r.text[:200]}", retryable=False, status=r.status_code)
        body = r.json()
        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or messages_tokens(messages)),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            model=body.get("model", self.spec.model),
            latency_s=time.perf_counter() - t0,
        )

    async def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[StreamEvent]:
        """Stream from an OpenAI-compatible server, asking it for a final usage chunk.

        Errors before the first byte raise BackendError (so the router can fail over).
        If the upstream ignores include_usage, tokens are estimated and marked so.
        """
        t0 = time.perf_counter()
        payload = {"model": self.spec.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as r:
                if r.status_code == 429 or r.status_code >= 500:
                    raise BackendError(f"{self.spec.name}: HTTP {r.status_code}", retryable=True, status=r.status_code)
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")
                    raise BackendError(f"{self.spec.name}: HTTP {r.status_code} {body[:200]}", retryable=False, status=r.status_code)
                parts: list[str] = []
                usage: dict = {}
                model = self.spec.model
                done = False
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    model = chunk.get("model", model)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            parts.append(piece)
                            yield ("delta", piece)
                if not done:
                    raise BackendError(f"{self.spec.name}: stream ended without [DONE]", retryable=False)
        except httpx.TimeoutException as exc:
            raise BackendError(f"{self.spec.name}: timeout", retryable=True) from exc
        except httpx.TransportError as exc:
            raise BackendError(f"{self.spec.name}: {exc.__class__.__name__}", retryable=True) from exc
        text = "".join(parts)
        yield ("done", Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or messages_tokens(messages)),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            model=model,
            latency_s=time.perf_counter() - t0,
            usage_source="upstream" if usage else "estimated",
        ))

    async def aclose(self) -> None:
        await self.client.aclose()


class MockBackend:
    """Deterministic stand-in for a model server (tests, demo, docker-compose without vLLM).

    Fault injection: ``fail_next`` errors, or ``always_fail``.
    """

    def __init__(self, spec: BackendSpec, delay_s: float = 0.0):
        self.spec = spec
        self.delay_s = delay_s
        self.fail_next = 0
        self.always_fail = False
        self.calls = 0
        self.fail_mid_stream: int | None = None

    async def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.always_fail or self.fail_next > 0:
            self.fail_next = max(0, self.fail_next - 1)
            raise BackendError(f"{self.spec.name}: HTTP 503 (injected)", retryable=True, status=503)
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        words = str(user).split()
        text = f"[{self.spec.name}] " + " ".join(words[: min(len(words), max(8, max_tokens // 4))])
        return Completion(text, messages_tokens(messages), estimate_tokens(text), self.spec.model, self.delay_s)

    async def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[StreamEvent]:
        """Word-by-word stream of the same text ``complete`` returns. ``fail_mid_stream`` breaks after N words."""
        c = await self.complete(messages, max_tokens, temperature)
        words = c.text.split(" ")
        for i, w in enumerate(words):
            if self.fail_mid_stream is not None and i >= self.fail_mid_stream:
                raise BackendError(f"{self.spec.name}: stream dropped (injected)", retryable=False)
            yield ("delta", w if i == 0 else " " + w)
        c.usage_source = "estimated"
        yield ("done", c)

    async def aclose(self) -> None:
        return None


def make_backend(spec: BackendSpec):
    if spec.kind == "mock":
        return MockBackend(spec, delay_s=float(spec.extra.get("delay_s", 0.0)))
    return OpenAICompatBackend(spec)
