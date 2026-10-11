"""Cost-based routing with failover and a per-backend circuit breaker."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .backends import BackendError, BackendSpec, Completion
from .metrics import Metrics
from .tokens import messages_tokens


class NoEligibleBackend(Exception):
    def __init__(self, message: str, rejected: dict[str, str]):
        super().__init__(message)
        self.rejected = rejected


class AllBackendsFailed(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    reset_after_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    def available(self, now: float) -> bool:
        if self.opened_at is None:
            return True
        return now - self.opened_at >= self.reset_after_s  # half-open: allow one probe

    def record_success(self) -> None:
        self.failures, self.opened_at = 0, None

    def record_failure(self, now: float) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = now


@dataclass
class RouteResult:
    completion: Completion
    backend: str
    cost_usd: float
    estimated_cost_usd: float
    tried: list[str] = field(default_factory=list)
    reason: str = "cheapest"


class CostRouter:
    """Pick the cheapest backend that satisfies the request, fail over in cost order.

    Eligibility: tier >= requested min tier, prompt + max_tokens fits the context,
    estimated cost <= the request's budget, circuit closed (or half-open).
    """

    def __init__(self, backends: dict[str, object], metrics: Metrics, clock=time.monotonic, breaker_threshold: int = 3, breaker_reset_s: float = 30.0):
        self.backends = backends
        self.metrics = metrics
        self.clock = clock
        self.breakers = {name: CircuitBreaker(breaker_threshold, breaker_reset_s) for name in backends}

    def spec(self, name: str) -> BackendSpec:
        return self.backends[name].spec  # type: ignore[attr-defined]

    def plan(self, messages: list[dict], max_tokens: int, min_tier: int = 1, max_cost_usd: float | None = None, pinned: str | None = None) -> tuple[list[tuple[float, str]], dict[str, str]]:
        prompt_tokens = messages_tokens(messages)
        now = self.clock()
        eligible, rejected = [], {}
        names = [pinned] if pinned else list(self.backends)
        for name in names:
            if name not in self.backends:
                rejected[name] = "unknown backend"
                continue
            s = self.spec(name)
            est = s.estimate_cost(prompt_tokens, max_tokens)
            if s.tier < min_tier:
                rejected[name] = f"tier {s.tier} < {min_tier}"
            elif prompt_tokens + max_tokens > s.max_context:
                rejected[name] = f"context {prompt_tokens + max_tokens} > {s.max_context}"
            elif max_cost_usd is not None and est > max_cost_usd:
                rejected[name] = f"est ${est:.6f} > budget ${max_cost_usd:.6f}"
            elif not self.breakers[name].available(now):
                rejected[name] = "circuit open"
            else:
                eligible.append((est, s.tier, name))
        eligible.sort()  # cheapest first; ties -> lower tier first, then name
        return [(e, n) for e, _, n in eligible], rejected

    async def complete(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0, min_tier: int = 1, max_cost_usd: float | None = None, pinned: str | None = None) -> RouteResult:
        order, rejected = self.plan(messages, max_tokens, min_tier, max_cost_usd, pinned)
        if not order:
            raise NoEligibleBackend("no backend satisfies tier/context/budget/health", rejected)
        errors, tried = [], []
        for i, (est, name) in enumerate(order):
            tried.append(name)
            backend = self.backends[name]
            try:
                c = await backend.complete(messages, max_tokens, temperature)  # type: ignore[attr-defined]
            except BackendError as exc:
                self.breakers[name].record_failure(self.clock())
                self.metrics.backend_errors.labels(backend=name, retryable=str(exc.retryable).lower()).inc()
                errors.append(str(exc))
                if not exc.retryable:
                    break
                if i + 1 < len(order):
                    self.metrics.failovers.labels(from_backend=name, to_backend=order[i + 1][1]).inc()
                continue
            self.breakers[name].record_success()
            cost = self.spec(name).estimate_cost(c.prompt_tokens, c.completion_tokens)
            reason = "pinned" if pinned else ("cheapest" if i == 0 else "failover")
            self.metrics.observe_completion(name, c, cost, reason)
            return RouteResult(c, name, cost, est, tried, reason)
        raise AllBackendsFailed(errors)

    async def stream(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0, min_tier: int = 1,
                     max_cost_usd: float | None = None, pinned: str | None = None) -> AsyncIterator[tuple[str, object]]:
        """Streaming twin of ``complete``.

        Failover only happens before the first token reaches the client: once a
        delta has been yielded, a backend error is re-raised (we can't un-send text).
        Yields ("delta", str)... then ("done", RouteResult).
        """
        order, rejected = self.plan(messages, max_tokens, min_tier, max_cost_usd, pinned)
        if not order:
            raise NoEligibleBackend("no backend satisfies tier/context/budget/health", rejected)
        errors, tried = [], []
        for i, (est, name) in enumerate(order):
            tried.append(name)
            started = False
            try:
                async for kind, val in self.backends[name].stream(messages, max_tokens, temperature):  # type: ignore[attr-defined]
                    if kind == "delta":
                        started = True
                        yield ("delta", val)
                    else:
                        c = val
            except BackendError as exc:
                self.breakers[name].record_failure(self.clock())
                self.metrics.backend_errors.labels(backend=name, retryable=str(exc.retryable).lower()).inc()
                if started:
                    raise
                errors.append(str(exc))
                if not exc.retryable:
                    break
                if i + 1 < len(order):
                    self.metrics.failovers.labels(from_backend=name, to_backend=order[i + 1][1]).inc()
                continue
            self.breakers[name].record_success()
            cost = self.spec(name).estimate_cost(c.prompt_tokens, c.completion_tokens)
            reason = "pinned" if pinned else ("cheapest" if i == 0 else "failover")
            self.metrics.observe_completion(name, c, cost, reason)
            yield ("done", RouteResult(c, name, cost, est, tried, reason))
            return
        raise AllBackendsFailed(errors)
