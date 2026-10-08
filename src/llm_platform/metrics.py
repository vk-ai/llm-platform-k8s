"""Prometheus metrics. Each app instance gets its own registry so tests don't collide."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter("llm_gateway_requests", "Gateway requests", ["route", "status"], registry=r)
        self.backend_requests = Counter("llm_backend_requests", "Successful backend completions", ["backend", "reason"], registry=r)
        self.backend_errors = Counter("llm_backend_errors", "Backend errors", ["backend", "retryable"], registry=r)
        self.failovers = Counter("llm_failovers", "Failovers between backends", ["from_backend", "to_backend"], registry=r)
        self.tokens = Counter("llm_tokens", "Tokens processed", ["backend", "kind"], registry=r)
        self.cost = Counter("llm_cost_usd", "Estimated spend in USD", ["backend"], registry=r)
        self.saved_cost = Counter("llm_cache_saved_cost_usd", "Estimated spend avoided by cache hits", ["cache"], registry=r)
        self.cache = Counter("llm_cache_requests", "Cache lookups", ["cache", "result"], registry=r)
        self.latency = Histogram("llm_backend_latency_seconds", "Backend completion latency", ["backend"], buckets=LATENCY_BUCKETS, registry=r)
        self.request_latency = Histogram("llm_gateway_request_seconds", "End-to-end gateway latency", ["route"], buckets=LATENCY_BUCKETS, registry=r)
        self.retrieval_latency = Histogram("rag_retrieval_seconds", "Vector search latency", ["store"], buckets=LATENCY_BUCKETS, registry=r)
        self.prompt_renders = Counter("prompt_renders", "Prompt renders by version", ["name", "version"], registry=r)

    def observe_completion(self, backend: str, c, cost: float, reason: str) -> None:
        self.backend_requests.labels(backend=backend, reason=reason).inc()
        self.tokens.labels(backend=backend, kind="prompt").inc(c.prompt_tokens)
        self.tokens.labels(backend=backend, kind="completion").inc(c.completion_tokens)
        self.cost.labels(backend=backend).inc(cost)
        self.latency.labels(backend=backend).observe(c.latency_s)

    def render(self) -> bytes:
        return generate_latest(self.registry)
