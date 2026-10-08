"""Gateway core (HTTP-agnostic): cache lookup -> cost router -> cache fill -> metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .backends import make_backend
from .cache import CachedAnswer, ExactCache, SemanticCache, request_key
from .config import Settings
from .embeddings import HashingEmbedder, OpenAIEmbedder
from .metrics import Metrics
from .prompts import PromptRegistry
from .rag.service import RagService
from .rag.stores import make_store
from .router import CostRouter


@dataclass
class ChatResult:
    text: str
    model: str
    backend: str | None
    cache: str            # miss | exact | semantic | off
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    tried: list[str]
    reason: str
    similarity: float | None = None


class Gateway:
    def __init__(self, settings: Settings, backends: dict | None = None, metrics: Metrics | None = None, store=None, embedder=None, registry=None):
        self.settings = settings
        self.metrics = metrics or Metrics()
        self.backends = backends or {s.name: make_backend(s) for s in settings.backends}
        self.router = CostRouter(self.backends, self.metrics)
        if embedder is None:
            e = settings.embedder
            embedder = OpenAIEmbedder(e["base_url"], e["model"], int(e["dim"]), e.get("api_key", "")) if e.get("base_url") else HashingEmbedder(settings.embedding_dim)
        self.embedder = embedder
        self.exact = ExactCache(ttl_s=settings.exact_cache_ttl_s) if settings.exact_cache else None
        self.semantic = SemanticCache(self.embedder, settings.semantic_threshold) if settings.semantic_cache else None
        self.registry = registry or PromptRegistry(settings.registry_path)
        if settings.prompt_seed:
            self.registry.load_seed(settings.prompt_seed)
        self.store = store or make_store(settings.vector_store, self.embedder.dim, **settings.vector_store_options)
        self.rag = RagService(self.embedder, self.store, self.registry, self.metrics)

    def resolve_tier(self, quality: str | int | None) -> int:
        if quality is None:
            return 1
        if isinstance(quality, int) or str(quality).isdigit():
            return int(quality)
        return int(self.settings.tier_aliases.get(str(quality), 1))

    async def chat(self, messages: list[dict], model: str = "auto", max_tokens: int = 256, temperature: float = 0.0,
                   quality: str | int | None = None, max_cost_usd: float | None = None, cache: str = "auto") -> ChatResult:
        min_tier = self.resolve_tier(quality)
        pinned = None if model in ("auto", "", None) else model
        scope = f"pinned:{pinned}" if pinned else f"auto:tier>={min_tier}"
        cacheable = cache != "off" and (temperature == 0 or self.settings.cache_nondeterministic or cache == "force")
        key = request_key(scope, messages, max_tokens, temperature)
        if cacheable and self.exact is not None:
            hit = self.exact.get(key)
            self.metrics.cache.labels(cache="exact", result="hit" if hit else "miss").inc()
            if hit:
                self.metrics.saved_cost.labels(cache="exact").inc(hit.cost_usd)
                return ChatResult(hit.text, hit.model, hit.backend, "exact", 0.0, hit.prompt_tokens, hit.completion_tokens, [], "cache")
        if cacheable and self.semantic is not None:
            hit, sim = self.semantic.lookup(scope, messages)
            self.metrics.cache.labels(cache="semantic", result="hit" if hit else "miss").inc()
            if hit:
                self.metrics.saved_cost.labels(cache="semantic").inc(hit.cost_usd)
                return ChatResult(hit.text, hit.model, hit.backend, "semantic", 0.0, hit.prompt_tokens, hit.completion_tokens, [], "cache", sim)
        r = await self.router.complete(messages, max_tokens, temperature, min_tier, max_cost_usd, pinned)
        c = r.completion
        if cacheable:
            entry = CachedAnswer(c.text, r.backend, c.model, c.prompt_tokens, c.completion_tokens, r.cost_usd, time.monotonic())
            if self.exact is not None:
                self.exact.put(key, entry)
            if self.semantic is not None:
                self.semantic.put(scope, messages, entry)
        return ChatResult(c.text, c.model, r.backend, "miss" if cacheable else "off", r.cost_usd, c.prompt_tokens, c.completion_tokens, r.tried, r.reason)

    async def rag_query(self, question: str, k: int = 4, label: str | None = "prod", version: str | None = None, **chat_kw) -> dict:
        hits = self.rag.retrieve(question, k)
        messages, prompt_version = self.rag.build_messages(question, hits, label=label, version=version)
        res = await self.chat(messages, **chat_kw)
        return {
            "answer": res.text,
            "prompt": {"name": self.rag.prompt_name, "version": prompt_version},
            "citations": [{"chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id, "score": round(h.score, 4)} for h in hits],
            "backend": res.backend,
            "cache": res.cache,
            "cost_usd": res.cost_usd,
        }

    async def aclose(self) -> None:
        for b in self.backends.values():
            await b.aclose()
