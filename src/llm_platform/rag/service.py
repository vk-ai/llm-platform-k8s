"""RAG: chunk -> embed -> upsert; query -> retrieve -> render registry prompt -> routed completion."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..prompts import NotFound
from .stores import Chunk, Hit


def chunk_text(doc_id: str, text: str, max_words: int = 80, overlap: int = 20, metadata: dict | None = None) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    words: list[str] = []
    for p in paragraphs:
        words += p.split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = words[start:start + max_words]
        chunks.append(Chunk(f"{doc_id}#{i}", doc_id, " ".join(piece), dict(metadata or {})))
        if start + max_words >= len(words):
            break
    return chunks


@dataclass
class RagAnswer:
    answer: str
    hits: list[Hit]
    prompt_version: str
    backend: str | None
    cost_usd: float
    cache: str


class RagService:
    def __init__(self, embedder, store, registry, metrics, prompt_name: str = "rag-answer"):
        self.embedder = embedder
        self.store = store
        self.registry = registry
        self.metrics = metrics
        self.prompt_name = prompt_name

    def ingest(self, docs: list[dict], max_words: int = 80, overlap: int = 20) -> int:
        chunks: list[Chunk] = []
        for d in docs:
            chunks += chunk_text(d["id"], d["text"], max_words, overlap, d.get("metadata"))
        if chunks:
            self.store.upsert(chunks, self.embedder.embed([c.text for c in chunks]))
        return len(chunks)

    def retrieve(self, question: str, k: int = 4, min_score: float = 0.0) -> list[Hit]:
        t0 = time.perf_counter()
        hits = self.store.search(self.embedder.embed([question])[0], k)
        self.metrics.retrieval_latency.labels(store=self.store.name).observe(time.perf_counter() - t0)
        return [h for h in hits if h.score >= min_score]

    def build_messages(self, question: str, hits: list[Hit], label: str | None = "prod", version: str | None = None) -> tuple[list[dict], str]:
        if version:
            prompt = self.registry.get(self.prompt_name, version=version)  # explicit pin: NotFound propagates
        else:
            try:
                prompt = self.registry.get(self.prompt_name, label=label)
            except NotFound:  # label not set yet: fall back to the latest version
                prompt = self.registry.get(self.prompt_name)
        context = "\n".join(f"[{i + 1}] ({h.chunk.doc_id}) {h.chunk.text}" for i, h in enumerate(hits)) or "(no relevant context found)"
        system = prompt.render(context=context)
        self.metrics.prompt_renders.labels(name=prompt.name, version=prompt.version).inc()
        return [{"role": "system", "content": system}, {"role": "user", "content": question}], prompt.version
