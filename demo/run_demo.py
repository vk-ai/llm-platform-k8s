"""Offline end-to-end demo of the gateway: prompt registry, RAG, cost routing, caching, failover, metrics.

Mock model backends stand in for vLLM/hosted models. Set PGVECTOR_DSN to also run the RAG part on
Postgres + pgvector, and LLMPLAT_REAL_BASE_URL (+ LLMPLAT_REAL_MODEL) to add a real OpenAI-compatible
backend (e.g. vLLM) as the cheapest tier-1 option.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from llm_platform.app import create_app  # noqa: E402
from llm_platform.backends import BackendSpec, MockBackend, OpenAICompatBackend  # noqa: E402
from llm_platform.config import Settings  # noqa: E402
from llm_platform.gateway import Gateway  # noqa: E402
from llm_platform.rag.stores import PgVectorStore, QdrantVectorStore  # noqa: E402

SPECS = [
    BackendSpec("local-small", "Qwen2.5-0.5B (mock)", tier=1, price_in_per_1k=0.00002, price_out_per_1k=0.00004, max_context=4096, kind="mock"),
    BackendSpec("hosted-medium", "medium (mock)", tier=2, price_in_per_1k=0.00015, price_out_per_1k=0.0006, max_context=32768, kind="mock"),
    BackendSpec("hosted-large", "large (mock)", tier=3, price_in_per_1k=0.0025, price_out_per_1k=0.01, max_context=128000, kind="mock"),
]


def docs():
    return [{"id": p.stem, "text": p.read_text()} for p in sorted((ROOT / "demo/docs").glob("*.md"))]


def ask(c, text, **headers):
    r = c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": text}], "max_tokens": 64}, headers=headers)
    if r.status_code != 200:
        return f"HTTP {r.status_code} {r.json()['detail']}"
    x = r.json()["x_llm"]
    sim = f" sim={x['similarity']:.3f}" if x.get("similarity") else ""
    return f"backend={x['backend']:<13} cache={x['cache']:<8} cost=${x['cost_usd']:.6f} reason={x['reason']}{sim}"


def main() -> None:
    backends = {s.name: MockBackend(s) for s in SPECS}
    real = os.environ.get("LLMPLAT_REAL_BASE_URL")
    if real:
        spec = BackendSpec("real-local", os.environ.get("LLMPLAT_REAL_MODEL", "qwen2.5-0.5b"), tier=1, price_in_per_1k=0.00001,
                           price_out_per_1k=0.00002, max_context=4096, base_url=real, timeout_s=300)
        backends = {"real-local": OpenAICompatBackend(spec), **backends}
    settings = Settings(backends=[b.spec for b in backends.values()], semantic_cache=True, semantic_threshold=0.9,
                        prompt_seed=str(ROOT / "config/prompts.yaml"))
    gw = Gateway(settings, backends=backends, store=QdrantVectorStore(384, location=":memory:"))
    with TestClient(create_app(gateway=gw)) as c:
        print("== 1. prompt registry: publish, label, diff, rollback")
        v = c.post("/v1/prompts/rag-answer/versions", json={"template": "You are a support assistant. Answer in one sentence and cite sources as [n].\nContext:\n{context}\n"}).json()
        print(f"   published rag-answer@{v['version']} sha256={v['sha256'][:12]} variables={v['variables']}")
        print("   labels after promote:", c.put("/v1/prompts/rag-answer/labels/prod", params={"version": "1.1.0"}).json())
        for line in c.get("/v1/prompts/rag-answer/diff", params={"a": "1.0.0", "b": "1.1.0"}).text.splitlines():
            print("   " + line)

        print("\n== 2. RAG on Qdrant (qdrant-client local mode)")
        print("   ingested chunks:", c.post("/v1/rag/ingest", json={"documents": docs(), "max_words": 40, "overlap": 10}).json())
        r = c.post("/v1/rag/query", json={"question": "How long do refunds take for annual plans?", "k": 2}).json()
        print(f"   prompt={r['prompt']['name']}@{r['prompt']['version']} backend={r['backend']} citations={[(x['chunk_id'], x['score']) for x in r['citations']]}")
        print("   rollback prod ->", c.post("/v1/prompts/rag-answer/labels/prod/rollback").json())
        r = c.post("/v1/rag/query", json={"question": "How long do refunds take for annual plans?", "k": 2}).json()
        print(f"   same question now uses rag-answer@{r['prompt']['version']}")

        print("\n== 3. cost-based routing")
        print("   standard quality          ", ask(c, "Summarise the shipping policy"))
        print("   x-llm-quality: medium     ", ask(c, "Summarise the shipping policy for a VIP", **{"x-llm-quality": "medium"}))
        print("   x-llm-quality: high       ", ask(c, "Draft a careful reply to a legal complaint", **{"x-llm-quality": "high"}))
        print("   high + budget $0.0001     ", ask(c, "Draft another careful legal reply", **{"x-llm-quality": "high", "x-llm-max-cost-usd": "0.0001"}))
        long_q = "Please review this transcript: " + "lorem ipsum " * 2200
        print("   prompt > 4k context       ", ask(c, long_q))

        print("\n== 4. prompt caching")
        print("   first ask                 ", ask(c, "How do I reset my password?"))
        print("   identical (exact)         ", ask(c, "How do I reset my password?"))
        print("   reworded (semantic)       ", ask(c, "password reset - how do I do it?"))
        print("   related but different     ", ask(c, "How do I change my password?"))

        print("\n== 5. failover and circuit breaker")
        backends["local-small"].fail_next = 1
        print("   local-small returns 503   ", ask(c, "Where is order 1001?"))
        backends["local-small"].always_fail = True
        for i in range(3):
            ask(c, f"Where is order 20{i}?")
        print("   after 3 failures (open)   ", ask(c, "Where is order 3001?"), "| breaker:", gw.router.plan([{"role": "user", "content": "x"}], 8)[1].get("local-small"))
        backends["local-small"].always_fail = False

        if real:
            print("\n== 5b. real OpenAI-compatible backend at", real)
            r = c.post("/v1/chat/completions", json={"model": "real-local", "messages": [{"role": "user", "content": "In one short sentence, what is a refund?"}], "max_tokens": 40})
            print("   ", r.status_code, r.json()["x_llm"]["backend"], repr(r.json()["choices"][0]["message"]["content"][:120]), r.json()["usage"])

        print("\n== 6. Prometheus metrics (excerpt of /metrics)")
        keep = ("llm_backend_requests_total", "llm_cache_requests_total", "llm_failovers_total", "llm_cost_usd_total", "llm_cache_saved_cost_usd_total", "prompt_renders_total")
        for line in c.get("/metrics").text.splitlines():
            if line.startswith(keep):
                print("   " + line)

    dsn = os.environ.get("PGVECTOR_DSN")
    if dsn:
        print("\n== 7. same RAG flow on Postgres + pgvector")
        store = PgVectorStore(dsn, 384, table="demo_chunks")
        try:
            gw2 = Gateway(settings, backends={s.name: MockBackend(s) for s in SPECS}, store=store)
            with TestClient(create_app(gateway=gw2)) as c:
                print("   ingested chunks:", c.post("/v1/rag/ingest", json={"documents": docs(), "max_words": 40, "overlap": 10}).json())
                r = c.post("/v1/rag/query", json={"question": "How long do refunds take for annual plans?", "k": 2}).json()
                print(f"   citations={[(x['chunk_id'], x['score']) for x in r['citations']]}")
        finally:
            store.drop()


if __name__ == "__main__":
    main()
