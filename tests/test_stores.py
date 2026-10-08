import os
import uuid

import numpy as np
import pytest

from llm_platform.embeddings import HashingEmbedder
from llm_platform.rag.stores import Chunk, InMemoryVectorStore, PgVectorStore, QdrantVectorStore

DOCS = [
    Chunk("refunds#0", "refunds", "Refunds are issued to the original payment method within 10 business days."),
    Chunk("shipping#0", "shipping", "Standard delivery takes 3 to 5 working days; express delivery takes 1 day."),
    Chunk("security#0", "security", "Reset your password from Settings > Security. Two-factor authentication is optional."),
]
QUERIES = {"how long until my refund arrives": "refunds", "express delivery time": "shipping", "how do I reset my password": "security"}
E = HashingEmbedder(384)


def check_store(store):
    store.upsert(DOCS, E.embed([d.text for d in DOCS]))
    store.upsert(DOCS[:1], E.embed([DOCS[0].text]))  # upsert is idempotent
    assert store.count() == 3
    for q, doc in QUERIES.items():
        hits = store.search(E.embed([q])[0], k=2)
        assert hits[0].chunk.doc_id == doc
        assert hits[0].score >= hits[1].score
        assert -1.0 <= hits[0].score <= 1.0 + 1e-6
    return [[(h.chunk.id, round(h.score, 4)) for h in store.search(E.embed([q])[0], k=3)] for q in QUERIES]


def test_memory_store():
    check_store(InMemoryVectorStore(384))
    assert InMemoryVectorStore(8).search(np.zeros(8, dtype=np.float32)) == []


def test_qdrant_local_mode_matches_memory():
    """qdrant-client's local mode runs the real client API in-process (no server)."""
    mem = check_store(InMemoryVectorStore(384))
    qd = check_store(QdrantVectorStore(384, collection=f"t{uuid.uuid4().hex[:6]}", location=":memory:"))
    assert [[c for c, _ in row] for row in qd] == [[c for c, _ in row] for row in mem]
    for rq, rm in zip(qd, mem):
        for (_, a), (_, b) in zip(rq, rm):
            assert a == pytest.approx(b, abs=1e-3)


@pytest.mark.pgvector
@pytest.mark.skipif(not os.environ.get("PGVECTOR_DSN"), reason="set PGVECTOR_DSN to a Postgres with pgvector")
def test_pgvector_matches_memory():
    store = PgVectorStore(os.environ["PGVECTOR_DSN"], 384, table=f"t_{uuid.uuid4().hex[:8]}")
    try:
        mem = check_store(InMemoryVectorStore(384))
        pg = check_store(store)
        assert [[c for c, _ in row] for row in pg] == [[c for c, _ in row] for row in mem]
        idx = store.conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = %s", (store.table,)).fetchall()
        assert any("hnsw" in r[0] and "vector_cosine_ops" in r[0] for r in idx)
    finally:
        store.drop()


def test_pg_table_name_validated():
    with pytest.raises(ValueError):
        PgVectorStore.__init__(object.__new__(PgVectorStore), "postgresql://x", 3, table="x; drop table y")
