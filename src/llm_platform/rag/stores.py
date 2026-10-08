"""Vector stores behind one small interface: in-memory (numpy), Qdrant, pgvector."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class VectorStore(Protocol):
    name: str

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int = 4) -> list[Hit]: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    """Exact cosine search. Used by unit tests and as the zero-dependency default."""

    name = "memory"

    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.vecs = np.zeros((0, dim), dtype=np.float32)
        self.index: dict[str, int] = {}

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for c, v in zip(chunks, vectors):
            if c.id in self.index:
                i = self.index[c.id]
                self.chunks[i], self.vecs[i] = c, v
            else:
                self.index[c.id] = len(self.chunks)
                self.chunks.append(c)
                self.vecs = np.vstack([self.vecs, v[None, :]])

    def search(self, vector: np.ndarray, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        sims = self.vecs @ vector
        top = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i])) for i in top]

    def count(self) -> int:
        return len(self.chunks)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    """Qdrant via qdrant-client. ``location=":memory:"`` runs the client's local mode (no server)."""

    name = "qdrant"

    def __init__(self, dim: int, collection: str = "docs", url: str | None = None, location: str | None = None, api_key: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url, api_key=api_key) if url else QdrantClient(location=location or ":memory:")
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=_point_id(c.id), vector=v.tolist(), payload={"chunk_id": c.id, "doc_id": c.doc_id, "text": c.text, "metadata": c.metadata})
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: np.ndarray, k: int = 4) -> list[Hit]:
        res = self.client.query_points(self.collection, query=vector.tolist(), limit=k, with_payload=True).points
        return [Hit(Chunk(p.payload["chunk_id"], p.payload["doc_id"], p.payload["text"], p.payload.get("metadata") or {}), float(p.score)) for p in res]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count


class PgVectorStore:
    """Postgres + pgvector, cosine distance with an HNSW index."""

    name = "pgvector"

    def __init__(self, dsn: str, dim: int, table: str = "rag_chunks"):
        if not table.isidentifier():
            raise ValueError("table must be a plain identifier")
        import psycopg
        from pgvector.psycopg import register_vector

        self.table = table
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id text PRIMARY KEY, doc_id text NOT NULL, text text NOT NULL, "
            f"metadata jsonb NOT NULL DEFAULT '{{}}', embedding vector({dim}) NOT NULL)"
        )
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)")

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, doc_id, text, metadata, embedding) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO UPDATE SET doc_id=EXCLUDED.doc_id, text=EXCLUDED.text, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                [(c.id, c.doc_id, c.text, json.dumps(c.metadata), v) for c, v in zip(chunks, vectors)],
            )

    def search(self, vector: np.ndarray, k: int = 4) -> list[Hit]:
        rows = self.conn.execute(
            f"SELECT id, doc_id, text, metadata, 1 - (embedding <=> %s) AS score FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
            (vector, vector, k),
        ).fetchall()
        return [Hit(Chunk(r[0], r[1], r[2], r[3] or {}), float(r[4])) for r in rows]

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def drop(self) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")


def make_store(kind: str, dim: int, **kw) -> VectorStore:
    if kind == "memory":
        return InMemoryVectorStore(dim)
    if kind == "qdrant":
        return QdrantVectorStore(dim, collection=kw.get("collection", "docs"), url=kw.get("url") or None, location=kw.get("location"))
    if kind == "pgvector":
        return PgVectorStore(kw["dsn"], dim, table=kw.get("table", "rag_chunks"))
    raise ValueError(f"unknown vector store {kind!r}")
