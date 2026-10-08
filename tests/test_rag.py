import pytest

from llm_platform.embeddings import HashingEmbedder, OpenAIEmbedder
from llm_platform.metrics import Metrics
from llm_platform.prompts import PromptRegistry
from llm_platform.rag.service import RagService, chunk_text
from llm_platform.rag.stores import InMemoryVectorStore


def test_chunking_with_overlap():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text("d", text, max_words=80, overlap=20)
    assert [c.id for c in chunks] == ["d#0", "d#1", "d#2"]
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert chunks[-1].text.split()[-1] == "w199"
    assert chunk_text("d", "   ") == []


def test_retrieve_and_build_messages_uses_label():
    reg = PromptRegistry()
    reg.publish("rag-answer", "v1 context:\n{context}")
    reg.publish("rag-answer", "v2 context:\n{context}")
    reg.set_label("rag-answer", "prod", "1.0.0")
    rag = RagService(HashingEmbedder(), InMemoryVectorStore(384), reg, Metrics())
    rag.ingest([{"id": "a", "text": "Invoices are emailed on the first day of each month."},
                {"id": "b", "text": "Parcels ship from our Leeds warehouse."}])
    hits = rag.retrieve("when are invoices emailed", k=1)
    assert hits[0].chunk.doc_id == "a"
    msgs, version = rag.build_messages("when are invoices emailed", hits)
    assert version == "1.0.0" and msgs[0]["content"].startswith("v1") and "[1] (a)" in msgs[0]["content"]
    _, latest = rag.build_messages("q", hits, label="nonexistent")
    assert latest == "1.1.0"  # falls back to latest when the label isn't set
    assert "(no relevant context found)" in rag.build_messages("q", [])[0][0]["content"]


def test_openai_embedder_normalises():
    import httpx

    def h(req):
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}]})

    e = OpenAIEmbedder("http://vllm/v1", "bge-small", 2, transport=httpx.MockTransport(h))
    v = e.embed(["a", "b"])
    assert v[0].tolist() == pytest.approx([0.6, 0.8]) and v[1].tolist() == pytest.approx([0.0, 1.0])
