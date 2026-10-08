import httpx
import pytest
from fastapi.testclient import TestClient

from llm_platform.app import create_app
from llm_platform.gateway import Gateway
from llm_platform.rag.stores import QdrantVectorStore

DOCS = [
    {"id": "refunds", "text": "Refunds are issued to the original payment method within 10 business days of approval."},
    {"id": "shipping", "text": "Standard delivery takes 3 to 5 working days. Express delivery takes 1 working day."},
    {"id": "security", "text": "Reset your password from Settings, then Security. Two-factor authentication is optional."},
]


@pytest.fixture
def client(settings, mock_backends):
    gw = Gateway(settings, backends=mock_backends, store=QdrantVectorStore(384, location=":memory:"))
    with TestClient(create_app(gateway=gw)) as c:
        c.gw = gw
        yield c


def chat(client, text, **headers):
    return client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": text}], "max_tokens": 64}, headers=headers)


def test_health_models_ready(client):
    assert client.get("/healthz").json() == {"ok": True}
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["auto", "local-small", "hosted-medium", "hosted-large"]
    assert client.get("/readyz").json()["vector_store"] == "qdrant"


def test_chat_routes_to_cheapest_and_reports_cost(client):
    r = chat(client, "What is the refund window for annual plans?")
    assert r.status_code == 200
    body = r.json()
    assert r.headers["x-llm-backend"] == "local-small" and r.headers["x-llm-cache"] == "miss"
    assert float(r.headers["x-llm-cost-usd"]) > 0
    assert body["object"] == "chat.completion" and body["usage"]["total_tokens"] > 0


def test_exact_then_semantic_cache(client):
    chat(client, "How do I reset my password?")
    r2 = chat(client, "How do I reset my password?")
    assert r2.headers["x-llm-cache"] == "exact" and float(r2.headers["x-llm-cost-usd"]) == 0
    r3 = chat(client, "how do i reset my password")
    assert r3.headers["x-llm-cache"] == "semantic" and r3.json()["x_llm"]["similarity"] >= 0.9
    r4 = chat(client, "How do I reset my password?", **{"x-llm-cache": "off"})
    assert r4.headers["x-llm-cache"] == "off"
    m = client.get("/metrics").text
    assert 'llm_cache_requests_total{cache="exact",result="hit"} 1.0' in m
    assert 'llm_cache_requests_total{cache="semantic",result="hit"} 1.0' in m


def test_nonzero_temperature_not_cached(client):
    for _ in range(2):
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "write a haiku"}], "temperature": 0.8})
    assert r.headers["x-llm-cache"] == "off"


def test_quality_header_and_budget(client):
    assert chat(client, "Draft a careful legal reply", **{"x-llm-quality": "high"}).headers["x-llm-backend"] == "hosted-large"
    r = chat(client, "Draft a careful legal reply please", **{"x-llm-quality": "high", "x-llm-max-cost-usd": "0.00001"})
    assert r.status_code == 422 and "budget" in r.json()["detail"]["rejected"]["hosted-large"]


def test_failover_and_all_down(client):
    client.gw.backends["local-small"].fail_next = 1
    r = chat(client, "Where is my order 123?")
    assert r.headers["x-llm-backend"] == "hosted-medium" and r.headers["x-llm-route-reason"] == "failover"
    for b in client.gw.backends.values():
        b.always_fail = True
    r = chat(client, "Where is my order 456?")
    assert r.status_code == 502
    assert 'llm_gateway_requests_total{route="chat",status="upstream_error"} 1.0' in client.get("/metrics").text


def test_stream_rejected(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "stream": True})
    assert r.status_code == 400


def test_rag_and_prompt_rollout(client):
    assert client.post("/v1/rag/ingest", json={"documents": DOCS}).json()["chunks"] == 3
    r = client.post("/v1/rag/query", json={"question": "how long do refunds take", "k": 2}).json()
    assert r["citations"][0]["doc_id"] == "refunds" and r["prompt"] == {"name": "rag-answer", "version": "1.0.0"}

    new = client.post("/v1/prompts/rag-answer/versions", json={"template": "Answer in one sentence and cite [n].\n{context}"})
    assert new.status_code == 201 and new.json()["version"] == "1.1.0"
    assert client.put("/v1/prompts/rag-answer/labels/prod", params={"version": "1.1.0"}).json() == {"prod": "1.1.0"}
    r = client.post("/v1/rag/query", json={"question": "how long do refunds take", "k": 2}).json()
    assert r["prompt"]["version"] == "1.1.0"
    diff = client.get("/v1/prompts/rag-answer/diff", params={"a": "1.0.0", "b": "1.1.0"}).text
    assert "+Answer in one sentence" in diff
    assert client.post("/v1/prompts/rag-answer/labels/prod/rollback").json() == {"label": "prod", "version": "1.0.0"}
    pinned = client.post("/v1/rag/query", json={"question": "refund time", "prompt_version": "1.1.0"}).json()
    assert pinned["prompt"]["version"] == "1.1.0"
    assert client.get("/v1/prompts").json()["rag-answer"]["labels"] == {"prod": "1.0.0"}
    m = client.get("/metrics").text
    assert 'prompt_renders_total{name="rag-answer",version="1.1.0"}' in m and "rag_retrieval_seconds_bucket" in m


def test_prompt_errors(client):
    assert client.get("/v1/prompts/missing").status_code == 404
    assert client.post("/v1/prompts/rag-answer/versions", json={"template": "x {context}", "version": "0.0.1"}).status_code == 409
    assert client.post("/v1/prompts/rag-answer/labels/prod/rollback").status_code == 409
    assert client.post("/v1/rag/query", json={"question": "q", "prompt_version": "9.9.9"}).status_code == 404


def test_load_settings_and_default_app(tmp_path, monkeypatch):
    from conftest import ROOT
    from llm_platform.config import load_settings

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("LLMPLAT_VECTOR_STORE", "memory")
    s = load_settings("config/platform.yaml")
    assert [b.name for b in s.backends] == ["local-small", "hosted-medium", "hosted-large"] and s.semantic_cache
    with TestClient(create_app(s)) as c:
        assert chat(c, "hello").headers["x-llm-backend"] == "local-small"
        assert c.get("/v1/prompts/rag-answer", params={"label": "prod"}).json()["version"] == "1.0.0"
