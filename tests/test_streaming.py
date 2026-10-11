"""Gateway streaming with include_usage + cost on the final chunk.

Shares test data with vk-ai/openai-compat-inference-stub:
tests/fixtures/include_usage_stream.{json,sse} are copied from that repo, where a
test proves the .sse transcript is exactly what the stub emits.
"""
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_platform.app import create_app
from llm_platform.backends import BackendError, BackendSpec, OpenAICompatBackend
from llm_platform.gateway import Gateway
from llm_platform.rag.stores import QdrantVectorStore

FIX = Path(__file__).parent / "fixtures"
CONTRACT = json.loads((FIX / "include_usage_stream.json").read_text())
TRANSCRIPT = (FIX / "include_usage_stream.sse").read_text()


@pytest.fixture
def client(settings, mock_backends):
    gw = Gateway(settings, backends=mock_backends, store=QdrantVectorStore(384, location=":memory:"))
    with TestClient(create_app(gateway=gw)) as c:
        c.gw = gw
        yield c


def events(raw: str) -> list:
    out = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            p = line[6:]
            out.append("[DONE]" if p.strip() == "[DONE]" else json.loads(p))
    return out


def stream(client, text, include_usage=True, **headers):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 64, "stream": True}
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
        return r.status_code, events(r.read().decode())


def check_contract(ev):
    """Same rules the stub is tested against."""
    rules = CONTRACT["rules"]
    assert ev[-1] == "[DONE]"
    last = ev[-2]
    assert last["choices"] == rules["usage_chunk_choices"]
    assert sorted(last["usage"]) == sorted(rules["usage_fields"])
    u = last["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    for c in ev[:-2]:
        assert "usage" in c and c["usage"] is None
    return last


def test_stream_matches_non_stream_and_reports_cost(client):
    q = "What is the refund window for annual plans?"
    status, ev = stream(client, q, **{"x-llm-cache": "off"})
    assert status == 200
    last = check_contract(ev)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in ev[:-2])
    non = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": q}], "max_tokens": 64},
                      headers={"x-llm-cache": "off"}).json()
    assert text == non["choices"][0]["message"]["content"]
    assert last["usage"] == non["usage"]
    assert last["x_llm"]["cost_usd"] == pytest.approx(non["x_llm"]["cost_usd"]) and last["x_llm"]["cost_usd"] > 0
    assert last["x_llm"]["backend"] == "local-small"
    assert ev[0]["choices"][0]["delta"]["role"] == "assistant"
    assert ev[-3]["choices"][0]["finish_reason"] == "stop"


def test_without_include_usage_no_usage_chunk(client):
    _, ev = stream(client, "hello there", include_usage=False)
    assert ev[-1] == "[DONE]" and all("usage" not in c for c in ev[:-1])


def test_stream_fills_and_hits_cache(client):
    stream(client, "Where is my parcel 77?")
    _, ev = stream(client, "Where is my parcel 77?")
    last = check_contract(ev)
    assert last["x_llm"]["cache"] == "exact" and last["x_llm"]["cost_usd"] == 0.0
    assert 'llm_gateway_requests_total{route="chat_stream",status="cache_exact"} 1.0' in client.get("/metrics").text


def test_failover_before_first_token(client):
    client.gw.backends["local-small"].fail_next = 1
    _, ev = stream(client, "Where is my order 9?")
    last = check_contract(ev)
    assert last["x_llm"]["backend"] == "hosted-medium" and last["x_llm"]["reason"] == "failover"


def test_all_down_is_http_502(client):
    for b in client.gw.backends.values():
        b.always_fail = True
    status, _ = stream(client, "anything")
    assert status == 502


def test_mid_stream_failure_sends_error_not_done(client):
    client.gw.backends["local-small"].fail_mid_stream = 2
    _, ev = stream(client, "one two three four five six seven eight nine", **{"x-llm-cache": "off"})
    assert "[DONE]" not in ev and "error" in ev[-1]
    assert not [c for c in ev if isinstance(c, dict) and c.get("choices") == []]
    # partial answer must not be cached: next (healthy) call is a real miss
    client.gw.backends["local-small"].fail_mid_stream = None
    _, ev2 = stream(client, "one two three four five six seven eight nine")
    assert ev2[-1] == "[DONE]" and ev2[-2]["x_llm"]["cache"] == "miss"
    assert 'route="chat_stream",status="stream_error"} 1.0' in client.get("/metrics").text


def _replay_backend(body: str, status: int = 200) -> tuple[OpenAICompatBackend, list]:
    seen = []

    def handler(req: httpx.Request):
        seen.append(json.loads(req.content))
        return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})

    spec = BackendSpec("stub", "stub-model", price_in_per_1k=1.0, price_out_per_1k=2.0, base_url="http://stub/v1")
    return OpenAICompatBackend(spec, transport=httpx.MockTransport(handler)), seen


async def _collect(b, messages):
    out = []
    async for ev in b.stream(messages, 64, 0.0):
        out.append(ev)
    return out


def test_replays_recorded_stub_transcript_with_upstream_usage():
    b, seen = _replay_backend(TRANSCRIPT)
    out = asyncio.run(_collect(b, CONTRACT["request"]["messages"]))
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}
    text = "".join(v for k, v in out if k == "delta")
    assert "echo=count my tokens please" in text
    done = out[-1][1]
    expected = events(TRANSCRIPT)[-2]["usage"]
    assert (done.prompt_tokens, done.completion_tokens) == (expected["prompt_tokens"], expected["completion_tokens"])
    assert done.usage_source == "upstream"
    assert b.spec.estimate_cost(done.prompt_tokens, done.completion_tokens) == pytest.approx((5 * 1.0 + 8 * 2.0) / 1000)


def test_upstream_without_usage_is_marked_estimated():
    no_usage = "\n".join(l for l in TRANSCRIPT.splitlines() if '"choices":[]' not in l)
    b, _ = _replay_backend(no_usage)
    done = asyncio.run(_collect(b, CONTRACT["request"]["messages"]))[-1][1]
    assert done.usage_source == "estimated" and done.completion_tokens > 0


def test_truncated_upstream_stream_is_an_error():
    cut = TRANSCRIPT.split("data: [DONE]")[0]
    b, _ = _replay_backend(cut)
    with pytest.raises(BackendError, match="without \\[DONE\\]"):
        asyncio.run(_collect(b, CONTRACT["request"]["messages"]))


def test_upstream_503_is_retryable():
    b, _ = _replay_backend("busy", status=503)
    with pytest.raises(BackendError) as ei:
        asyncio.run(_collect(b, CONTRACT["request"]["messages"]))
    assert ei.value.retryable
