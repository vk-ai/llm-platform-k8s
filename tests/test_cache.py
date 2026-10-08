from llm_platform.cache import CachedAnswer, ExactCache, SemanticCache, request_key
from llm_platform.embeddings import HashingEmbedder

M = [{"role": "user", "content": "How do I reset my password?"}]


def ans(text="a", cost=0.01):
    return CachedAnswer(text, "b", "m", 10, 5, cost, 0.0)


def test_request_key_is_canonical_and_sensitive():
    assert request_key("s", M, 10, 0) == request_key("s", [dict(reversed(list(M[0].items())))], 10, 0)
    assert request_key("s", M, 10, 0) != request_key("s", M, 11, 0)
    assert request_key("s", M, 10, 0) != request_key("t", M, 10, 0)


def test_exact_lru_and_ttl():
    now = [0.0]
    c = ExactCache(max_entries=2, ttl_s=10, clock=lambda: now[0])
    c.put("a", CachedAnswer("A", "b", "m", 1, 1, 0, 0.0))
    c.put("b", CachedAnswer("B", "b", "m", 1, 1, 0, 0.0))
    assert c.get("a").text == "A"           # a is now most recent
    c.put("c", CachedAnswer("C", "b", "m", 1, 1, 0, 0.0))
    assert c.get("b") is None and len(c) == 2  # b evicted (LRU)
    now[0] = 11
    assert c.get("a") is None               # expired


def test_embedder_properties():
    e = HashingEmbedder(256)
    v = e.embed(["reset my password", "reset my password", "track my parcel", ""])
    assert v.shape == (4, 256)
    assert abs(float(v[0] @ v[0]) - 1) < 1e-5 and float(v[0] @ v[1]) > 0.999
    assert float(v[0] @ v[2]) < 0.5
    assert float(abs(v[3]).sum()) == 0.0


def test_semantic_hit_on_paraphrase_not_on_other_topic():
    s = SemanticCache(HashingEmbedder(384), threshold=0.75)
    s.put("auto", M, ans("reset via settings"))
    hit, sim = s.lookup("auto", [{"role": "user", "content": "how do I reset my password"}])
    assert hit is not None and hit.text == "reset via settings" and sim >= 0.75
    miss, sim2 = s.lookup("auto", [{"role": "user", "content": "where is my parcel?"}])
    assert miss is None and sim2 < 0.75


def test_semantic_scoped_by_system_prompt_and_route():
    s = SemanticCache(HashingEmbedder(384), threshold=0.5)
    s.put("auto", [{"role": "system", "content": "context A"}] + M, ans())
    assert s.lookup("auto", [{"role": "system", "content": "context B"}] + M)[0] is None
    assert s.lookup("pinned:x", [{"role": "system", "content": "context A"}] + M)[0] is None
    assert s.lookup("auto", [{"role": "system", "content": "context A"}] + M)[0] is not None


def test_semantic_ttl_and_capacity():
    now = [0.0]
    s = SemanticCache(HashingEmbedder(64), threshold=0.5, max_entries=2, ttl_s=5, clock=lambda: now[0])
    for i in range(3):
        s.put("auto", [{"role": "user", "content": f"q{i} password"}], CachedAnswer(f"a{i}", "b", "m", 1, 1, 0, 0.0))
    assert len(s) == 2
    now[0] = 10
    assert s.lookup("auto", [{"role": "user", "content": "q2 password"}])[0] is None
