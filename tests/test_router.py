import asyncio

import pytest

from llm_platform.backends import BackendSpec, MockBackend
from llm_platform.metrics import Metrics
from llm_platform.router import AllBackendsFailed, CostRouter, NoEligibleBackend
from conftest import specs

M = [{"role": "user", "content": "Summarise our refund policy in one line."}]


def run(coro):
    return asyncio.run(coro)


def router(backends, clock=None, **kw):
    return CostRouter(backends, Metrics(), clock=clock or (lambda: 0.0), **kw)


def sample(metrics, name, labels):
    return metrics.registry.get_sample_value(name, labels) or 0.0


def test_cheapest_eligible_backend(mock_backends):
    r = run(router(mock_backends).complete(M, max_tokens=64))
    assert r.backend == "local-small" and r.reason == "cheapest" and r.cost_usd > 0


def test_quality_tier_raises_floor(mock_backends):
    rt = router(mock_backends)
    assert run(rt.complete(M, 64, min_tier=2)).backend == "hosted-medium"
    assert run(rt.complete(M, 64, min_tier=3)).backend == "hosted-large"


def test_context_window_excludes_small_model(mock_backends):
    long = [{"role": "user", "content": "word " * 5000}]
    order, rejected = router(mock_backends).plan(long, 256)
    assert [n for _, n in order] == ["hosted-medium", "hosted-large"]
    assert "context" in rejected["local-small"]


def test_budget_excludes_expensive_and_can_reject_all(mock_backends):
    rt = router(mock_backends)
    order, rejected = rt.plan(M, 1000, min_tier=3, max_cost_usd=0.001)
    assert order == [] and "budget" in rejected["hosted-large"]
    with pytest.raises(NoEligibleBackend) as exc:
        run(rt.complete(M, 1000, min_tier=3, max_cost_usd=0.001))
    assert set(exc.value.rejected) == {"local-small", "hosted-medium", "hosted-large"}


def test_failover_in_cost_order_with_metrics(mock_backends):
    rt = router(mock_backends)
    mock_backends["local-small"].fail_next = 1
    r = run(rt.complete(M, 64))
    assert r.backend == "hosted-medium" and r.reason == "failover" and r.tried == ["local-small", "hosted-medium"]
    assert sample(rt.metrics, "llm_failovers_total", {"from_backend": "local-small", "to_backend": "hosted-medium"}) == 1
    assert sample(rt.metrics, "llm_backend_errors_total", {"backend": "local-small", "retryable": "true"}) == 1


def test_non_retryable_error_does_not_fail_over():
    from llm_platform.backends import BackendError

    class Bad(MockBackend):
        async def complete(self, *a, **k):
            raise BackendError("400", retryable=False, status=400)

    s = specs()
    backends = {s[0].name: Bad(s[0]), s[1].name: MockBackend(s[1])}
    with pytest.raises(AllBackendsFailed):
        run(router(backends).complete(M, 64))
    assert backends["hosted-medium"].calls == 0


def test_circuit_breaker_opens_then_half_opens(mock_backends):
    now = [0.0]
    rt = router(mock_backends, clock=lambda: now[0], breaker_threshold=2, breaker_reset_s=30)
    mock_backends["local-small"].always_fail = True
    run(rt.complete(M, 64))
    run(rt.complete(M, 64))
    calls = mock_backends["local-small"].calls
    r = run(rt.complete(M, 64))  # circuit open: small not even tried
    assert mock_backends["local-small"].calls == calls and r.tried == ["hosted-medium"]
    assert "circuit open" in rt.plan(M, 64)[1]["local-small"]
    now[0] = 31
    mock_backends["local-small"].always_fail = False
    r = run(rt.complete(M, 64))  # half-open probe succeeds and closes it
    assert r.backend == "local-small" and rt.breakers["local-small"].opened_at is None


def test_all_fail(mock_backends):
    for b in mock_backends.values():
        b.always_fail = True
    with pytest.raises(AllBackendsFailed) as exc:
        run(router(mock_backends).complete(M, 64))
    assert len(exc.value.errors) == 3


def test_pinned_backend(mock_backends):
    rt = router(mock_backends)
    assert run(rt.complete(M, 64, pinned="hosted-large")).reason == "pinned"
    with pytest.raises(NoEligibleBackend):
        run(rt.complete(M, 64, pinned="nope"))


def test_cost_estimate():
    s = BackendSpec("x", "m", price_in_per_1k=0.001, price_out_per_1k=0.002)
    assert s.estimate_cost(1000, 500) == pytest.approx(0.002)
