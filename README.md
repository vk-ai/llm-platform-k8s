# llm-platform-k8s

A small self-hosted LLM platform you can read in an afternoon: an
OpenAI-compatible FastAPI gateway that routes each request to the cheapest
backend that meets its quality tier, context length and budget; exact and
semantic prompt caching; a versioned prompt registry; RAG on Qdrant or
pgvector; Prometheus metrics with a provisioned Grafana dashboard; and a Helm
chart, Terraform and a Makefile to run vLLM (CPU) or KServe on a local kind
cluster.

> **Learning project.** I built this to practise LLM platform engineering end to
> end. It is not used in production anywhere. The gateway, registry, caches, RAG
> and metrics are real code with tests. The Kubernetes deploy is **validated but
> not run**: the chart passes `helm lint` and `kubeconform`, and Terraform passes
> `validate`, but I have not applied either to a cluster because my build machine
> has no Docker. See [What is real vs stubbed](#what-is-real-vs-stubbed).

## What it does

```
client ──OpenAI API──► gateway (FastAPI)
                         │ 1. render prompt from registry (name@version or label)   ◄── SQLite prompt registry
                         │ 2. RAG: embed → top-k from Qdrant / pgvector / in-memory  ◄── /v1/rag/ingest
                         │ 3. exact cache → semantic cache (scoped by system prompt)
                         │ 4. cost router: filter by tier / context / budget / circuit breaker,
                         │    try cheapest first, fail over in cost order on 429/5xx/timeout
                         ▼
          local-small (vLLM CPU / KServe)   hosted-medium   hosted-large      (any OpenAI-compatible URL)
                         │
                     /metrics ──► Prometheus (ServiceMonitor) ──► Grafana dashboard (ConfigMap sidecar)
```

| Capability | How |
|---|---|
| Cost-based routing | Each backend has a tier, context window and per-1k token prices. `CostRouter` estimates request cost from prompt tokens and `max_tokens`, drops backends that are below the requested tier (`x-llm-quality: standard\|medium\|high`), too small for the prompt, over the budget (`x-llm-max-cost-usd`) or have an open circuit, then tries the rest cheapest first. Retryable errors (429, 5xx, timeouts) fail over to the next one; 4xx errors don't. A per-backend circuit breaker opens after 3 consecutive failures and half-opens after a cooldown. A 422 response lists why each backend was rejected. |
| Prompt caching | Exact cache: LRU + TTL keyed on the full normalised request. Semantic cache: cosine similarity over embeddings of the last user message, scoped by system prompt and by the pinned model or quality tier, so a hit can't cross prompt versions (threshold 0.9 by default). Only `temperature: 0` requests are cached unless configured otherwise; `x-llm-cache: off` skips both. Hits record the cost they saved. vLLM additionally runs with `--enable-prefix-caching` for KV-cache reuse. |
| Prompt registry | SQLite. Prompts are versioned with semver (`--bump major\|minor\|patch` or an explicit version), content-addressed by sha256 (re-publishing the same text is a no-op, re-using a version for different text is an error), with movable labels (`prod`, `staging`) and a label history so `rollback` goes back to the previous version. `diff` returns a unified diff. Available over HTTP and the `llm-platform prompts` CLI. |
| RAG | Paragraph chunking with overlap, then `InMemoryStore` (tests), `QdrantStore` (qdrant-client; `:memory:` locally, a server in the chart) or `PgVectorStore` (psycopg + `vector(384)` with an HNSW cosine index). Answers use the registry prompt `rag-answer` and return citations with scores. |
| Metrics | `prometheus-client`: requests by route and status, backend requests by route reason (cheapest / failover / cache), failovers, tokens, cost in USD, cache hit/miss and saved cost, latency histograms, RAG retrieval time, prompt renders by version. `scripts/build_dashboard.py` generates the 12-panel Grafana dashboard JSON, and a test checks that every series it queries exists (ours from the code; vLLM's from a list scraped from a real vLLM 0.31.0 `/metrics`). |
| Kubernetes | Helm chart: gateway Deployment + Service + ConfigMap, vLLM CPU Deployment (`vllm/vllm-openai-cpu:v0.31.0`, Qwen2.5-0.5B-Instruct) or a KServe `InferenceService`, Qdrant or pgvector StatefulSet, `ServiceMonitor` and a dashboard ConfigMap for kube-prometheus-stack. Terraform (`tehcyx/kind`, `hashicorp/helm` 3.x) creates the kind cluster and installs kube-prometheus-stack 92.1.1 and the chart. |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # add ,pgvector for Postgres
pytest -q                          # offline: mock backends, in-memory Qdrant
python demo/run_demo.py            # the run captured below
llm-platform serve --config config/platform.yaml --port 8080

curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -H 'x-llm-quality: medium' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

Point a backend at a real server by setting `kind: openai` and `base_url` in
`config/platform.yaml` (any OpenAI-compatible API: vLLM, llama.cpp, Ollama, a
hosted provider; `api_key` supports `${ENV_VAR}`).

Kubernetes (needs Docker, kind, helm, terraform; **I have not run this**):

```bash
make lint          # helm lint + kubeconform (incl. CRD schemas) + terraform fmt/validate  ← run in CI
make deploy        # terraform: kind cluster → load gateway image → kube-prometheus-stack → chart
make port-forward smoke
make cluster-down
```

## Demo run

`python demo/run_demo.py` drives the real FastAPI app through `TestClient`
with three mock backends, Qdrant in local mode and (when `PGVECTOR_DSN` is set)
Postgres 17 + pgvector 0.8.0. Output from 2026-10-08:

```
== 1. prompt registry: publish, label, diff, rollback
   published rag-answer@1.1.0 sha256=79e46d916a4c variables=['context']
   labels after promote: {'prod': '1.1.0'}
   --- rag-answer@1.0.0
   +++ rag-answer@1.1.0
   @@ -1,3 +1,3 @@
   -You are a support assistant. Answer using only the context below.
   +You are a support assistant. Answer in one sentence and cite sources as [n].
    Context:
    {context}

== 2. RAG on Qdrant (qdrant-client local mode)
   ingested chunks: {'chunks': 5, 'total': 5}
   prompt=rag-answer@1.1.0 backend=local-small citations=[('refunds#0', 0.3644), ('refunds#1', 0.1207)]
   rollback prod -> {'label': 'prod', 'version': '1.0.0'}
   same question now uses rag-answer@1.0.0

== 3. cost-based routing
   standard quality           backend=local-small   cache=miss     cost=$0.000001 reason=cheapest
   x-llm-quality: medium      backend=hosted-medium cache=miss     cost=$0.000011 reason=cheapest
   x-llm-quality: high        backend=hosted-large  cache=miss     cost=$0.000192 reason=cheapest
   high + budget $0.0001      HTTP 422 {'error': 'no backend satisfies tier/context/budget/health', 'rejected': {'local-small': 'tier 1 < 3', 'hosted-medium': 'tier 2 < 3', 'hosted-large': 'est $0.000678 > budget $0.000100'}}
   prompt > 4k context        backend=hosted-medium cache=miss     cost=$0.001010 reason=cheapest

== 4. prompt caching
   first ask                  backend=local-small   cache=miss     cost=$0.000001 reason=cheapest
   identical (exact)          backend=local-small   cache=exact    cost=$0.000000 reason=cache
   reworded (semantic)        backend=local-small   cache=semantic cost=$0.000000 reason=cache sim=0.927
   related but different      backend=local-small   cache=miss     cost=$0.000001 reason=cheapest

== 5. failover and circuit breaker
   local-small returns 503    backend=hosted-medium cache=miss     cost=$0.000007 reason=failover
   after 3 failures (open)    backend=hosted-medium cache=miss     cost=$0.000007 reason=cheapest | breaker: circuit open

== 6. Prometheus metrics (excerpt of /metrics)
   llm_backend_requests_total{backend="local-small",reason="cheapest"} 5.0
   llm_backend_requests_total{backend="hosted-medium",reason="cheapest"} 4.0
   llm_backend_requests_total{backend="hosted-large",reason="cheapest"} 1.0
   llm_backend_requests_total{backend="hosted-medium",reason="failover"} 3.0
   llm_failovers_total{from_backend="local-small",to_backend="hosted-medium"} 3.0
   llm_cost_usd_total{backend="local-small"} 8.660000000000002e-06
   llm_cost_usd_total{backend="hosted-medium"} 0.0010561499999999998
   llm_cost_usd_total{backend="hosted-large"} 0.0001925
   llm_cache_saved_cost_usd_total{cache="exact"} 7.000000000000001e-07
   llm_cache_saved_cost_usd_total{cache="semantic"} 7.000000000000001e-07
   llm_cache_requests_total{cache="exact",result="miss"} 15.0
   llm_cache_requests_total{cache="semantic",result="miss"} 14.0
   llm_cache_requests_total{cache="exact",result="hit"} 1.0
   llm_cache_requests_total{cache="semantic",result="hit"} 1.0
   prompt_renders_total{name="rag-answer",version="1.1.0"} 1.0
   prompt_renders_total{name="rag-answer",version="1.0.0"} 1.0

== 7. same RAG flow on Postgres + pgvector
   ingested chunks: {'chunks': 5, 'total': 5}
   citations=[('refunds#0', 0.3644), ('refunds#1', 0.1207)]
```

Notes on the run:
- In section 4 the semantic cache matches "password reset - how do I do it?"
  to the earlier "How do I reset my password?" (similarity 0.927), but not
  "How do I change my password?". The default embedder is a hashing embedder
  (lexical overlap with stemming), so it only catches near-paraphrases. Use
  `OpenAIEmbedder` against a real embedding endpoint for semantic matches.
- In section 5, after three 503s the breaker opens and `local-small` is no
  longer tried at all (`reason=cheapest`, not `failover`).

**Real model check.** I also ran the gateway against a real vLLM server, using
the official vLLM 0.31.0 CPU wheel (torch 2.13 CPU) on the build machine, not
inside kind:

```bash
VLLM_CPU_KVCACHE_SPACE=2 vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen2.5-0.5b \
  --max-model-len 2048 --enable-prefix-caching --port 8011
LLMPLAT_REAL_BASE_URL=http://localhost:8011/v1 LLMPLAT_REAL_MODEL=qwen2.5-0.5b python demo/run_demo.py
```

```
== 5b. real OpenAI-compatible backend at http://localhost:8011/v1
    200 real-local 'A refund is the process of returning money or goods that have been purchased, usually after an agreement has been reache' {'prompt_tokens': 39, 'completion_tokens': 27, 'total_tokens': 66}
```

The same run checked `tests/test_live_backend.py` (skipped unless
`LLMPLAT_REAL_BASE_URL` is set) and saved vLLM's metric names to
`tests/fixtures/vllm-0.31.0-cpu-metric-names.txt` for the dashboard test.

## Tests

```
pytest -q                                   55 passed, 2 skipped   (pgvector, live)
PGVECTOR_DSN=postgresql://... pytest -q     56 passed, 1 skipped
+ LLMPLAT_REAL_BASE_URL=...                 57 passed              (92% line coverage)
```

Covered: the prompt registry (semver, idempotent publish, labels, rollback,
diff), both caches (TTL, LRU, scoping, threshold), the router (eligibility,
budget, failover order, non-retryable errors, breaker open/half-open), the
OpenAI-compatible client (via `httpx.MockTransport` and the bundled mock server),
all three vector stores, the HTTP API end to end, and the deploy artifacts:
the chart renders with the expected objects in both vLLM and KServe modes, the
rendered gateway ConfigMap loads as a valid config, and the dashboard only
queries metrics that exist.

CI (`.github/workflows/ci.yml`) runs, all offline except for image and chart downloads:
- `test`: pytest + the demo on Python 3.11 and 3.12;
- `pgvector`: the pgvector tests and demo section against a `pgvector/pgvector:pg17` service container;
- `deploy-lint`: `helm lint`, `kubeconform -strict` with CRD schemas from the datreeio CRDs-catalog, and `terraform fmt -check` + `validate`.

## What is real vs stubbed

**Real (runs and is tested here):**
- The gateway, cost router, failover, circuit breaker, both caches, prompt registry, RAG service and metrics.
- Qdrant through `qdrant-client` in local mode, and pgvector on a real Postgres 17 + pgvector 0.8.0 (locally and in CI).
- One real completion through the gateway from Qwen2.5-0.5B-Instruct on the vLLM 0.31.0 CPU wheel, run outside Kubernetes.
- Helm chart rendering, `helm lint`, `kubeconform`, `terraform validate`.

**Stubbed, simulated or not verified:**
- **The cluster deploy is validated, not run.** I have not run `make deploy`, the Dockerfile or `docker-compose.yml`, because the build machine has no Docker. KServe and kube-prometheus-stack are only checked against their schemas.
- **The backends in the demo and tests are mocks.** They return canned text. "hosted-medium" and "hosted-large" are placeholders with made-up prices, not real providers, and the `local-small` price is a notional amortised cost.
- **The default embedder is lexical, not semantic** (feature hashing + stopwords + stemming). It is deterministic and dependency-free, which is why tests use it, but the RAG scores and semantic-cache hits above reflect word overlap. `OpenAIEmbedder` exists but is only unit-tested.
- **Token counts are estimated** (about 4 characters per token) for routing decisions; billing uses the backend's `usage` field when present.
- **No streaming** (`stream: true` returns 400), no auth, no rate limiting, no multi-replica cache (each gateway pod has its own in-process caches).
- **The Grafana dashboard was never opened in Grafana.** It is generated JSON whose queries are checked against the metric names.

## Layout

```
src/llm_platform/  app.py · gateway.py · router.py · backends.py · cache.py · embeddings.py
                   prompts.py · rag/{service,stores}.py · metrics.py · config.py · mock_server.py · cli.py
config/            platform.yaml (local) · compose.yaml · prompts.yaml (seed prompts)
deploy/helm/llm-platform/   chart + values-kind.yaml + dashboards/llm-platform.json
deploy/terraform/  kind cluster + helm releases
deploy/grafana, deploy/prometheus.yml   provisioning for docker-compose
scripts/build_dashboard.py
tests/             57 tests (2 opt-in)
demo/run_demo.py, demo/docs/
Dockerfile, docker-compose.yml, Makefile
```

## License

MIT
