"""Generates deploy/helm/llm-platform/dashboards/llm-platform.json (committed).

Run `python scripts/build_dashboard.py` after changing panels. tests/test_deploy.py
checks the file is up to date and that every gateway metric it queries exists.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "deploy/helm/llm-platform/dashboards/llm-platform.json"
DS = {"type": "prometheus", "uid": "${datasource}"}

PANELS = [
    # (title, unit, [(expr, legend)], width)
    ("Gateway requests/s by route and status", "reqps", [("sum by (route, status) (rate(llm_gateway_requests_total[5m]))", "{{route}} {{status}}")], 12),
    ("Spend (USD/hour) by backend", "currencyUSD", [("sum by (backend) (rate(llm_cost_usd_total[5m])) * 3600", "{{backend}}")], 12),
    ("Cache hit ratio (exact + semantic)", "percentunit", [(
        'sum(rate(llm_cache_requests_total{result="hit"}[5m])) / clamp_min(sum(rate(llm_cache_requests_total{cache="exact"}[5m])), 1e-9)', "hit ratio")], 8),
    ("Spend avoided by cache (USD/hour)", "currencyUSD", [("sum by (cache) (rate(llm_cache_saved_cost_usd_total[5m])) * 3600", "{{cache}}")], 8),
    ("Routing decisions/s", "reqps", [("sum by (backend, reason) (rate(llm_backend_requests_total[5m]))", "{{backend}} ({{reason}})")], 8),
    ("Backend latency p50 / p95", "s", [
        ("histogram_quantile(0.5, sum by (le, backend) (rate(llm_backend_latency_seconds_bucket[5m])))", "p50 {{backend}}"),
        ("histogram_quantile(0.95, sum by (le, backend) (rate(llm_backend_latency_seconds_bucket[5m])))", "p95 {{backend}}"),
    ], 12),
    ("Gateway latency p95 by route", "s", [("histogram_quantile(0.95, sum by (le, route) (rate(llm_gateway_request_seconds_bucket[5m])))", "{{route}}")], 12),
    ("Backend errors and failovers/s", "reqps", [
        ("sum by (backend, retryable) (rate(llm_backend_errors_total[5m]))", "error {{backend}} retryable={{retryable}}"),
        ("sum by (from_backend, to_backend) (rate(llm_failovers_total[5m]))", "failover {{from_backend}} -> {{to_backend}}"),
    ], 12),
    ("Tokens/s by backend", "short", [("sum by (backend, kind) (rate(llm_tokens_total[5m]))", "{{backend}} {{kind}}")], 12),
    ("RAG retrieval p95", "s", [("histogram_quantile(0.95, sum by (le, store) (rate(rag_retrieval_seconds_bucket[5m])))", "{{store}}")], 8),
    ("Prompt versions in use (renders/s)", "reqps", [("sum by (name, version) (rate(prompt_renders_total[5m]))", "{{name}}@{{version}}")], 8),
    ("vLLM running / waiting requests", "short", [
        ("sum(vllm:num_requests_running)", "running"),
        ("sum(vllm:num_requests_waiting)", "waiting"),
    ], 8),
]


def build() -> dict:
    panels, x, y = [], 0, 0
    for i, (title, unit, targets, w) in enumerate(PANELS):
        if x + w > 24:
            x, y = 0, y + 8
        panels.append({
            "id": i + 1, "type": "timeseries", "title": title, "datasource": DS,
            "gridPos": {"h": 8, "w": w, "x": x, "y": y},
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
            "targets": [{"refId": chr(65 + j), "datasource": DS, "expr": e, "legendFormat": leg} for j, (e, leg) in enumerate(targets)],
        })
        x += w
    return {
        "uid": "llm-platform", "title": "LLM platform", "tags": ["llm", "gateway", "vllm"],
        "schemaVersion": 39, "version": 1, "editable": True, "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "templating": {"list": [{"name": "datasource", "type": "datasource", "query": "prometheus", "label": "Prometheus"}]},
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT}")
