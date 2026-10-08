"""Consistency checks between the code and the deploy artefacts (dashboard, Helm chart, configs)."""
import importlib.util
import json
import re
import shutil
import subprocess

import pytest
import yaml

from conftest import ROOT
from llm_platform.backends import Completion
from llm_platform.config import load_settings
from llm_platform.metrics import Metrics

CHART = ROOT / "deploy/helm/llm-platform"
DASHBOARD = CHART / "dashboards/llm-platform.json"


def exported_metric_names() -> set[str]:
    m = Metrics()
    m.observe_completion("b", Completion("x", 1, 1, "m", 0.1), 0.001, "cheapest")
    m.requests.labels(route="chat", status="ok").inc()
    m.backend_errors.labels(backend="b", retryable="true").inc()
    m.failovers.labels(from_backend="a", to_backend="b").inc()
    m.cache.labels(cache="exact", result="hit").inc()
    m.saved_cost.labels(cache="exact").inc(0.1)
    m.request_latency.labels(route="chat").observe(0.1)
    m.retrieval_latency.labels(store="memory").observe(0.01)
    m.prompt_renders.labels(name="p", version="1.0.0").inc()
    return {line.split("{")[0].split(" ")[0] for line in m.render().decode().splitlines() if line and not line.startswith("#")}


def test_dashboard_is_generated_file():
    spec = importlib.util.spec_from_file_location("build_dashboard", ROOT / "scripts/build_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert json.loads(DASHBOARD.read_text()) == mod.build(), "run `make dashboard`"


def test_dashboard_queries_only_existing_metrics():
    names = exported_metric_names()
    dash = json.loads(DASHBOARD.read_text())
    used = set()
    for panel in dash["panels"]:
        for t in panel["targets"]:
            used |= set(re.findall(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\s*[\[{]", t["expr"]))
    ours = {u for u in used if not u.startswith("vllm:")}
    assert ours and ours <= names, f"unknown metrics: {ours - names}"
    assert all(p["datasource"]["uid"] == "${datasource}" for p in dash["panels"])
    # vLLM series must exist in the metric names scraped from a real vLLM 0.31.0 (CPU) /metrics endpoint.
    vllm_names = set((ROOT / "tests/fixtures/vllm-0.31.0-cpu-metric-names.txt").read_text().split())
    theirs = {m for p in dash["panels"] for t in p["targets"] for m in re.findall(r"vllm:[a-z_]+", t["expr"])}
    assert theirs and theirs <= vllm_names, f"unknown vLLM metrics: {theirs - vllm_names}"


def test_local_and_compose_configs_load(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert len(load_settings("config/platform.yaml").backends) == 3
    s = load_settings("config/compose.yaml")
    assert s.vector_store == "qdrant" and s.vector_store_options["url"] == "http://qdrant:6333"


helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")


def render(*args: str) -> list[dict]:
    out = subprocess.run(["helm", "template", "llm-platform", str(CHART), *args], check=True, capture_output=True, text=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


@helm
def test_chart_renders_expected_objects_and_valid_gateway_config(tmp_path, monkeypatch):
    docs = render("-f", str(CHART / "values-kind.yaml"))
    kinds = sorted(f"{d['kind']}/{d['metadata']['name']}" for d in docs)
    assert "Deployment/llm-platform-vllm" in kinds and "StatefulSet/llm-platform-qdrant" in kinds
    assert "ServiceMonitor/llm-platform" in kinds and "ConfigMap/llm-platform-dashboard" in kinds
    cm = next(d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "llm-platform-gateway")
    (tmp_path / "platform.yaml").write_text(cm["data"]["platform.yaml"])
    s = load_settings(tmp_path / "platform.yaml")
    assert s.backends[0].name == "local-vllm" and s.backends[0].base_url == "http://llm-platform-vllm:8000/v1"
    assert s.vector_store_options["url"] == "http://llm-platform-qdrant:6333" and s.semantic_cache
    vllm = next(d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == "llm-platform-vllm")
    assert "--enable-prefix-caching" in vllm["spec"]["template"]["spec"]["containers"][0]["args"]
    dash = next(d for d in docs if d["metadata"]["name"] == "llm-platform-dashboard")
    assert dash["metadata"]["labels"]["grafana_dashboard"] == "1"
    assert json.loads(dash["data"]["llm-platform.json"]) == json.loads(DASHBOARD.read_text())
    svc = next(d for d in docs if d["kind"] == "Service" and d["metadata"]["name"] == "llm-platform-gateway")
    assert svc["spec"]["type"] == "NodePort" and svc["spec"]["ports"][0]["nodePort"] == 30080


@helm
def test_chart_kserve_pgvector_and_hosted_backend(tmp_path, monkeypatch):
    docs = render("--set", "vllm.enabled=false,kserve.enabled=true,qdrant.enabled=false,pgvector.enabled=true,vectorStore.kind=pgvector",
                  "--set", "gateway.existingSecret=llm-keys,gateway.extraBackends[0].name=hosted-large,gateway.extraBackends[0].model=big,"
                           "gateway.extraBackends[0].baseUrl=https://api.example.com/v1,gateway.extraBackends[0].tier=3,"
                           "gateway.extraBackends[0].priceInPer1k=0.002,gateway.extraBackends[0].priceOutPer1k=0.008,gateway.extraBackends[0].maxContext=128000")
    kinds = {d["kind"] for d in docs}
    assert "InferenceService" in kinds and "StatefulSet" in kinds and not any(d["metadata"]["name"].endswith("-vllm") for d in docs)
    cm = next(d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "llm-platform-gateway")
    (tmp_path / "p.yaml").write_text(cm["data"]["platform.yaml"])
    monkeypatch.setenv("PGPASSWORD", "pw")
    monkeypatch.setenv("BACKEND_KEY_HOSTED_LARGE", "sk-test")
    s = load_settings(tmp_path / "p.yaml")
    assert [b.name for b in s.backends] == ["local-kserve", "hosted-large"]
    assert s.backends[1].api_key == "sk-test" and s.vector_store_options["dsn"].startswith("postgresql://llmplat:pw@")
    gw = next(d for d in docs if d["kind"] == "Deployment")
    env = {e["name"] for e in gw["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert {"PGPASSWORD", "BACKEND_KEY_HOSTED_LARGE"} <= env


@helm
def test_chart_requires_secret_for_hosted_backends():
    with pytest.raises(subprocess.CalledProcessError) as exc:
        render("--set", "gateway.extraBackends[0].name=x,gateway.extraBackends[0].model=m,gateway.extraBackends[0].baseUrl=u,"
                        "gateway.extraBackends[0].tier=3,gateway.extraBackends[0].priceInPer1k=1,gateway.extraBackends[0].priceOutPer1k=1,gateway.extraBackends[0].maxContext=1")
    assert "gateway.existingSecret is required" in exc.value.stderr
