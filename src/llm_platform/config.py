"""Platform config: YAML file (``LLMPLAT_CONFIG``) with env overrides for secrets/URLs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .backends import BackendSpec


@dataclass
class Settings:
    backends: list[BackendSpec]
    exact_cache: bool = True
    exact_cache_ttl_s: float = 3600
    cache_nondeterministic: bool = False
    semantic_cache: bool = False
    semantic_threshold: float = 0.92
    vector_store: str = "memory"              # memory | qdrant | pgvector
    vector_store_options: dict = field(default_factory=dict)
    embedding_dim: int = 384
    embedder: dict = field(default_factory=dict)  # {} = hashing; {"base_url":..., "model":..., "dim":...}
    registry_path: str = ":memory:"
    prompt_seed: str | None = None
    tier_aliases: dict = field(default_factory=lambda: {"small": 1, "standard": 1, "medium": 2, "high": 3, "large": 3})


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_settings(path: str | Path | None = None) -> Settings:
    path = path or os.environ.get("LLMPLAT_CONFIG", "config/platform.yaml")
    raw = _expand(yaml.safe_load(Path(path).read_text()))
    backends = [BackendSpec(**b) for b in raw.pop("backends")]
    cache = raw.pop("cache", {}) or {}
    rag = raw.pop("rag", {}) or {}
    prompts = raw.pop("prompts", {}) or {}
    return Settings(
        backends=backends,
        exact_cache=cache.get("exact", True),
        exact_cache_ttl_s=cache.get("ttl_s", 3600),
        cache_nondeterministic=cache.get("cache_nondeterministic", False),
        semantic_cache=cache.get("semantic", False),
        semantic_threshold=cache.get("semantic_threshold", 0.92),
        vector_store=os.environ.get("LLMPLAT_VECTOR_STORE", rag.get("store", "memory")),
        vector_store_options=rag.get("options", {}),
        embedding_dim=rag.get("dim", 384),
        embedder=rag.get("embedder", {}) or {},
        registry_path=os.environ.get("LLMPLAT_REGISTRY", prompts.get("path", ":memory:")),
        prompt_seed=prompts.get("seed"),
    )
