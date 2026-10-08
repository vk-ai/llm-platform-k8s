from pathlib import Path

import pytest

from llm_platform.backends import BackendSpec, MockBackend
from llm_platform.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def specs():
    return [
        BackendSpec("local-small", "small", tier=1, price_in_per_1k=0.00002, price_out_per_1k=0.00004, max_context=4096, kind="mock"),
        BackendSpec("hosted-medium", "medium", tier=2, price_in_per_1k=0.00015, price_out_per_1k=0.0006, max_context=32768, kind="mock"),
        BackendSpec("hosted-large", "large", tier=3, price_in_per_1k=0.0025, price_out_per_1k=0.01, max_context=128000, kind="mock"),
    ]


@pytest.fixture
def mock_backends():
    return {s.name: MockBackend(s) for s in specs()}


@pytest.fixture
def settings():
    return Settings(backends=specs(), semantic_cache=True, semantic_threshold=0.9, prompt_seed=str(ROOT / "config" / "prompts.yaml"))
