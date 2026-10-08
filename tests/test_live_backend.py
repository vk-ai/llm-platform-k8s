"""Opt-in smoke test against a real OpenAI-compatible server (e.g. `vllm serve`).

Skipped unless LLMPLAT_REAL_BASE_URL is set; CI never sets it.
"""
import asyncio
import os

import pytest

from llm_platform.backends import BackendSpec, OpenAICompatBackend

BASE = os.environ.get("LLMPLAT_REAL_BASE_URL")
pytestmark = [pytest.mark.live, pytest.mark.skipif(not BASE, reason="LLMPLAT_REAL_BASE_URL not set")]


def test_real_backend_round_trip():
    spec = BackendSpec("real-local", os.environ.get("LLMPLAT_REAL_MODEL", "qwen2.5-0.5b"), base_url=BASE, timeout_s=300)
    c = asyncio.run(OpenAICompatBackend(spec).complete([{"role": "user", "content": "Reply with the single word: pong"}], 8, 0.0))
    assert c.text.strip() and c.prompt_tokens > 0 and c.completion_tokens > 0
