"""A tiny OpenAI-compatible server wrapping MockBackend, for docker-compose/kind when no real model runs."""
from __future__ import annotations

import os

from fastapi import FastAPI

from .backends import BackendSpec, MockBackend


def create_mock_app(name: str | None = None) -> FastAPI:
    name = name or os.environ.get("MOCK_NAME", "mock")
    backend = MockBackend(BackendSpec(name=name, model=f"{name}-model", kind="mock"))
    app = FastAPI(title=f"mock model server ({name})")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        c = await backend.complete(body["messages"], int(body.get("max_tokens", 256)), float(body.get("temperature", 0)))
        return {"object": "chat.completion", "model": c.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": c.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens, "total_tokens": c.prompt_tokens + c.completion_tokens}}

    return app
