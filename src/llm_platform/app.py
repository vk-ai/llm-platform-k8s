"""FastAPI app: OpenAI-compatible chat endpoint plus RAG, prompt registry and metrics."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .gateway import Gateway
from .prompts import NotFound, PromptError
from .backends import BackendError
from .router import AllBackendsFailed, NoEligibleBackend


class Message(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[Message]
    max_tokens: int = Field(256, ge=1, le=32768)
    temperature: float = Field(0.0, ge=0, le=2)
    stream: bool = False
    stream_options: StreamOptions | None = None


class IngestRequest(BaseModel):
    documents: list[dict]
    max_words: int = 80
    overlap: int = 20


class RagRequest(BaseModel):
    question: str
    k: int = Field(4, ge=1, le=50)
    prompt_label: str | None = "prod"
    prompt_version: str | None = None
    max_tokens: int = 256
    quality: str | None = None


class PublishRequest(BaseModel):
    template: str
    version: str | None = None
    bump: str = "minor"
    metadata: dict = {}


def create_app(settings: Settings | None = None, gateway: Gateway | None = None) -> FastAPI:
    holder: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        holder["gw"] = gateway or Gateway(settings or load_settings())
        yield
        await holder["gw"].aclose()

    app = FastAPI(title="llm-platform gateway", version="0.1.0", lifespan=lifespan)

    def gw() -> Gateway:
        return holder["gw"]

    def timed(route: str):
        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()
                self.status = "ok"
                return self

            def __exit__(self, exc_type, exc, tb):
                gw().metrics.request_latency.labels(route=route).observe(time.perf_counter() - self.t0)
                status = self.status if exc_type is None or issubclass(exc_type, HTTPException) else "error"
                gw().metrics.requests.labels(route=route, status=status).inc()
                return False

        return _T()

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz():
        return {"ok": True, "backends": list(gw().backends), "vector_store": gw().store.name, "chunks": gw().store.count()}

    @app.get("/metrics")
    async def metrics():
        return Response(gw().metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        data = [{"id": "auto", "object": "model", "owned_by": "router"}]
        for name, b in gw().backends.items():
            s = b.spec
            data.append({"id": name, "object": "model", "owned_by": "llm-platform", "tier": s.tier, "model": s.model,
                         "price_in_per_1k": s.price_in_per_1k, "price_out_per_1k": s.price_out_per_1k})
        return {"object": "list", "data": data}

    def _sse(obj) -> bytes:
        return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()

    async def stream_chat(req: ChatRequest, quality, max_cost, cache_mode):
        """OpenAI-style SSE. Routing errors before the first token are normal HTTP errors.

        With stream_options.include_usage the last chunk (choices=[]) carries usage
        and an ``x_llm`` block with backend, cache and cost_usd. The gateway always
        asks upstream for include_usage, so cost is based on reported tokens when
        the backend provides them.
        """
        t0 = time.perf_counter()
        agen = gw().chat_stream([m.model_dump() for m in req.messages], req.model, req.max_tokens, req.temperature,
                                quality, max_cost, cache_mode)

        def finish(status: str) -> None:
            gw().metrics.request_latency.labels(route="chat_stream").observe(time.perf_counter() - t0)
            gw().metrics.requests.labels(route="chat_stream", status=status).inc()

        try:
            first = await agen.__anext__()
        except NoEligibleBackend as exc:
            finish("no_backend")
            raise HTTPException(422, {"error": str(exc), "rejected": exc.rejected})
        except AllBackendsFailed as exc:
            finish("upstream_error")
            raise HTTPException(502, {"error": "all backends failed", "details": exc.errors})
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        include_usage = bool(req.stream_options and req.stream_options.include_usage)

        def chunk(delta: dict, finish_reason=None, model="auto") -> dict:
            c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if include_usage:
                c["usage"] = None
            return c

        async def body():
            ev, val = first
            role_sent = False
            try:
                while True:
                    if ev == "delta":
                        d = {"content": val}
                        if not role_sent:
                            d["role"], role_sent = "assistant", True
                        yield _sse(chunk(d))
                    else:
                        r = val
                        break
                    ev, val = await agen.__anext__()
            except BackendError as exc:
                finish("stream_error")
                yield _sse({"error": {"message": str(exc), "type": "upstream_stream_error"}})
                return
            yield _sse(chunk({}, "stop", r.model))
            if include_usage:
                yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": r.model, "choices": [],
                            "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                                      "total_tokens": r.prompt_tokens + r.completion_tokens},
                            "x_llm": {"backend": r.backend, "cache": r.cache, "cost_usd": r.cost_usd, "tried": r.tried,
                                      "reason": r.reason, "usage_source": r.usage_source}})
            yield b"data: [DONE]\n\n"
            finish(f"cache_{r.cache}" if r.cache in ("exact", "semantic") else "ok")

        return StreamingResponse(body(), media_type="text/event-stream", headers={"cache-control": "no-cache"})

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, response: Response,
                   x_llm_quality: str | None = Header(None), x_llm_max_cost_usd: float | None = Header(None),
                   x_llm_cache: str = Header("auto")):
        if req.stream:
            return await stream_chat(req, x_llm_quality, x_llm_max_cost_usd, x_llm_cache)
        with timed("chat") as t:
            try:
                r = await gw().chat([m.model_dump() for m in req.messages], req.model, req.max_tokens, req.temperature,
                                    x_llm_quality, x_llm_max_cost_usd, x_llm_cache)
            except NoEligibleBackend as exc:
                t.status = "no_backend"
                raise HTTPException(422, {"error": str(exc), "rejected": exc.rejected})
            except AllBackendsFailed as exc:
                t.status = "upstream_error"
                raise HTTPException(502, {"error": "all backends failed", "details": exc.errors})
            t.status = f"cache_{r.cache}" if r.cache in ("exact", "semantic") else "ok"
        response.headers["x-llm-backend"] = r.backend or ""
        response.headers["x-llm-cache"] = r.cache
        response.headers["x-llm-cost-usd"] = f"{r.cost_usd:.8f}"
        response.headers["x-llm-route-reason"] = r.reason
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": r.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": r.text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "total_tokens": r.prompt_tokens + r.completion_tokens},
            "x_llm": {"backend": r.backend, "cache": r.cache, "cost_usd": r.cost_usd, "tried": r.tried, "reason": r.reason, "similarity": r.similarity},
        }

    @app.post("/v1/rag/ingest")
    async def ingest(req: IngestRequest):
        with timed("rag_ingest"):
            n = gw().rag.ingest(req.documents, req.max_words, req.overlap)
        return {"chunks": n, "total": gw().store.count()}

    @app.post("/v1/rag/query")
    async def rag_query(req: RagRequest):
        with timed("rag_query") as t:
            try:
                return await gw().rag_query(req.question, req.k, req.prompt_label, req.prompt_version, max_tokens=req.max_tokens, quality=req.quality)
            except NotFound as exc:
                t.status = "not_found"
                raise HTTPException(404, str(exc))
            except (NoEligibleBackend, AllBackendsFailed) as exc:
                t.status = "upstream_error"
                raise HTTPException(502, str(exc))

    @app.get("/v1/prompts")
    async def list_prompts():
        reg = gw().registry
        return {n: {"versions": [v.version for v in reg.versions(n)], "labels": reg.labels(n)} for n in reg.names()}

    @app.get("/v1/prompts/{name}")
    async def get_prompt(name: str, version: str | None = None, label: str | None = None):
        try:
            return gw().registry.get(name, version=version, label=label).to_dict()
        except NotFound as exc:
            raise HTTPException(404, str(exc))

    @app.post("/v1/prompts/{name}/versions", status_code=201)
    async def publish(name: str, req: PublishRequest):
        try:
            return gw().registry.publish(name, req.template, req.version, req.bump, req.metadata).to_dict()
        except PromptError as exc:
            raise HTTPException(409, str(exc))

    @app.put("/v1/prompts/{name}/labels/{label}")
    async def set_label(name: str, label: str, version: str, actor: str = "api"):
        try:
            gw().registry.set_label(name, label, version, actor)
        except NotFound as exc:
            raise HTTPException(404, str(exc))
        return gw().registry.labels(name)

    @app.post("/v1/prompts/{name}/labels/{label}/rollback")
    async def rollback(name: str, label: str, actor: str = "api"):
        try:
            return {"label": label, "version": gw().registry.rollback(name, label, actor)}
        except NotFound as exc:
            raise HTTPException(404, str(exc))
        except PromptError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/v1/prompts/{name}/diff", response_class=PlainTextResponse)
    async def diff(name: str, a: str, b: str):
        try:
            return gw().registry.diff(name, a, b)
        except NotFound as exc:
            raise HTTPException(404, str(exc))

    return app
