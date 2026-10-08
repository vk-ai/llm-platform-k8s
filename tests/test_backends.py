import asyncio

import httpx
import pytest

from llm_platform.backends import BackendError, BackendSpec, OpenAICompatBackend
from llm_platform.mock_server import create_mock_app

SPEC = BackendSpec("vllm", "Qwen/Qwen2.5-0.5B-Instruct", base_url="http://vllm:8000/v1")
M = [{"role": "user", "content": "hi there"}]


def call(handler, spec=SPEC):
    b = OpenAICompatBackend(spec, transport=httpx.MockTransport(handler))
    return asyncio.run(b.complete(M, 16, 0.0))


def test_parses_openai_response_and_usage():
    def h(req: httpx.Request):
        assert req.url.path == "/v1/chat/completions"
        body = req.read()
        assert b'"model":"Qwen/Qwen2.5-0.5B-Instruct"' in body.replace(b" ", b"")
        return httpx.Response(200, json={"model": "qwen", "choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 1}})

    c = call(h)
    assert (c.text, c.prompt_tokens, c.completion_tokens, c.model) == ("hello", 9, 1, "qwen")


def test_missing_usage_is_estimated():
    c = call(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "four words right here"}}]}))
    assert c.prompt_tokens > 0 and c.completion_tokens >= 4


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (503, True), (400, False), (404, False)])
def test_status_mapping(status, retryable):
    with pytest.raises(BackendError) as exc:
        call(lambda r: httpx.Response(status, text="nope"))
    assert exc.value.retryable is retryable and exc.value.status == status


def test_timeout_and_connect_errors_are_retryable():
    def timeout(r):
        raise httpx.ReadTimeout("slow", request=r)

    def refused(r):
        raise httpx.ConnectError("refused", request=r)

    for h in (timeout, refused):
        with pytest.raises(BackendError) as exc:
            call(h)
        assert exc.value.retryable


def test_against_mock_model_server_over_asgi():
    transport = httpx.ASGITransport(app=create_mock_app("small"))
    b = OpenAICompatBackend(BackendSpec("small", "small-model", base_url="http://mock/v1"), transport=transport)
    c = asyncio.run(b.complete(M, 16, 0.0))
    assert c.text.startswith("[small]") and c.prompt_tokens > 0
