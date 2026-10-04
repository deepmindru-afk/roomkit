"""``transport=``: every request of an OpenAI-family provider goes through it.

The seam a host puts an outbound policy in (one that judges the address
actually dialled), and a test a ``MockTransport``. Honoured by
``OpenAIAIProvider`` and every provider built on it, those that build their
own SDK client included (Azure, OpenRouter) and ``create_vllm_provider``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.deepseek import DeepSeekAIProvider, DeepSeekConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.vllm import create_vllm_provider
from roomkit.providers.vllm.config import VLLMConfig

_COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

Factory = Callable[[httpx.AsyncBaseTransport], Any]

_PROVIDERS: dict[str, Factory] = {
    "openai": lambda t: OpenAIAIProvider(
        OpenAIConfig(api_key="k", model="gpt-x", base_url="https://llm.example/v1"), transport=t
    ),
    "azure": lambda t: AzureAIProvider(
        AzureAIConfig(api_key="k", azure_endpoint="https://az.example", model="dep"),
        transport=t,
    ),
    "openrouter": lambda t: OpenRouterAIProvider(
        OpenRouterConfig(api_key="k", model="vendor/model"), transport=t
    ),
    "vllm": lambda t: create_vllm_provider(
        VLLMConfig(model="m", base_url="https://vllm.example/v1"), transport=t
    ),
    "inherited (deepseek)": lambda t: DeepSeekAIProvider(
        DeepSeekConfig(api_key="k", model="deepseek-chat"), transport=t
    ),
}


@pytest.mark.parametrize("kind", list(_PROVIDERS))
async def test_every_request_goes_through_the_transport(kind: str) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_COMPLETION)

    provider = _PROVIDERS[kind](httpx.MockTransport(handler))
    try:
        response = await provider.generate(
            AIContext(messages=[AIMessage(role="user", content="hello")])
        )
    finally:
        await provider.close()

    assert response.content == "ok"
    assert len(seen) == 1
    assert seen[0].url.path.endswith("/chat/completions")


async def test_the_sdk_defaults_stay_redirects_followed_through_the_transport() -> None:
    """The transport sits inside the SDK's own default client: redirects are
    still followed, each hop through the transport, where a policy judges it."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(307, headers={"location": "https://llm.example/v2/chat"})
        return httpx.Response(200, json=_COMPLETION)

    provider = _PROVIDERS["openai"](httpx.MockTransport(handler))
    try:
        response = await provider.generate(
            AIContext(messages=[AIMessage(role="user", content="hello")])
        )
    finally:
        await provider.close()

    assert response.content == "ok"
    assert seen == ["/v1/chat/completions", "/v2/chat"]
