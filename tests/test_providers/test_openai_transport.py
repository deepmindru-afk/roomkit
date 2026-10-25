"""``transport=``: every request of an OpenAI-family provider goes through it.

The seam a host puts an outbound policy in (one that judges the address
actually dialled), and a test a ``MockTransport``. Honoured by
``OpenAIAIProvider`` and every provider built on it, those that build their
own SDK client included (Azure, OpenRouter) and ``create_vllm_provider``; and
by the other clients built on the ``openai`` SDK, whose endpoint a host's
users name too: the OpenAI, Azure, xAI and Meta image providers and
``OpenAIVisionProvider``; and by the requests beside a completion, such as
LiteLLM's model listing (RMK-453).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

import httpx
import openai
import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, ProviderError
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig, AzureImageConfig
from roomkit.providers.azure.image import AzureImageProvider
from roomkit.providers.deepseek import DeepSeekAIProvider, DeepSeekConfig
from roomkit.providers.litellm import LiteLLMAIProvider, LiteLLMConfig
from roomkit.providers.meta.config import MetaImageConfig
from roomkit.providers.meta.image import MetaImageProvider
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig, OpenAIImageConfig
from roomkit.providers.openai.image import OpenAIImageProvider
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.url_safety import validate_public_url
from roomkit.providers.vllm import create_vllm_provider
from roomkit.providers.vllm.config import VLLMConfig
from roomkit.providers.xai.config import XAIImageConfig
from roomkit.providers.xai.image import XAIImageProvider
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider

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


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_IMAGES = {"created": 0, "data": [{"b64_json": _PNG_B64}]}


def _answer(request: httpx.Request) -> httpx.Response:
    """The endpoint's answer: an image to an images call, an empty model
    listing to a listing, a completion otherwise."""
    if request.url.path.endswith("/images/generations"):
        return httpx.Response(200, json=_IMAGES)
    if request.url.path.endswith("/model/info"):
        return httpx.Response(200, json={"data": []})
    return httpx.Response(200, json=_COMPLETION)


class _Client(NamedTuple):
    """A client on the ``openai`` SDK, and one call it makes beside a completion."""

    build: Callable[[str, httpx.AsyncBaseTransport], Any]
    """The client for an endpoint at this origin, over this transport."""
    call: Callable[[Any], Awaitable[object]]


def _draw(provider: Any) -> Awaitable[object]:
    return provider.generate("an origami fox")


def _list(provider: Any) -> Awaitable[object]:
    return provider.list_models()


def _look(provider: Any) -> Awaitable[object]:
    return provider.analyze_frame(
        VideoFrame(data=b"\x00\x00\x01", codec="h264", width=2, height=2)
    )


_CALLS: dict[str, _Client] = {
    "openai image": _Client(
        lambda origin, t: OpenAIImageProvider(
            OpenAIImageConfig(api_key="k", model="gpt-image-2", base_url=f"{origin}/v1"),
            transport=t,
        ),
        _draw,
    ),
    "azure image": _Client(
        lambda origin, t: AzureImageProvider(
            AzureImageConfig(api_key="k", azure_endpoint=origin, model="dep"), transport=t
        ),
        _draw,
    ),
    "xai image": _Client(
        lambda origin, t: XAIImageProvider(
            XAIImageConfig(api_key="k", base_url=f"{origin}/v1"), transport=t
        ),
        _draw,
    ),
    "meta image": _Client(
        lambda origin, t: MetaImageProvider(
            MetaImageConfig(api_key="k", base_url=f"{origin}/v1"), transport=t
        ),
        _draw,
    ),
    "openai vision": _Client(
        lambda origin, t: OpenAIVisionProvider(
            OpenAIVisionConfig(base_url=f"{origin}/v1"), transport=t
        ),
        _look,
    ),
    "litellm models": _Client(
        lambda origin, t: LiteLLMAIProvider(
            LiteLLMConfig(api_key="k", model="m", base_url=origin), transport=t
        ),
        _list,
    ),
}


class _PublicOnly(httpx.AsyncBaseTransport):
    """A host's policy: refuses a private or reserved address, then hands on."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        validate_public_url(str(request.url), resolve_dns=False)
        return await self._inner.handle_async_request(request)


@pytest.mark.parametrize("kind", list(_CALLS))
async def test_each_call_sends_its_request_through_the_transport(kind: str) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _answer(request)

    client = _CALLS[kind]
    provider = client.build("https://models.example", httpx.MockTransport(handler))
    try:
        await client.call(provider)
    finally:
        await provider.close()

    assert [request.url.host for request in seen] == ["models.example"]


@pytest.mark.parametrize("kind", list(_CALLS))
async def test_a_policy_in_the_transport_refuses_a_private_endpoint_before_any_request(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The vision client keeps the SDK's default retries, each one refused by
    # the policy too; none waits here.
    monkeypatch.setattr("openai._base_client.INITIAL_RETRY_DELAY", 0.0)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _answer(request)

    client = _CALLS[kind]
    provider = client.build("https://169.254.169.254", _PublicOnly(httpx.MockTransport(handler)))
    try:
        with pytest.raises((ProviderError, openai.APIConnectionError)) as caught:
            await client.call(provider)
    finally:
        await provider.close()

    assert sent == []
    cause: BaseException = caught.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, ValueError)
    assert "169.254.169.254" in str(cause)
