"""A streamed turn of the OpenAI family reports its usage and a cost (RMK-312).

Every tool round streams, and without ``stream_options.include_usage`` the
server sends no usage: the turn reported ``{}`` and priced at zero. The
family now asks for it by default; a server that rejects the option sets
``include_stream_usage=False``. Cerebras keeps False: it sends usage unasked.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from roomkit.providers.ai.base import AIContext, AIMessage, StreamDone
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.vllm.config import VLLMConfig

_MODEL = "gpt-5.6-sol"
_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 30,
    "total_tokens": 130,
    "prompt_tokens_details": {"cached_tokens": 80},
    "completion_tokens_details": {"reasoning_tokens": 20},
}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (OpenAIConfig(api_key="k", model=_MODEL), True),
        (DeepSeekConfig(api_key="k", model="deepseek-chat"), True),
        (AzureAIConfig(api_key="k", azure_endpoint="https://x", model="d"), True),
        (VLLMConfig(model="m"), True),
        (CerebrasConfig(api_key="k", model="gpt-oss-120b"), False),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, BaseModel) else "",
)
def test_the_openai_family_asks_for_stream_usage_by_default(
    config: BaseModel, expected: bool
) -> None:
    assert config.include_stream_usage is expected  # ty: ignore[unresolved-attribute]


def _stream(requests: list[dict[str, Any]]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        frames = [
            {"choices": [{"index": 0, "delta": {"content": "Sunny."}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        if body.get("stream_options", {}).get("include_usage"):
            frames.append({"choices": [], "usage": _USAGE})
        chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": _MODEL}
        payload = "".join(f"data: {json.dumps({**chunk, **f})}\n\n" for f in frames)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload + "data: [DONE]\n\n",
        )

    return handle


@asynccontextmanager
async def _provider(
    handler: Callable[[httpx.Request], httpx.Response], **config: Any
) -> AsyncIterator[OpenAIAIProvider]:
    sdk = pytest.importorskip("openai")
    constructor = sdk.AsyncOpenAI
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch(
            "openai.AsyncOpenAI",
            side_effect=lambda **kwargs: constructor(http_client=client, **kwargs),
        ):
            provider = OpenAIAIProvider(OpenAIConfig(api_key="k", model=_MODEL, **config))
        try:
            yield provider
        finally:
            await provider.close()


async def _streamed_usage(provider: OpenAIAIProvider) -> dict[str, int]:
    context = AIContext(messages=[AIMessage(role="user", content="Weather?")])
    events = [event async for event in provider.generate_structured_stream(context)]
    assert isinstance(events[-1], StreamDone)
    return events[-1].usage


async def test_a_streamed_turn_reports_its_usage_and_a_cost() -> None:
    requests: list[dict[str, Any]] = []
    async with _provider(_stream(requests)) as provider:
        usage = await _streamed_usage(provider)
        entry = provider.catalog_entry()

    assert requests[0]["stream_options"] == {"include_usage": True}
    # reasoning_tokens is a detail of output_tokens, which already counts it.
    assert usage == {
        "input_tokens": 20,
        "output_tokens": 30,
        "cache_read_input_tokens": 80,
        "reasoning_tokens": 20,
    }
    assert entry is not None and entry.pricing is not None
    assert entry.pricing.cost_for(usage) > 0
    assert entry.pricing.cost_for(usage) == entry.pricing.cost_for(
        {k: v for k, v in usage.items() if k != "reasoning_tokens"}
    )


async def test_a_server_that_rejects_the_option_can_turn_it_off() -> None:
    requests: list[dict[str, Any]] = []
    async with _provider(_stream(requests), include_stream_usage=False) as provider:
        usage = await _streamed_usage(provider)

    assert "stream_options" not in requests[0]
    assert usage == {}
