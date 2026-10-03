"""An outbound policy on an OpenAI-compatible provider, through ``transport=``.

A host that lets its users name an OpenAI-compatible endpoint (a self-hosted
model, a gateway) must not let that URL reach its own network: a cloud
metadata address, a loopback service. The policy belongs where the address is
dialled, so it judges every request, redirects included. Shows:
- OpenAIAIProvider(config, transport=...): every request of the SDK goes
  through the transport, inside the SDK's own default client
- a transport wrapper that refuses a private address before a byte leaves
- the same seam on AzureAIProvider, OpenRouterAIProvider, create_vllm_provider

Runs without a key: a MockTransport stands in for the endpoint.

Run with:
    uv run python examples/openai_outbound_policy.py
"""

from __future__ import annotations

import asyncio

import httpx
from shared import setup_logging

from roomkit.providers.ai.base import AIContext, AIMessage, ProviderError
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.url_safety import validate_public_url

logger = setup_logging("openai_outbound_policy")

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "local-model",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


class PublicOnly(httpx.AsyncBaseTransport):
    """Refuses a request to a private or reserved address, then hands it on.

    The demo skips DNS (``resolve_dns=False``) to run offline; a production
    policy resolves the name and connects to the address it judged.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        validate_public_url(str(request.url), resolve_dns=False)
        return await self._inner.handle_async_request(request)


def root_cause(exc: BaseException) -> BaseException:
    """The policy's refusal, under the SDK's connection error that carries it."""
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return exc


async def ask(base_url: str) -> None:
    endpoint = httpx.MockTransport(lambda request: httpx.Response(200, json=COMPLETION))
    config = OpenAIConfig(api_key="user-key", model="local-model", base_url=base_url)
    provider = OpenAIAIProvider(config, transport=PublicOnly(endpoint))
    try:
        reply = await provider.generate(AIContext(messages=[AIMessage(role="user", content="Hi")]))
        logger.info("%s answered: %s", base_url, reply.content)
    except ProviderError as exc:
        logger.info("%s refused: %s", base_url, root_cause(exc))
    finally:
        await provider.close()


async def main() -> None:
    await ask("https://models.example.com/v1")
    await ask("http://169.254.169.254/v1")  # a cloud metadata address


if __name__ == "__main__":
    asyncio.run(main())
