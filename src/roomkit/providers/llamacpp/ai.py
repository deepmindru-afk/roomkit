"""llama.cpp provider: a local model RoomKit downloads, starts and stops itself."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar

from roomkit.providers.ai.base import AIContext, AIResponse, ModelInfo, StreamEvent
from roomkit.providers.llamacpp.config import LlamaCppConfig
from roomkit.providers.llamacpp.server import LlamaServer
from roomkit.providers.vllm import _openai_config, _VLLMProvider
from roomkit.providers.vllm.config import VLLMConfig


class LlamaCppAIProvider(_VLLMProvider):
    """Run a GGUF model locally through llama.cpp, with nothing to install or start.

    The first request (or :meth:`start`) downloads the pinned llama.cpp build
    for this machine and the model, then starts ``llama-server`` on a free
    local port; :meth:`close` stops it. Everything else — tool calls in the
    model's native format (``--jinja``), streaming, ``enable_thinking`` — is the
    OpenAI-compatible path RoomKit already uses for vLLM.

    Example::

        from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig

        ai = LlamaCppAIProvider(
            LlamaCppConfig(model="unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")
        )
    """

    _install_extra: ClassVar[str] = "llamacpp"

    def __init__(self, config: LlamaCppConfig) -> None:
        self._server = LlamaServer(config)
        self._llama_config = config
        super().__init__(
            _openai_config(
                VLLMConfig(
                    model=config.model,
                    base_url=self._server.base_url,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    timeout=config.timeout,
                    enable_thinking=config.enable_thinking,
                )
            )
        )

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def _provider_name(self) -> str:
        return "llamacpp"

    @property
    def base_url(self) -> str:
        """The local OpenAI-compatible endpoint of the managed ``llama-server``."""
        return self._server.base_url

    async def start(self) -> None:
        """Download what is missing and start the server now, not on the first request.

        Call it at startup so the first user turn does not wait for the model
        to load. Safe to call more than once.
        """
        await self._server.start()

    async def generate(self, context: AIContext) -> AIResponse:
        await self.start()
        return await super().generate(context)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        await self.start()
        async for event in super().generate_structured_stream(context):
            yield event

    async def list_models(self) -> list[ModelInfo]:
        await self.start()
        return await super().list_models()

    async def close(self) -> None:
        """Close the HTTP client and stop ``llama-server``."""
        try:
            await super().close()
        finally:
            await self._server.stop()
