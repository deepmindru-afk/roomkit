"""The OpenAI Chat Completions wire: OpenAI and the ten providers built on it."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any, ClassVar

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.cerebras.ai import CerebrasAIProvider
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.deepseek.ai import DeepSeekAIProvider
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.litellm.ai import LiteLLMAIProvider
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.llamacpp.ai import LlamaCppAIProvider
from roomkit.providers.llamacpp.config import LlamaCppConfig
from roomkit.providers.meta.ai import MetaAIProvider
from roomkit.providers.meta.config import MetaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.qwen.ai import QwenAIProvider
from roomkit.providers.qwen.config import QwenConfig
from roomkit.providers.vllm import VLLMConfig, _VLLMProvider, create_vllm_provider
from roomkit.providers.xai.ai import XAIAIProvider
from roomkit.providers.xai.config import XAIConfig
from tests.text_conformance.driver import (
    REDACTED_REASONING,
    SIGNED_REASONING,
    Driver,
    ReasoningConvention,
)
from tests.text_conformance.script import Call, Item, Script

_FINISH = {"stop": "stop", "tool": "tool_calls", "cut": "length", "none": None}
_THINK = re.compile(r"^<think>(.*?)</think>", re.DOTALL)


def _pieces(text: str, count: int) -> list[str]:
    size = max(1, -(-len(text) // max(count, 1)))
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _usage(script: Script) -> Any:
    usage = script.usage
    return SimpleNamespace(
        prompt_tokens=usage.input + usage.cache_read + usage.cache_write,
        completion_tokens=usage.output,
        total_tokens=None,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=usage.cache_read, cache_write_tokens=usage.cache_write
        ),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=usage.reasoning),
    )


def _fragment(call: Call, first: bool, piece: str) -> Any:
    return SimpleNamespace(
        index=call.index,
        id=call.id if first else None,
        function=SimpleNamespace(name=call.name if first else None, arguments=piece),
    )


def _delta(**fields: Any) -> Any:
    return SimpleNamespace(**{"content": None, "tool_calls": None, "refusal": None, **fields})


def _chunk(delta: Any = None, finish: str | None = None, usage: Any = None) -> Any:
    choices = [] if delta is None else [SimpleNamespace(delta=delta, finish_reason=finish)]
    return SimpleNamespace(choices=choices, usage=usage)


def _call_chunks(script: Script) -> list[Any]:
    pieces = [_pieces(call.arguments, call.fragments) for call in script.calls]
    chunks: list[Any] = []
    if script.calls_in_one_chunk:
        firsts = [_fragment(c, True, p[0]) for c, p in zip(script.calls, pieces, strict=True)]
        chunks.append(_chunk(_delta(tool_calls=firsts)))
        for call, rest in zip(script.calls, pieces, strict=True):
            chunks.extend(_chunk(_delta(tool_calls=[_fragment(call, False, p)])) for p in rest[1:])
        return chunks
    for call, parts in zip(script.calls, pieces, strict=True):
        for n, piece in enumerate(parts):
            chunks.append(_chunk(_delta(tool_calls=[_fragment(call, n == 0, piece)])))
    return chunks


def _stream(script: Script) -> list[Any]:
    chunks = [
        _chunk(_delta(reasoning_content=block.text)) for block in script.reasoning if block.text
    ]
    if script.text:
        chunks.append(_chunk(_delta(content=script.text)))
    chunks.extend(_call_chunks(script))
    finish = _FINISH[script.finish]
    if finish is not None:
        chunks.append(_chunk(_delta(), finish))
    chunks.append(_chunk(usage=_usage(script)))
    return chunks


def _response(script: Script) -> Any:
    calls = [
        SimpleNamespace(id=c.id, function=SimpleNamespace(name=c.name, arguments=c.arguments))
        for c in script.calls
    ]
    reasoning = "".join(block.text for block in script.reasoning) or None
    message = SimpleNamespace(
        content=script.text or None,
        tool_calls=calls or None,
        refusal=None,
        reasoning_content=reasoning,
    )
    choice = SimpleNamespace(message=message, finish_reason=_FINISH[script.finish])
    return SimpleNamespace(choices=[choice], usage=_usage(script), model="m")


class _StartedServer:
    """A ``llama-server`` already running: nothing to download or start."""

    base_url = "http://127.0.0.1:1/v1"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _llamacpp() -> AIProvider:
    provider = LlamaCppAIProvider(LlamaCppConfig(model="unsloth/Qwen3-4B-GGUF:Q4_K_M"))
    provider._server = _StartedServer()  # type: ignore[assignment]
    return provider


async def _iterate(chunks: list[Any]) -> AsyncIterator[Any]:
    for chunk in chunks:
        yield chunk


def _assistant_items(message: dict[str, Any]) -> list[Item]:
    items: list[Item] = []
    if message.get("reasoning"):
        items.append(("field", message["reasoning"]))
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    if content:
        match = _THINK.match(content)
        if match:
            items.append(("inline", match.group(1)))
            content = content[match.end() :]
        if content:
            items.append(("text", content))
    for call in message.get("tool_calls") or []:
        function = call["function"]
        items.append(("call", call["id"], json.loads(function["arguments"])))
    return items


def _carries_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        part.get("type") == "image_url" for part in content if isinstance(part, dict)
    )


class OpenAIWire(Driver):
    """A provider built on OpenAI's Chat Completions client."""

    cannot: ClassVar[dict[str, str]] = {
        SIGNED_REASONING: "Chat Completions carries no reasoning signature",
        REDACTED_REASONING: "Chat Completions has no redacted reasoning",
    }

    def __init__(
        self,
        provider_cls: type[AIProvider],
        build: Callable[[], AIProvider],
        *,
        label: str,
        reasoning: ReasoningConvention = "inline",
        refused_names: tuple[str, ...] = (),
        accepted_names: tuple[str, ...] = ("lookup", "files.read"),
    ) -> None:
        super().__init__()
        self._build = build
        self.label = label  # type: ignore[misc]
        self.covers = (provider_cls,)  # type: ignore[misc]
        self.reasoning = reasoning  # type: ignore[misc]
        self.refused_names = refused_names  # type: ignore[misc]
        self.accepted_names = accepted_names  # type: ignore[misc]

    def provider(self, script: Script) -> AIProvider:
        provider = self._build()

        async def create(**kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if kwargs.get("stream"):
                return _iterate(_stream(script))
            return _response(script)

        provider._client = SimpleNamespace(  # type: ignore[attr-defined]
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return provider

    def declared(self, request: Any) -> dict[str, dict[str, Any]]:
        return {
            tool["function"]["name"]: tool["function"]["parameters"]
            for tool in request.get("tools") or []
        }

    def replayed(self, request: Any) -> list[Item]:
        items: list[Item] = []
        for message in request["messages"]:
            if message["role"] == "assistant":
                items.extend(_assistant_items(message))
            elif message["role"] == "tool":
                items.append(("result", message["tool_call_id"], message["content"], None))
            elif _carries_image(message):
                items.append(("image",))
        return items


def wires() -> list[Driver]:
    """One driver per provider class on the OpenAI wire."""
    return [
        OpenAIWire(
            OpenAIAIProvider,
            lambda: OpenAIAIProvider(OpenAIConfig(api_key="k", model="gpt-5.4")),
            label="openai",
            refused_names=("files.read",),
            accepted_names=("lookup",),
        ),
        OpenAIWire(
            AzureAIProvider,
            lambda: AzureAIProvider(
                AzureAIConfig(api_key="k", azure_endpoint="https://x.azure.com", model="m")
            ),
            label="azure",
        ),
        OpenAIWire(
            CerebrasAIProvider,
            lambda: CerebrasAIProvider(CerebrasConfig(api_key="k", model="gpt-oss-120b")),
            label="cerebras",
            reasoning="field",
        ),
        OpenAIWire(
            DeepSeekAIProvider,
            lambda: DeepSeekAIProvider(DeepSeekConfig(api_key="k", model="deepseek-chat")),
            label="deepseek",
        ),
        OpenAIWire(
            LiteLLMAIProvider,
            lambda: LiteLLMAIProvider(LiteLLMConfig(api_key="k", model="gpt-4o")),
            label="litellm",
        ),
        OpenAIWire(
            LlamaCppAIProvider,
            _llamacpp,
            label="llamacpp",
        ),
        OpenAIWire(
            MetaAIProvider,
            lambda: MetaAIProvider(MetaConfig(api_key="k", model="muse-spark")),
            label="meta",
        ),
        OpenAIWire(
            OpenRouterAIProvider,
            lambda: OpenRouterAIProvider(OpenRouterConfig(api_key="k", model="openai/gpt-4o")),
            label="openrouter",
        ),
        OpenAIWire(
            QwenAIProvider,
            lambda: QwenAIProvider(QwenConfig(api_key="k", model="qwen-plus")),
            label="qwen",
        ),
        OpenAIWire(
            _VLLMProvider,
            lambda: create_vllm_provider(VLLMConfig(model="local")),
            label="vllm",
        ),
        OpenAIWire(
            XAIAIProvider,
            lambda: XAIAIProvider(XAIConfig(api_key="k", model="grok-4")),
            label="xai",
        ),
    ]
