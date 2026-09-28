"""Tests for the Meta Muse Spark chat provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ProviderError
from roomkit.providers.meta.config import MetaConfig
from roomkit.providers.meta.models import MODELS


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    pass


def _openai_stub() -> MagicMock:
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    mod.APIConnectionError = _FakeAPIConnectionError
    return mod


def _provider(stub: MagicMock | None = None, **overrides: Any) -> Any:
    with patch.dict("sys.modules", {"openai": stub or _openai_stub()}):
        from roomkit.providers.meta.ai import MetaAIProvider

        provider = MetaAIProvider(MetaConfig(api_key="k", **overrides))
    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock(return_value=_response())
    return provider


def _response(text: str = "Québec") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop"
            )
        ],
        model="muse-spark-1.3",
        usage=SimpleNamespace(prompt_tokens=16, completion_tokens=82),
    )


def _context(**overrides: Any) -> AIContext:
    return AIContext(
        messages=[AIMessage(role="user", content="Capitale du Québec ?")], **overrides
    )


def _sent(provider: Any) -> dict[str, Any]:
    return provider._client.chat.completions.create.await_args.kwargs


_TOOL = AITool(
    name="get_weather",
    description="Météo d'une ville",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


class TestConfig:
    def test_defaults(self) -> None:
        cfg = MetaConfig(api_key="k")
        assert cfg.base_url == "https://api.meta.ai/v1"
        assert cfg.model == "muse-spark-1.3"
        assert cfg.use_max_completion_tokens is True
        assert cfg.include_stream_usage is True

    def test_client_points_at_meta(self) -> None:
        stub = _openai_stub()
        _provider(stub)
        assert stub.AsyncOpenAI.call_args.kwargs["base_url"] == "https://api.meta.ai/v1"


class TestReasoningEffort:
    async def test_none_is_sent_as_minimal(self) -> None:
        # Muse Spark cannot stop reasoning: "none" is a 400 from the service.
        provider = _provider()
        await provider.generate(_context(reasoning_effort="none"))
        assert _sent(provider)["reasoning_effort"] == "minimal"

    async def test_the_turns_effort_wins_over_the_config(self) -> None:
        provider = _provider(reasoning_effort="low")
        await provider.generate(_context(reasoning_effort="high"))
        assert _sent(provider)["reasoning_effort"] == "high"

    async def test_effort_rides_tool_turns_too(self) -> None:
        provider = _provider(reasoning_effort="low")
        await provider.generate(_context(tools=[_TOOL]))
        assert _sent(provider)["reasoning_effort"] == "low"

    async def test_no_effort_leaves_the_service_default(self) -> None:
        provider = _provider()
        await provider.generate(_context())
        assert "reasoning_effort" not in _sent(provider)


class TestProvider:
    async def test_generate(self) -> None:
        provider = _provider()
        result = await provider.generate(_context())
        assert result.content == "Québec"
        assert _sent(provider)["model"] == "muse-spark-1.3"

    def test_provider_name_and_vision(self) -> None:
        provider = _provider()
        assert provider._provider_name == "meta"
        assert provider.supports_vision is True
        assert _provider(model="muse-spark-9.9").supports_vision is True  # newer than the catalog

    async def test_list_models_keeps_the_chat_models(self) -> None:
        provider = _provider()
        ids = [
            "sam-3.1",
            "muse-spark-1.3",
            "muse-image-1.0",
            "muse-voice-transcribe-1.0",
            "muse-spark-1.2-contributor",
        ]
        provider._client.models.list = AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(id=i) for i in ids])
        )
        listed = await provider.list_models()
        assert [m.id for m in listed] == ["muse-spark-1.3", "muse-spark-1.2-contributor"]
        assert listed[0].context_window == 1_048_576  # backfilled from the catalog

    async def test_status_errors_carry_the_meta_provider(self) -> None:
        provider = _provider()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=_FakeAPIStatusError("billing_not_configured", status_code=402)
        )
        with pytest.raises(ProviderError) as info:
            await provider.generate(_context())
        assert info.value.provider == "meta"
        assert info.value.status_code == 402
        assert info.value.retryable is False


class TestCatalog:
    def test_every_model_is_priced_and_reasons(self) -> None:
        for model in MODELS:
            assert model.pricing is not None
            assert "thinking" in model.capabilities
            assert model.context_window == 1_048_576

    def test_contributor_ids_say_what_they_cost(self) -> None:
        contributors = [m for m in MODELS if m.id.endswith("-contributor")]
        assert contributors
        assert all("Meta trains on the traffic" in (m.display_name or "") for m in contributors)
        assert all(m.id != MetaConfig(api_key="k").model for m in contributors)
