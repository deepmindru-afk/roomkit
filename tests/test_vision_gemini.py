"""Tests for GeminiVisionProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider


class TestGeminiVisionConfig:
    def test_defaults(self) -> None:
        config = GeminiVisionConfig()
        assert config.model == "gemini-3.8-flash"
        assert config.api_key == ""
        assert config.max_tokens == 1024

    def test_custom(self) -> None:
        config = GeminiVisionConfig(
            api_key="AIza-test",
            model="gemini-2.5-flash",
            max_tokens=500,
        )
        assert config.api_key == "AIza-test"
        assert config.model == "gemini-2.5-flash"


class TestGeminiVisionProvider:
    def test_name(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))
        assert provider.name == "gemini-vision:gemini-3.8-flash"

    def test_custom_model_name(self) -> None:
        config = GeminiVisionConfig(api_key="test", model="gemini-2.5-flash")
        provider = GeminiVisionProvider(config)
        assert provider.name == "gemini-vision:gemini-2.5-flash"

    def test_config_constructor(self) -> None:
        config = GeminiVisionConfig(api_key="k", model="custom")
        provider = GeminiVisionProvider(config=config)
        assert provider.name == "gemini-vision:custom"

    async def test_analyze_frame(self) -> None:
        """Test analyze_frame with a mocked Gemini client."""
        config = GeminiVisionConfig(api_key="test-key", model="gemini-3.1-flash-lite")
        provider = GeminiVisionProvider(config)

        # Mock response
        mock_response = MagicMock()
        mock_response.text = "A person at a desk with a monitor"
        mock_response.usage_metadata.prompt_token_count = 80
        mock_response.usage_metadata.candidates_token_count = 15

        # Mock client
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        # Mock types for Part.from_bytes
        mock_types = MagicMock()
        provider._types = mock_types

        frame = VideoFrame(
            data=b"\x00" * (64 * 48 * 3),
            codec="raw_rgb24",
            width=64,
            height=48,
        )

        result = await provider.analyze_frame(frame)

        assert result.description == "A person at a desk with a monitor"
        assert result.metadata["model"] == "gemini-3.1-flash-lite"
        assert result.metadata["usage"]["prompt_tokens"] == 80
        assert result.metadata["usage"]["completion_tokens"] == 15

        mock_client.aio.models.generate_content.assert_called_once()

    async def test_automatic_function_calling_is_off(self) -> None:
        """No tools here; left on, the SDK logs a warning on every frame."""
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test-key"))
        mock_response = MagicMock()
        mock_response.text = "A desk"
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        mock_types = MagicMock()
        provider._types = mock_types

        await provider.analyze_frame(
            VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)
        )

        mock_types.AutomaticFunctionCallingConfig.assert_called_once_with(disable=True)
        sent = mock_types.GenerateContentConfig.call_args.kwargs
        assert sent["automatic_function_calling"] is (
            mock_types.AutomaticFunctionCallingConfig.return_value
        )

    async def test_analyze_frame_empty_response(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.usage_metadata = None

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        provider._types = MagicMock()

        frame = VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)
        result = await provider.analyze_frame(frame)
        assert result.description == ""

    async def test_close(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))
        client = MagicMock()
        client.aio.aclose = AsyncMock()
        http = httpx.AsyncClient()
        provider._client = client
        provider._http = http
        provider._types = MagicMock()

        await provider.close()
        client.aio.aclose.assert_awaited_once()
        assert http.is_closed  # the SDK never closes a client it was given
        assert provider._client is None
        assert provider._http is None
        assert provider._types is None

    async def test_close_no_client(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))
        await provider.close()  # no-op


class _ApiError(Exception):
    """What google-genai raises: the HTTP status on ``code``."""

    def __init__(self, code: int) -> None:
        super().__init__(f"{code} error")
        self.code = code


def _frame() -> VideoFrame:
    return VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)


def _vision(answers: list[object], **config: object) -> tuple[GeminiVisionProvider, MagicMock]:
    """A provider whose client plays *answers* in turn: a response, or an
    exception to raise. Returns it with its mocked ``types``."""
    provider = GeminiVisionProvider(
        GeminiVisionConfig(api_key="test-key", **config)  # type: ignore[arg-type]
    )
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=answers)
    provider._client = client
    provider._types = MagicMock()
    return provider, provider._types


def _answer(text: str = "A desk") -> MagicMock:
    response = MagicMock()
    response.text = text
    return response


def _sent_thinking(types: MagicMock) -> list[bool]:
    """Per call, whether its config carried a thinking config."""
    return [
        "thinking_config" in call.kwargs for call in types.GenerateContentConfig.call_args_list
    ]


class TestThinkingOffWhereTheModelTakesIt:
    """No one setting minimises reasoning on every model (measured 2026-09-27):
    ``thinking_budget=0`` is sent, and dropped for a model that answers it 400."""

    async def test_the_budget_is_sent_to_a_model_that_takes_it(self) -> None:
        provider, types = _vision([_answer()])

        await provider.analyze_frame(_frame())

        types.ThinkingConfig.assert_called_once_with(thinking_budget=0)
        assert _sent_thinking(types) == [True]

    async def test_a_model_that_refuses_it_is_asked_again_without_it(self) -> None:
        provider, types = _vision(
            [_ApiError(400), _answer("Blue"), _answer("Red")], model="gemini-3.5-flash-lite"
        )

        first = await provider.analyze_frame(_frame())
        second = await provider.analyze_frame(_frame())

        assert (first.description, second.description) == ("Blue", "Red")
        # Refused once, then never sent again.
        assert _sent_thinking(types) == [True, False, False]

    async def test_a_failing_retry_raises_its_own_error_and_forgets_nothing(self) -> None:
        provider, types = _vision([_ApiError(400), _ApiError(413), _answer()])

        with pytest.raises(_ApiError) as caught:
            await provider.analyze_frame(_frame())
        assert caught.value.code == 413

        await provider.analyze_frame(_frame())
        # The retry failed for its own reason: the budget is tried again.
        assert _sent_thinking(types) == [True, False, True]

    async def test_the_callers_own_thinking_config_is_never_dropped(self) -> None:
        own = object()
        provider, types = _vision([_ApiError(400)], extra_config={"thinking_config": own})

        with pytest.raises(_ApiError):
            await provider.analyze_frame(_frame())

        assert types.GenerateContentConfig.call_count == 1
        assert types.GenerateContentConfig.call_args.kwargs["thinking_config"] is own

    async def test_another_error_is_not_retried(self) -> None:
        provider, types = _vision([_ApiError(429)])

        with pytest.raises(_ApiError):
            await provider.analyze_frame(_frame())

        assert types.GenerateContentConfig.call_count == 1


class TestExports:
    def test_importable_from_subpackage(self) -> None:
        from roomkit.video import GeminiVisionConfig, GeminiVisionProvider

        assert GeminiVisionProvider is not None
        assert GeminiVisionConfig is not None

    def test_importable_from_video(self) -> None:
        from roomkit.video import GeminiVisionConfig, GeminiVisionProvider

        assert GeminiVisionProvider is not None
        assert GeminiVisionConfig is not None
