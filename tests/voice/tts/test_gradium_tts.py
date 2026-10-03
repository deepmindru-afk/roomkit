"""Tests for the Gradium TTS provider."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_gradium_module() -> MagicMock:
    """Create a MagicMock that stands in for the gradium module."""
    mod = MagicMock()
    client = MagicMock()
    mod.GradiumClient.return_value = client
    return mod


def _make_provider(grad_mock: MagicMock, **config_kwargs: Any) -> Any:
    """Build a GradiumTTSProvider with mocked gradium dependency."""
    with patch.dict("sys.modules", {"gradium": grad_mock}):
        import roomkit.voice.tts.gradium as tts_mod

        importlib.reload(tts_mod)
        from roomkit.voice.tts.gradium import GradiumTTSConfig, GradiumTTSProvider

        cfg = GradiumTTSConfig(**config_kwargs)
        return GradiumTTSProvider(cfg)


class TestGradiumTTSProvider:
    def test_constructor_stores_config(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="test-key")
        assert provider._config.api_key == "test-key"
        assert provider._config.voice_id == "default"
        assert provider._client is None  # lazy init

    def test_name(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")
        assert provider.name == "GradiumTTS"

    def test_default_voice(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", voice_id="my-voice")
        assert provider.default_voice == "my-voice"

    def test_supports_streaming_input(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")
        assert provider.supports_streaming_input is True

    def test_sample_rate_pcm_16000(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", output_format="pcm_16000")
        assert provider._get_sample_rate() == 16000

    def test_sample_rate_pcm_24000(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", output_format="pcm_24000")
        assert provider._get_sample_rate() == 24000

    def test_sample_rate_pcm_48000(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", output_format="pcm_48000")
        assert provider._get_sample_rate() == 48000

    def test_sample_rate_plain_pcm_defaults_48k(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", output_format="pcm")
        assert provider._get_sample_rate() == 48000

    def test_audio_format_pcm(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", output_format="pcm_16000")
        assert provider._get_audio_format() == "pcm_s16le"

    def test_audio_format_opus(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", output_format="opus")
        assert provider._get_audio_format() == "opus"

    async def test_synthesize(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", output_format="pcm_16000")

        fake_result = SimpleNamespace(raw_data=b"\x00\x01" * 50)
        fake_client = MagicMock()
        fake_client.tts = AsyncMock(return_value=fake_result)
        provider._client = fake_client

        result = await provider.synthesize("hello world")

        assert result.transcript == "hello world"
        assert result.url.startswith("data:audio/pcm;base64,")
        fake_client.tts.assert_awaited_once()

    async def test_close_clears_client(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")
        provider._client = MagicMock()
        await provider.close()
        assert provider._client is None

    def test_build_setup_with_options(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(
            grad,
            api_key="k",
            voice_id="my-voice",
            model_name="fast",
            padding_bonus=-1.0,
            temperature=0.5,
            rewrite_rules="en",
        )
        setup = provider._build_setup(output_format="pcm_16000")
        assert setup["model_name"] == "fast"
        assert setup["voice_id"] == "my-voice"
        assert setup["json_config"]["padding_bonus"] == -1.0
        assert setup["json_config"]["temp"] == 0.5
        assert setup["json_config"]["rewrite_rules"] == "en"

    def test_build_setup_default_voice(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", voice_id="default")
        setup = provider._build_setup(output_format="pcm_16000")
        # default voice uses "voice" key, not "voice_id"
        assert setup["voice"] == "default"
        assert "voice_id" not in setup


def _from_text(provider: Any) -> Any:
    return provider.synthesize_stream("hello")


def _from_text_stream(provider: Any) -> Any:
    async def text() -> Any:
        yield "hello"

    return provider.synthesize_stream_input(text())


def _fake_stream() -> Any:
    async def iter_bytes() -> Any:
        yield b"\x00\x01" * 8

    return SimpleNamespace(sample_rate=48000, iter_bytes=iter_bytes)


_STREAMS = pytest.mark.parametrize(
    "open_stream", [_from_text, _from_text_stream], ids=["text", "text-stream"]
)


class TestGradiumStreamedWav:
    """A WAV stream opens with a RIFF header the chunks would play as audio (RMK-413)."""

    @_STREAMS
    @pytest.mark.parametrize(
        ("configured", "asked"),
        [("wav", "pcm"), ("pcm", "pcm"), ("pcm_16000", "pcm_16000"), ("opus", "opus")],
    )
    async def test_a_stream_asks_raw_pcm_in_place_of_wav(
        self, open_stream: Any, configured: str, asked: str
    ) -> None:
        provider = _make_provider(_mock_gradium_module(), api_key="k", output_format=configured)
        fake_client = MagicMock()
        fake_client.tts_stream = AsyncMock(return_value=_fake_stream())
        provider._client = fake_client

        chunks = [chunk async for chunk in open_stream(provider)]

        assert fake_client.tts_stream.call_args.args[0]["output_format"] == asked
        assert chunks[0].data == b"\x00\x01" * 8

    async def test_synthesize_keeps_wav(self) -> None:
        provider = _make_provider(_mock_gradium_module(), api_key="k", output_format="wav")
        fake_client = MagicMock()
        fake_client.tts = AsyncMock(return_value=SimpleNamespace(raw_data=b"RIFF"))
        provider._client = fake_client

        result = await provider.synthesize("hello")

        assert fake_client.tts.call_args.args[0]["output_format"] == "wav"
        assert result.url.startswith("data:audio/wav;base64,")


class TestGradiumListVoices:
    """``list_voices`` answers VoiceInfo, like every catalog (RFC §12.2)."""

    async def test_voices_are_voice_infos_filtered_like_every_catalog(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")
        client = MagicMock()
        client.voice_get = AsyncMock(
            return_value={
                "voices": [
                    {"uid": "u1", "name": "Élise", "language": "fr", "description": "Calme"},
                    {"uid": "u2", "name": "Tom", "language": "en"},
                    {"uid": "u3"},
                ]
            }
        )
        provider._client = client

        french = await provider.list_voices(language="fr")
        everyone = await provider.list_voices()

        assert [(v.id, v.name, v.description) for v in french] == [("u1", "Élise", "Calme")]
        # A voice with no name is named by its uid.
        assert [(v.id, v.name) for v in everyone] == [("u1", "Élise"), ("u2", "Tom"), ("u3", "u3")]
        # Fetched once, filtered twice.
        assert client.voice_get.await_count == 1
