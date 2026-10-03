"""Tests for ElevenLabsTTSProvider — config, expressive mode, and synthesis."""

from __future__ import annotations

import base64
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.voice.tts.context import TTSContext
from roomkit.voice.tts.elevenlabs import (
    EXPRESSIVE_TAGS,
    MODEL_MULTILINGUAL_V2,
    MODEL_V3,
    MODEL_V4,
    MODEL_V4_TURBO,
    ElevenLabsConfig,
    ElevenLabsTTSProvider,
)


async def _sdk_audio(*parts: bytes):
    """What the SDK's ``text_to_speech.convert()`` returns: an async generator of bytes."""
    for part in parts:
        yield part


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestElevenLabsConfig:
    def test_defaults(self):
        cfg = ElevenLabsConfig(api_key="test-key")
        assert cfg.voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert cfg.model_id == MODEL_MULTILINGUAL_V2
        assert cfg.stability == 0.5
        assert cfg.similarity_boost == 0.75
        assert cfg.style == 0.0
        assert cfg.use_speaker_boost is True
        assert cfg.output_format == "mp3_44100_128"
        assert cfg.optimize_streaming_latency is None
        assert cfg.expressive is False

    def test_custom_values(self):
        cfg = ElevenLabsConfig(
            api_key="test-key",
            voice_id="custom-voice",
            model_id="eleven_flash_v2_5",
            stability=0.8,
            similarity_boost=0.9,
            style=0.3,
            use_speaker_boost=False,
            output_format="pcm_24000",
            optimize_streaming_latency=1,
        )
        assert cfg.voice_id == "custom-voice"
        assert cfg.model_id == "eleven_flash_v2_5"
        assert cfg.stability == 0.8
        assert cfg.style == 0.3

    def test_expressive_flag(self):
        cfg = ElevenLabsConfig(api_key="test-key", expressive=True)
        assert cfg.expressive is True


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------


class TestProviderBasics:
    def test_name(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider.name == "ElevenLabsTTS"

    def test_default_voice(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", voice_id="custom"))
        assert provider.default_voice == "custom"

    def test_supports_streaming_input_off_by_default(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider.supports_streaming_input is False

    def test_supports_streaming_input_when_asked(self):
        """The default model streams input over the Text to Speech socket."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", stream_input=True))
        assert provider.supports_streaming_input is True

    def test_supports_streaming_input_v3_false(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=MODEL_V3))
        assert provider.supports_streaming_input is False


# ---------------------------------------------------------------------------
# Expressive mode
# ---------------------------------------------------------------------------


class TestExpressiveMode:
    def test_expressive_sets_v4_turbo_model(self):
        """expressive=True picks Eleven v4 Turbo."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._config.model_id == MODEL_V4_TURBO

    def test_expressive_overrides_a_model_without_audio_tags(self):
        """expressive=True replaces a model_id that does not render audio tags."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(
                api_key="k",
                model_id="eleven_flash_v2_5",
                expressive=True,
            )
        )
        assert provider._config.model_id == MODEL_V4_TURBO

    @pytest.mark.parametrize("model_id", [MODEL_V4, MODEL_V3, "eleven_v3_conversational"])
    def test_expressive_keeps_a_model_with_audio_tags(self, model_id):
        """An explicit v3 or v4 model already renders the tags and is kept."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", model_id=model_id, expressive=True)
        )
        assert provider._config.model_id == model_id

    def test_non_expressive_keeps_model(self):
        """Without expressive, model_id is untouched."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", model_id="eleven_flash_v2_5")
        )
        assert provider._config.model_id == "eleven_flash_v2_5"

    def test_is_v3_model_expressive(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._is_v3_model() is False

    @pytest.mark.parametrize("model_id", [MODEL_V4, MODEL_V4_TURBO])
    def test_is_v3_model_v4(self, model_id):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=model_id))
        assert provider._is_v3_model() is False

    def test_is_v3_model_explicit(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=MODEL_V3))
        assert provider._is_v3_model() is True

    def test_is_v3_model_v2(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider._is_v3_model() is False

    def test_expressive_tags_constant(self):
        assert "[laughs]" in EXPRESSIVE_TAGS
        assert "[whispers]" in EXPRESSIVE_TAGS
        assert "[sighs]" in EXPRESSIVE_TAGS
        assert "[slow]" in EXPRESSIVE_TAGS
        assert "[excited]" in EXPRESSIVE_TAGS
        assert len(EXPRESSIVE_TAGS) == 5


# ---------------------------------------------------------------------------
# Voice settings
# ---------------------------------------------------------------------------


class TestVoiceSettings:
    def test_v2_includes_all_settings(self):
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", style=0.3, use_speaker_boost=False)
        )
        settings = provider._build_voice_settings()
        assert settings == {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": False,
        }

    def test_expressive_includes_all_settings(self):
        """Expressive mode runs on v4 Turbo, which takes every voice setting."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", expressive=True, style=0.3, use_speaker_boost=False)
        )
        settings = provider._build_voice_settings()
        assert settings == {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": False,
        }

    def test_v3_omits_style_and_speaker_boost(self):
        """v3 model should only include stability and similarity_boost."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(
                api_key="k",
                model_id=MODEL_V3,
                style=0.5,
                use_speaker_boost=True,
            )
        )
        settings = provider._build_voice_settings()
        assert settings == {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
        assert "style" not in settings
        assert "use_speaker_boost" not in settings

    def test_v3_explicit_model_omits_style(self):
        """Setting model_id to v3 directly also omits style/speaker_boost."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=MODEL_V3))
        settings = provider._build_voice_settings()
        assert "style" not in settings
        assert "use_speaker_boost" not in settings


# ---------------------------------------------------------------------------
# Synthesize (mocked SDK)
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_synthesize_sends_correct_payload(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = MagicMock(
            side_effect=lambda **_: _sdk_audio(b"fake-audio-bytes")
        )

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="mock-settings"),
        ):
            result = await provider.synthesize("Hello world")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["model_id"] == MODEL_MULTILINGUAL_V2
        assert call_kwargs["text"] == "Hello world"
        assert call_kwargs["voice_settings"] == "mock-settings"
        assert result.transcript == "Hello world"

    async def test_synthesize_reads_the_sdk_stream_to_the_end(self):
        """``convert()`` is an async generator, never a coroutine (RMK-413)."""
        config = ElevenLabsConfig(api_key="k", output_format="pcm_16000")
        provider = ElevenLabsTTSProvider(config)
        mock_client = MagicMock()
        mock_client.text_to_speech.convert = MagicMock(
            side_effect=lambda **_: _sdk_audio(b"\x00\x01", b"\x02\x03")
        )

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            result = await provider.synthesize("Hello")

        audio = base64.b64encode(b"\x00\x01\x02\x03").decode()
        assert result.url == f"data:audio/pcm;base64,{audio}"

    async def test_synthesize_expressive_payload(self):
        """Expressive mode sends the v4 Turbo model."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = MagicMock(
            side_effect=lambda **_: _sdk_audio(b"fake-audio")
        )

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="mock-settings"),
        ):
            await provider.synthesize("[laughs] That's funny!")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["model_id"] == MODEL_V4_TURBO
        assert call_kwargs["text"] == "[laughs] That's funny!"

    async def test_synthesize_custom_voice(self):
        """Voice override is forwarded to SDK."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = MagicMock(
            side_effect=lambda **_: _sdk_audio(b"audio")
        )

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            await provider.synthesize("Hi", voice="custom-voice-id")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["voice_id"] == "custom-voice-id"


# ---------------------------------------------------------------------------
# Streaming synthesis (mocked SDK)
# ---------------------------------------------------------------------------


async def _over_http(provider: ElevenLabsTTSProvider) -> list:
    """The chunks of ``synthesize_stream`` without a context, over a mocked SDK."""
    client = MagicMock()
    client.text_to_speech.stream = lambda **_: _sdk_audio(b"\x00\x01")
    with _sdk(provider, client):
        return [chunk async for chunk in provider.synthesize_stream("Hello")]


async def _over_stitching(provider: ElevenLabsTTSProvider) -> list:
    """The chunks of ``synthesize_stream`` with a context, over a mocked raw response."""

    @contextlib.asynccontextmanager
    async def raw_stream(**_):
        yield SimpleNamespace(headers={}, data=_sdk_audio(b"\x00\x01"))

    client = MagicMock()
    client.text_to_speech.with_raw_response.stream = raw_stream
    context = TTSContext(context_id="session", turns=(), next_turn_id="turn")
    with _sdk(provider, client):
        return [chunk async for chunk in provider.synthesize_stream("Hello", context=context)]


async def _over_websocket(provider: ElevenLabsTTSProvider) -> list:
    """The chunks of ``synthesize_stream_input``, over a mocked socket."""

    async def texts():
        yield "Hello."

    provider._config.stream_input = True
    with (
        _sdk(provider, MagicMock()),
        patch(
            "roomkit.voice.tts.elevenlabs.stream_audio", lambda *_, **__: _sdk_audio(b"\x00\x01")
        ),
    ):
        return [chunk async for chunk in provider.synthesize_stream_input(texts())]


class TestDeclaredFormat:
    """Chunks and data URLs declare the codec and rate ``output_format`` asks for (RMK-413)."""

    @pytest.mark.parametrize(
        ("output_format", "rate", "chunk_format"),
        [
            ("mp3_44100_128", 44100, "mp3"),
            ("mp3_22050_32", 22050, "mp3"),
            ("pcm_8000", 8000, "pcm_s16le"),
            ("pcm_16000", 16000, "pcm_s16le"),
            ("pcm_32000", 32000, "pcm_s16le"),
            ("pcm_48000", 48000, "pcm_s16le"),
            ("ulaw_8000", 8000, "ulaw"),
            ("alaw_8000", 8000, "alaw"),
            ("opus_48000_64", 48000, "opus"),
        ],
    )
    @pytest.mark.parametrize(
        "stream", [_over_http, _over_stitching, _over_websocket], ids=["http", "stitching", "ws"]
    )
    async def test_streamed_chunks(self, stream, output_format, rate, chunk_format):
        config = ElevenLabsConfig(api_key="k", output_format=output_format)

        chunks = await stream(ElevenLabsTTSProvider(config))

        assert chunks[0].data == b"\x00\x01"
        assert {(chunk.sample_rate, chunk.format) for chunk in chunks} == {(rate, chunk_format)}

    @pytest.mark.parametrize(
        ("output_format", "mime_type"),
        [
            ("mp3_44100_128", "audio/mpeg"),
            ("pcm_16000", "audio/pcm"),
            ("wav_16000", "audio/wav"),
            ("ulaw_8000", "audio/basic"),
            ("alaw_8000", "audio/alaw"),
            ("opus_48000_64", "audio/ogg"),
            ("something_new", "audio/mpeg"),
        ],
    )
    async def test_synthesized_data_url(self, output_format, mime_type):
        config = ElevenLabsConfig(api_key="k", output_format=output_format)
        provider = ElevenLabsTTSProvider(config)
        mock_client = MagicMock()
        mock_client.text_to_speech.convert = MagicMock(
            side_effect=lambda **_: _sdk_audio(b"audio")
        )
        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            result = await provider.synthesize("Hello")

        assert result.mime_type == mime_type
        assert result.url.startswith(f"data:{mime_type};base64,")


class TestSynthesizeStream:
    async def test_synthesize_stream_yields_chunks(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", output_format="pcm_24000"))

        async def mock_stream(**kwargs):
            yield b"chunk1"
            yield b"chunk2"

        mock_client = MagicMock()
        mock_client.text_to_speech.stream = mock_stream

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            chunks = []
            async for chunk in provider.synthesize_stream("Hello"):
                chunks.append(chunk)

        # 2 data chunks + 1 final marker
        assert len(chunks) == 3
        assert chunks[0].data == b"chunk1"
        assert chunks[0].sample_rate == 24000
        assert chunks[0].format == "pcm_s16le"
        assert chunks[0].is_final is False
        assert chunks[1].data == b"chunk2"
        assert chunks[2].data == b""
        assert chunks[2].is_final is True

    async def test_synthesize_stream_skips_empty_chunks(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        async def mock_stream(**kwargs):
            yield b"data"
            yield b""
            yield b"more"

        mock_client = MagicMock()
        mock_client.text_to_speech.stream = mock_stream

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            chunks = []
            async for chunk in provider.synthesize_stream("Hi"):
                chunks.append(chunk)

        # 2 data chunks (empty skipped) + 1 final marker
        assert len(chunks) == 3
        assert chunks[0].data == b"data"
        assert chunks[1].data == b"more"
        assert chunks[2].is_final is True


# ---------------------------------------------------------------------------
# Streaming latency param
# ---------------------------------------------------------------------------


def _mock_client() -> tuple[list[dict], MagicMock]:
    """An SDK client whose stream() records the keyword arguments of each call."""
    calls: list[dict] = []

    async def mock_stream(**kwargs):
        calls.append(kwargs)
        yield b"data"

    mock_client = MagicMock()
    mock_client.text_to_speech.stream = mock_stream
    mock_client.text_to_speech.convert = MagicMock(side_effect=lambda **_: _sdk_audio(b"audio"))
    return calls, mock_client


def _sdk(provider: ElevenLabsTTSProvider, client: MagicMock):
    return patch.multiple(
        provider, _get_client=MagicMock(return_value=client), _make_voice_settings=MagicMock()
    )


class TestStreamingLatency:
    async def test_default_sends_no_latency_param(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        calls, client = _mock_client()
        with _sdk(provider, client):
            [c async for c in provider.synthesize_stream("Hi")]
        assert "optimize_streaming_latency" not in calls[0]
        assert calls[0]["output_format"] == "mp3_44100_128"

    @pytest.mark.parametrize("model_id", [MODEL_MULTILINGUAL_V2, "eleven_flash_v2_5"])
    async def test_a_set_level_reaches_a_v2_model(self, model_id):
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", model_id=model_id, optimize_streaming_latency=3)
        )
        calls, client = _mock_client()
        with _sdk(provider, client):
            [c async for c in provider.synthesize_stream("Hi")]
            await provider.synthesize("Hi")
        assert calls[0]["optimize_streaming_latency"] == 3
        assert client.text_to_speech.convert.call_args.kwargs["optimize_streaming_latency"] == 3

    @pytest.mark.parametrize("model_id", [MODEL_V4_TURBO, MODEL_V4, MODEL_V3])
    async def test_a_model_that_refuses_it_never_gets_it(self, model_id, caplog):
        with caplog.at_level("WARNING", logger="roomkit.voice.tts.elevenlabs"):
            provider = ElevenLabsTTSProvider(
                ElevenLabsConfig(api_key="k", model_id=model_id, optimize_streaming_latency=3)
            )
        assert "does not take optimize_streaming_latency" in caplog.text
        calls, client = _mock_client()
        with _sdk(provider, client):
            [c async for c in provider.synthesize_stream("Hi")]
        assert "optimize_streaming_latency" not in calls[0]


# ---------------------------------------------------------------------------
# Voice listing (mocked SDK)
# ---------------------------------------------------------------------------


class TestListVoices:
    async def test_list_voices(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_voice1 = MagicMock()
        mock_voice1.voice_id = "v1"
        mock_voice1.name = "Rachel"
        mock_voice1.category = "premade"
        mock_voice1.labels = {"accent": "american"}

        mock_voice2 = MagicMock()
        mock_voice2.voice_id = "v2"
        mock_voice2.name = "Adam"
        mock_voice2.category = "premade"
        mock_voice2.labels = {}

        mock_response = MagicMock()
        mock_response.voices = [mock_voice1, mock_voice2]

        mock_client = MagicMock()
        mock_client.voices.get_all = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            voices = await provider.list_voices()

        # VoiceInfo, like every catalog (RFC §12.2): the shared labels become
        # fields, the rest stays under attributes.
        assert [v.id for v in voices] == ["v1", "v2"]
        assert voices[0].name == "Rachel"
        assert voices[0].accent == "american"
        assert voices[0].attributes == {"category": "premade"}
        assert voices[1].accent is None

    async def test_list_voices_applies_the_filters(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        french = MagicMock(voice_id="v1", category="cloned", description="Voix posée")
        french.name = "Chloé"
        french.labels = {"language": "fr", "gender": "female", "age": "young"}
        english = MagicMock(voice_id="v2", category="premade", description=None)
        english.name = "Adam"
        english.labels = {"language": "en", "gender": "male"}
        mock_client = MagicMock()
        mock_client.voices.get_all = AsyncMock(return_value=MagicMock(voices=[french, english]))

        with patch.object(provider, "_get_client", return_value=mock_client):
            found = await provider.list_voices(language="fr", gender="female", query="posée")

        assert [v.id for v in found] == ["v1"]
        assert found[0].attributes == {"age": "young", "category": "cloned"}

    def test_available_voices_is_the_curated_catalog(self):
        from roomkit.providers.elevenlabs.voices import VOICES

        assert ElevenLabsTTSProvider.available_voices() == VOICES

    async def test_list_voices_caches(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_voice = MagicMock()
        mock_voice.voice_id = "v1"
        mock_voice.name = "R"
        mock_voice.category = "premade"
        mock_voice.labels = {}

        mock_response = MagicMock()
        mock_response.voices = [mock_voice]

        mock_client = MagicMock()
        mock_client.voices.get_all = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            await provider.list_voices()
            await provider.list_voices()

        # Should only call the API once
        assert mock_client.voices.get_all.call_count == 1


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_clears_client(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        provider._client = MagicMock()  # Simulate existing client

        await provider.close()

        assert provider._client is None

    async def test_close_noop_when_no_client(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        await provider.close()  # should not raise
        assert provider._client is None
