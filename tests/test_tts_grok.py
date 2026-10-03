"""Tests for GrokTTSProvider (xAI TTS)."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.grok import _CODEC_META, GROK_VOICES, GrokTTSConfig, GrokTTSProvider

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestGrokTTSConfig:
    def test_defaults(self):
        cfg = GrokTTSConfig(api_key="xai-test")
        assert cfg.voice_id == "eve"
        assert cfg.language == "en"
        assert cfg.codec == "pcm"
        assert cfg.sample_rate == 24000
        assert cfg.bit_rate == 128000
        assert cfg.base_url == "https://api.x.ai/v1"
        assert cfg.ws_url == "wss://api.x.ai/v1/tts"

    def test_custom_values(self):
        cfg = GrokTTSConfig(
            api_key="xai-test",
            voice_id="rex",
            language="fr",
            codec="mp3",
            sample_rate=44100,
            bit_rate=192000,
        )
        assert cfg.voice_id == "rex"
        assert cfg.language == "fr"
        assert cfg.codec == "mp3"
        assert cfg.sample_rate == 44100
        assert cfg.bit_rate == 192000


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------


class TestProviderBasics:
    def test_name(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))
        assert provider.name == "GrokTTS"

    def test_default_voice(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test", voice_id="sal"))
        assert provider.default_voice == "sal"

    def test_supports_streaming_input(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))
        assert provider.supports_streaming_input is True

    def test_voices_constant(self):
        assert set(GROK_VOICES) == {"eve", "ara", "rex", "sal", "leo"}


# ---------------------------------------------------------------------------
# Codec metadata
# ---------------------------------------------------------------------------


class TestCodecMeta:
    @pytest.mark.parametrize(
        ("codec", "expected_mime", "expected_fmt"),
        [
            ("pcm", "audio/pcm", "pcm_s16le"),
            ("wav", "audio/wav", "pcm_s16le"),
            ("mp3", "audio/mpeg", "mp3"),
            ("mulaw", "audio/basic", "mulaw"),
            ("alaw", "audio/alaw", "alaw"),
        ],
    )
    def test_codec_meta(self, codec: str, expected_mime: str, expected_fmt: str):
        mime, fmt = _CODEC_META[codec]
        assert mime == expected_mime
        assert fmt == expected_fmt


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


class TestBuildRequestBody:
    def test_pcm_body(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))
        body = provider._build_request_body("Hello", None, codec="pcm")
        assert body == {
            "text": "Hello",
            "voice_id": "eve",
            "language": "en",
            "output_format": {"codec": "pcm", "sample_rate": 24000},
        }

    def test_mp3_body_includes_bit_rate(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test", codec="mp3"))
        body = provider._build_request_body("Hi", None, codec="mp3")
        assert body["output_format"]["bit_rate"] == 128000

    def test_voice_override(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))
        body = provider._build_request_body("Hi", "rex", codec="pcm")
        assert body["voice_id"] == "rex"


# ---------------------------------------------------------------------------
# Synthesize (REST)
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_synthesize_returns_audio_content(self):
        fake_audio = b"\x00\x01" * 1200  # 1200 samples = 2400 bytes
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = fake_audio
        mock_response.raise_for_status = MagicMock()

        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.synthesize("Hello world")

        assert result.type == "audio"
        assert result.url.startswith("data:audio/pcm;base64,")
        assert result.mime_type == "audio/pcm"
        assert result.transcript == "Hello world"
        # 2400 bytes / 2 bytes-per-sample / 24000 Hz = 0.05s
        assert result.duration_seconds == pytest.approx(0.05, abs=0.001)

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/tts"

    async def test_synthesize_mp3_no_duration(self):
        """MP3 codec cannot compute duration from byte count."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b"\xff\xfb\x90" * 100
        mock_response.raise_for_status = MagicMock()

        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test", codec="mp3"))
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.synthesize("Test")
        assert result.duration_seconds is None
        assert result.mime_type == "audio/mpeg"


# ---------------------------------------------------------------------------
# Synthesize stream (HTTP chunked)
# ---------------------------------------------------------------------------


class TestSynthesizeStream:
    async def test_stream_yields_chunks_and_final(self):
        chunks_data = [b"\x00" * 4096, b"\x01" * 2048]

        async def fake_aiter_bytes(chunk_size: int = 4096):
            for c in chunks_data:
                yield c

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = fake_aiter_bytes

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Make stream() return an async context manager yielding our mock response.
        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_cm)

        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))
        provider._client = mock_client

        result: list[AudioChunk] = []
        async for chunk in provider.synthesize_stream("Hello"):
            result.append(chunk)

        # 2 data chunks + 1 final marker
        assert len(result) == 3
        assert result[0].data == chunks_data[0]
        assert result[0].is_final is False
        assert result[0].sample_rate == 24000
        assert result[0].format == "pcm_s16le"
        assert result[1].data == chunks_data[1]
        assert result[1].is_final is False
        assert result[2].data == b""
        assert result[2].is_final is True


# ---------------------------------------------------------------------------
# Synthesize stream input (WebSocket)
# ---------------------------------------------------------------------------


class TestSynthesizeStreamInput:
    async def test_ws_stream_input(self):
        audio_bytes = b"\x00\x01" * 100

        ws_messages = [
            json.dumps({"type": "audio.delta", "delta": base64.b64encode(audio_bytes).decode()}),
            json.dumps({"type": "audio.done", "trace_id": "test-123"}),
        ]

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        async def ws_aiter():
            for msg in ws_messages:
                yield msg

        mock_ws.__aiter__ = lambda self: ws_aiter()

        async def text_gen():
            yield "Hello "
            yield "world"

        mock_websockets = MagicMock()
        ws_cm = AsyncMock()
        ws_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        ws_cm.__aexit__ = AsyncMock(return_value=False)
        mock_websockets.connect = MagicMock(return_value=ws_cm)
        mock_websockets.exceptions = MagicMock()

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))

            result: list[AudioChunk] = []
            async for chunk in provider.synthesize_stream_input(text_gen()):
                result.append(chunk)

        assert len(result) == 2
        assert result[0].data == audio_bytes
        assert result[0].is_final is False
        assert result[0].sample_rate == 24000
        assert result[1].data == b""
        assert result[1].is_final is True


# ---------------------------------------------------------------------------
# Streamed WAV (RMK-413)
# ---------------------------------------------------------------------------


async def _stream_over_http(provider: GrokTTSProvider) -> tuple[str, list[AudioChunk]]:
    """Run ``synthesize_stream`` on a mocked client: the codec asked for, the chunks."""

    async def aiter_bytes(chunk_size: int = 4096):
        yield b"\x00\x01" * 8

    response = MagicMock()
    response.status_code = 200
    response.aiter_bytes = aiter_bytes
    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(return_value=stream_cm)
    provider._client = client
    chunks = [chunk async for chunk in provider.synthesize_stream("Hello")]
    return client.stream.call_args.kwargs["json"]["output_format"]["codec"], chunks


async def _stream_over_ws(provider: GrokTTSProvider) -> tuple[str, list[AudioChunk]]:
    """Run ``synthesize_stream_input`` on a mocked socket: the codec asked for, the chunks."""
    audio = base64.b64encode(b"\x00\x01" * 8).decode()
    messages = [
        json.dumps({"type": "audio.delta", "delta": audio}),
        json.dumps({"type": "audio.done", "trace_id": "t"}),
    ]

    async def ws_aiter():
        for message in messages:
            yield message

    async def text():
        yield "Hello"

    ws = AsyncMock()
    ws.__aiter__ = lambda self: ws_aiter()
    ws_cm = AsyncMock()
    ws_cm.__aenter__ = AsyncMock(return_value=ws)
    ws_cm.__aexit__ = AsyncMock(return_value=False)
    websockets = MagicMock()
    websockets.connect = MagicMock(return_value=ws_cm)
    websockets.exceptions = MagicMock()
    with patch.dict("sys.modules", {"websockets": websockets}):
        chunks = [chunk async for chunk in provider.synthesize_stream_input(text())]
    query = parse_qs(urlsplit(websockets.connect.call_args.args[0]).query)
    return query["codec"][0], chunks


_STREAMS = pytest.mark.parametrize(
    "stream", [_stream_over_http, _stream_over_ws], ids=["http", "websocket"]
)


class TestStreamedWav:
    """A WAV stream opens with a RIFF header the chunks would play as audio."""

    @_STREAMS
    async def test_a_wav_config_streams_raw_pcm(self, stream):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test", codec="wav"))

        codec, chunks = await stream(provider)

        assert codec == "pcm"
        assert {chunk.format for chunk in chunks} == {"pcm_s16le"}

    @_STREAMS
    @pytest.mark.parametrize(
        ("configured", "declared"),
        [("pcm", "pcm_s16le"), ("mp3", "mp3"), ("mulaw", "mulaw"), ("alaw", "alaw")],
    )
    async def test_other_codecs_stream_as_configured(self, stream, configured, declared):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test", codec=configured))

        codec, chunks = await stream(provider)

        assert codec == configured
        assert {chunk.format for chunk in chunks} == {declared}

    async def test_synthesize_keeps_wav(self):
        response = MagicMock(spec=httpx.Response)
        response.content = b"RIFF" + b"\x00" * 40 + b"\x00\x01" * 100
        response.raise_for_status = MagicMock()
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=response)
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test", codec="wav"))
        provider._client = client

        result = await provider.synthesize("Hello")

        assert client.post.call_args.kwargs["json"]["output_format"]["codec"] == "wav"
        assert result.mime_type == "audio/wav"


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_releases_client(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        provider._client = mock_client

        await provider.close()

        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    async def test_close_noop_without_client(self):
        provider = GrokTTSProvider(GrokTTSConfig(api_key="xai-test"))
        await provider.close()  # Should not raise


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


class TestLazyLoaders:
    def test_get_grok_tts_provider(self):
        from roomkit.voice import get_grok_tts_provider

        cls = get_grok_tts_provider()
        assert cls.__name__ == "GrokTTSProvider"

    def test_get_grok_tts_config(self):
        from roomkit.voice import get_grok_tts_config

        cls = get_grok_tts_config()
        assert cls.__name__ == "GrokTTSConfig"
