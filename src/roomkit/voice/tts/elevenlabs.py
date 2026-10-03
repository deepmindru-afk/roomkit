"""ElevenLabs text-to-speech provider.

Supports expressive mode via the ``eleven_v4_turbo`` model, streaming text
input over WebSocket (opt-in), so a streaming AI response is spoken from its
first sentence, and request stitching from the TTS conversation context (RFC
§12.2.2): a response synthesized over HTTP continues the voice of the previous
ones.
When ``expressive=True``, synthesis uses Eleven v4 Turbo, which renders
audio tags such as ``[laughs]``, ``[whispers]``, ``[sighs]``, ``[pause]``
and ``[excited]`` embedded in the text, alone or stacked.

.. note::

    Do **not** combine expressive mode with :class:`StripBrackets` — that
    filter removes all ``[...]`` content, including the expressive tags.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from roomkit.providers.elevenlabs.voices import VOICES as ELEVENLABS_VOICES
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts._elevenlabs_stitching import RequestIdLedger, request_id_of
from roomkit.voice.tts._elevenlabs_ws import DialogueSocket, socket_for, stream_audio
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.context import TTSContextLevel
from roomkit.voice.voices import VoiceInfo, filter_voices

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger(__name__)

# Model ID constants
MODEL_MULTILINGUAL_V2 = "eleven_multilingual_v2"
MODEL_TURBO_V2_5 = "eleven_turbo_v2_5"  # deprecated by ElevenLabs for MODEL_FLASH_V2_5
MODEL_FLASH_V2_5 = "eleven_flash_v2_5"
MODEL_V3 = "eleven_v3"
MODEL_V4 = "eleven_v4"
MODEL_V4_TURBO = "eleven_v4_turbo"

# The model families that render inline audio tags: ``expressive=True``
# keeps a ``model_id`` of one of them and picks v4 Turbo otherwise.
_AUDIO_TAG_MODELS = ("eleven_v3", "eleven_v4")

# The model families that take ``optimize_streaming_latency`` (the ``_v2_5``
# variants included); v3 and v4 answer 400 to it.
_LATENCY_MODELS = ("eleven_multilingual_v2", "eleven_flash_v2", "eleven_turbo_v2")

# The MIME type and the AudioChunk format of each output codec, the part of
# an ``output_format`` before its sample rate (``pcm_16000``, ``mp3_44100_128``).
# ElevenLabs refuses ``wav`` on a streamed request, so only synthesize() sees it.
_CODECS: dict[str, tuple[str, str]] = {
    "mp3": ("audio/mpeg", "mp3"),
    "pcm": ("audio/pcm", "pcm_s16le"),
    "wav": ("audio/wav", "wav"),
    "ulaw": ("audio/basic", "ulaw"),
    "alaw": ("audio/alaw", "alaw"),
    "opus": ("audio/ogg", "opus"),
}

# Expressive tags recognised by v3 Conversational TTS. Eleven v4 documents a
# wider set (``[pause]``, ``[long pause]``, sound effects) and lets tags stack.
EXPRESSIVE_TAGS = frozenset({"[laughs]", "[whispers]", "[sighs]", "[slow]", "[excited]"})


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs TTS provider.

    Set ``expressive=True`` to enable expressive mode: synthesis uses Eleven
    v4 Turbo (``eleven_v4_turbo``), which renders inline audio tags at
    conversational latency. A ``model_id`` that already renders them
    (``eleven_v4``, ``eleven_v3``) is kept. v3 models do not take ``style``
    or ``use_speaker_boost``; both are left out of their requests.
    """

    api_key: str = field(repr=False)
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)
    model_id: str = MODEL_MULTILINGUAL_V2
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    # <codec>_<rate>[_<bitrate>]: mp3_44100_128, pcm_16000, ulaw_8000, opus_48000_64...
    # wav_* only for synthesize(): ElevenLabs refuses it on a streamed request.
    output_format: str = "mp3_44100_128"
    # Latency optimization, 0-4 (higher = lower latency, at some cost of
    # quality). None sends nothing. Only the v2 / v2.5 models take it, and
    # ElevenLabs deprecates it.
    optimize_streaming_latency: int | None = None
    # Expressive mode — Eleven v4 Turbo with inline audio tags
    expressive: bool = False
    # Request stitching from the conversation context: the provider receives
    # its own previous turns (SELF) and sends their request ids, or their text.
    # Not available on v3 models, nor on the streaming-input socket.
    use_context: bool = True
    # Streaming text input over WebSocket: a streaming AI response is spoken
    # from its first sentence instead of once it is complete. v4 / v4 Turbo
    # use the Text to Dialogue socket, which applies no voice settings; the
    # v2 / v2.5 models use the Text to Speech socket; v3 has none. Opt-in: on
    # that path the Voice Channel runs no BEFORE_TTS hook, a TTS failure ends
    # the AI response where it failed, and nothing is stitched across responses.
    stream_input: bool = False


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs text-to-speech provider with streaming support.

    When *expressive mode* is enabled (``config.expressive=True``), the
    provider uses the ``eleven_v4_turbo`` model which supports
    inline audio tags (``[laughs]``, ``[whispers]``, etc.) and adapts
    tone and timing based on conversational context.
    """

    def __init__(self, config: ElevenLabsConfig) -> None:
        self._config = config
        if config.expressive and not config.model_id.startswith(_AUDIO_TAG_MODELS):
            self._config.model_id = MODEL_V4_TURBO
        if config.optimize_streaming_latency is not None and not self._takes_latency_param():
            logger.warning(
                "ElevenLabs model %s does not take optimize_streaming_latency; it is not sent",
                config.model_id,
            )
        if self._dialogue_socket_drops_voice_settings():
            logger.warning(
                "ElevenLabs model %s streams input over the Text to Dialogue socket, "
                "which applies no voice settings; set stream_input=False to keep them",
                config.model_id,
            )
        self._client: Any = None  # AsyncElevenLabs (lazy)
        self._voices_cache: list[VoiceInfo] | None = None
        self._request_ids = RequestIdLedger()

    @property
    def name(self) -> str:
        return "ElevenLabsTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice_id

    @property
    def supports_streaming_input(self) -> bool:
        """True when ``stream_input`` is on and the model has a socket (v3 has none)."""
        return self._config.stream_input and socket_for(self._config.model_id) is not None

    @property
    def context_level(self) -> TTSContextLevel:
        """SELF: its own previous turns drive request stitching.

        NONE when ``use_context`` is off, and on v3 models, which ElevenLabs
        does not stitch.
        """
        if not self._config.use_context or self._is_v3_model():
            return TTSContextLevel.NONE
        return TTSContextLevel.SELF

    def release_context(self, context_id: str) -> None:
        self._request_ids.forget(context_id)

    def _is_v3_model(self) -> bool:
        """Return True when the selected model is a v3 variant."""
        return "v3" in self._config.model_id

    def _takes_latency_param(self) -> bool:
        return self._config.model_id.startswith(_LATENCY_MODELS)

    def _dialogue_socket_drops_voice_settings(self) -> bool:
        """Whether voice settings set away from their defaults would be lost on
        the Text to Dialogue socket."""
        if not self.supports_streaming_input:
            return False
        if not isinstance(socket_for(self._config.model_id), DialogueSocket):
            return False
        defaults = {f.name: f.default for f in fields(ElevenLabsConfig)}
        return any(
            getattr(self._config, name) != defaults[name]
            for name in ("stability", "similarity_boost", "style", "use_speaker_boost")
        )

    def _query_params(self) -> dict[str, Any]:
        """The query arguments of a synthesis request: the output format, and
        the latency level when one is set and the model takes it."""
        params: dict[str, Any] = {"output_format": self._config.output_format}
        if self._config.optimize_streaming_latency is not None and self._takes_latency_param():
            params["optimize_streaming_latency"] = self._config.optimize_streaming_latency
        return params

    def _get_client(self) -> Any:
        if self._client is None:
            from elevenlabs.client import AsyncElevenLabs

            self._client = AsyncElevenLabs(api_key=self._config.api_key)
        return self._client

    def _build_voice_settings(self) -> dict[str, float | bool]:
        """Build voice settings for synthesis.

        v3 Conversational only supports ``stability`` and
        ``similarity_boost``; ``style`` and ``use_speaker_boost`` are
        omitted when a v3 model is active.
        """
        settings: dict[str, float | bool] = {
            "stability": self._config.stability,
            "similarity_boost": self._config.similarity_boost,
        }
        if not self._is_v3_model():
            settings["style"] = self._config.style
            settings["use_speaker_boost"] = self._config.use_speaker_boost
        return settings

    def _make_voice_settings(self) -> Any:
        """Build an SDK ``VoiceSettings`` object for API calls."""
        from elevenlabs import VoiceSettings

        return VoiceSettings(**self._build_voice_settings())

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """The curated ElevenLabs default voices, shared with the realtime provider."""
        return list(ELEVENLABS_VOICES)

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """Voices the account exposes, its own cloned voices included.

        The account's list is fetched once and kept; the filters apply to it
        (RFC §12.2).
        """
        if self._voices_cache is None:
            response = await self._get_client().voices.get_all()
            self._voices_cache = [_voice_info(voice) for voice in response.voices]
        return filter_voices(self._voices_cache, language=language, gender=gender, query=query)

    async def _convert(self, voice_id: str, text: str) -> bytes:
        """The whole audio of *text*, read from the SDK's ``convert`` stream.

        ``convert()`` is an async generator: awaiting it raises ``TypeError``.
        """
        audio = self._get_client().text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self._config.model_id,
            voice_settings=self._make_voice_settings(),
            **self._query_params(),
        )
        return b"".join([chunk async for chunk in audio])

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice ID (uses default_voice if not specified).

        Returns:
            AudioContent with URL to generated audio.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        voice_id = voice or self._config.voice_id

        t0 = time.monotonic()
        audio_bytes = await self._convert(voice_id, text)

        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.tts.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": "elevenlabs", "model": self._config.model_id},
        )

        import base64

        mime_type = self._get_mime_type()
        data_url = f"data:{mime_type};base64,{base64.b64encode(audio_bytes).decode()}"

        # Estimate duration (rough: ~150 words/minute, ~5 chars/word)
        words = len(text.split())
        duration = words / 150 * 60  # seconds

        return AudioContentModel(
            url=data_url,
            mime_type=mime_type,
            transcript=text,
            duration_seconds=duration,
        )

    async def synthesize_stream_input(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio from a stream of text chunks, over one WebSocket.

        Each chunk (a sentence, as the Voice Channel sends them) is spoken as
        soon as it arrives. ``context`` is not used: neither socket stitches a
        response to the previous ones.

        Args:
            text_stream: Async iterator yielding text chunks.
            voice: Voice ID (uses default_voice if not specified).
            context: The session's previous turns; unused on this path.

        Yields:
            AudioChunk with raw audio data, then a final empty chunk.
        """
        socket = socket_for(self._config.model_id) if self._config.stream_input else None
        if socket is None:
            raise NotImplementedError(
                f"ElevenLabs model {self._config.model_id} does not stream input here."
            )
        voice_id = voice or self._config.voice_id
        audio = stream_audio(
            socket,
            api_key=self._config.api_key,
            voice_id=voice_id,
            query={"model_id": self._config.model_id, **self._query_params()},
            voice_settings=self._build_voice_settings(),
            text_stream=text_stream,
        )
        try:
            async for data in audio:
                yield self._chunk(data)
        finally:
            await audio.aclose()
        yield AudioChunk(
            data=b"",
            sample_rate=self._get_sample_rate(),
            format=self._get_audio_format(),
            is_final=True,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks as they're generated.

        Uses the ElevenLabs SDK streaming API for low-latency synthesis. With
        a ``context``, the request carries the stitching arguments of the
        session's previous turns, and its own ``request-id`` is kept for the
        next turns once the audio has been read to the end.

        Args:
            text: Text to synthesize.
            voice: Voice ID (uses default_voice if not specified).
            context: The session's previous turns (passed at SELF level).

        Yields:
            AudioChunk with raw audio data.
        """
        voice_id = voice or self._config.voice_id
        client = self._get_client()
        request = {
            "voice_id": voice_id,
            "text": text,
            "model_id": self._config.model_id,
            "voice_settings": self._make_voice_settings(),
            **self._query_params(),
        }

        if context is None or self.context_level == TTSContextLevel.NONE:
            async for chunk in client.text_to_speech.stream(**request):
                if chunk:
                    yield self._chunk(chunk)
        else:
            stitching = self._request_ids.stitching_params(context, voice_id)
            logger.debug(
                "ElevenLabs stitching for %s: %s",
                context.context_id,
                {k: len(v) if isinstance(v, list) else "text" for k, v in stitching.items()},
            )
            async with client.text_to_speech.with_raw_response.stream(
                **request, **stitching
            ) as response:
                request_id = request_id_of(response.headers)
                async for chunk in response.data:
                    if chunk:
                        yield self._chunk(chunk)
            # Reached only when the audio was read to the end, the one case
            # ElevenLabs can continue from; a stream closed early keeps no id.
            if request_id:
                logger.debug("ElevenLabs request %s read to the end", request_id)
                self._request_ids.record(
                    context.context_id, context.next_turn_id, request_id, voice_id
                )

        # Send final chunk marker
        yield AudioChunk(
            data=b"",
            sample_rate=self._get_sample_rate(),
            format=self._get_audio_format(),
            is_final=True,
        )

    def _chunk(self, data: bytes) -> AudioChunk:
        return AudioChunk(
            data=data,
            sample_rate=self._get_sample_rate(),
            format=self._get_audio_format(),
            is_final=False,
        )

    def _get_mime_type(self) -> str:
        """Get MIME type from output format."""
        return self._codec()[0]

    def _get_sample_rate(self) -> int:
        """The sample rate ``output_format`` names (``pcm_8000`` is 8 kHz), 44.1 kHz if none."""
        parts = self._config.output_format.split("_")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 44100

    def _get_audio_format(self) -> str:
        """Get audio format string."""
        return self._codec()[1]

    def _codec(self) -> tuple[str, str]:
        """The MIME type and chunk format of ``output_format``'s codec, MP3's if unknown."""
        codec = self._config.output_format.split("_")[0]
        return _CODECS.get(codec, _CODECS["mp3"])

    async def close(self) -> None:  # noqa: B027
        """Release resources."""
        self._client = None


def _voice_info(voice: Any) -> VoiceInfo:
    """An ElevenLabs voice as a :class:`VoiceInfo`: the labels it shares with
    every catalog become fields, the rest stays under ``attributes``."""
    labels = dict(voice.labels) if isinstance(getattr(voice, "labels", None), dict) else {}
    attributes = {str(k): str(v) for k, v in labels.items() if v is not None}
    category = getattr(voice, "category", None)
    if isinstance(category, str) and category:
        attributes["category"] = category
    description = getattr(voice, "description", None)
    return VoiceInfo(
        id=voice.voice_id,
        name=voice.name or None,
        language=attributes.pop("language", None),
        gender=attributes.pop("gender", None),
        accent=attributes.pop("accent", None),
        description=(description if isinstance(description, str) and description else None)
        or attributes.pop("description", None),
        # Last: the fields above take their labels out of it first.
        attributes=attributes,
    )
