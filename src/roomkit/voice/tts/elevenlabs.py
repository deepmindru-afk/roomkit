"""ElevenLabs text-to-speech provider.

Supports expressive mode via the ``eleven_v4_turbo`` model, and request
stitching from the TTS conversation context (RFC §12.2.2): each response
continues the voice of the previous ones.
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.providers.elevenlabs.voices import VOICES as ELEVENLABS_VOICES
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts._elevenlabs_stitching import RequestIdLedger, request_id_of
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
    output_format: str = "mp3_44100_128"  # mp3, pcm_16000, pcm_22050, etc.
    # Latency optimization, 0-4 (higher = lower latency, at some cost of
    # quality). None sends nothing. Only the v2 / v2.5 models take it, and
    # ElevenLabs deprecates it.
    optimize_streaming_latency: int | None = None
    # Expressive mode — Eleven v4 Turbo with inline audio tags
    expressive: bool = False
    # Request stitching from the conversation context: the provider receives
    # its own previous turns (SELF) and sends their request ids, or their text.
    # Not available on v3 models.
    use_context: bool = True


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
        # The official SDK does not expose WebSocket input streaming.
        return False

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
        client = self._get_client()

        t0 = time.monotonic()
        response = await client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self._config.model_id,
            voice_settings=self._make_voice_settings(),
            **self._query_params(),
        )

        # SDK convert() may return bytes or an async iterator — normalise.
        if isinstance(response, bytes):
            audio_bytes = response
        else:
            parts: list[bytes] = []
            async for chunk in response:
                parts.append(chunk)
            audio_bytes = b"".join(parts)

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
        fmt = self._config.output_format
        if fmt.startswith("mp3"):
            return "audio/mpeg"
        elif fmt.startswith("pcm"):
            return "audio/pcm"
        elif fmt.startswith("ulaw"):
            return "audio/basic"
        return "audio/mpeg"

    def _get_sample_rate(self) -> int:
        """Get sample rate from output format."""
        fmt = self._config.output_format
        if "44100" in fmt:
            return 44100
        elif "24000" in fmt:
            return 24000
        elif "22050" in fmt:
            return 22050
        elif "16000" in fmt:
            return 16000
        return 44100

    def _get_audio_format(self) -> str:
        """Get audio format string."""
        fmt = self._config.output_format
        if fmt.startswith("mp3"):
            return "mp3"
        elif fmt.startswith("pcm"):
            return "pcm_s16le"
        elif fmt.startswith("ulaw"):
            return "ulaw"
        return "mp3"

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
