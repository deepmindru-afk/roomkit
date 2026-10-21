"""Mistral Voxtral speech-to-text provider (Voxtral Transcribe 3).

Voxtral 3 transcribes two ways, and this provider speaks both:

* :meth:`VoxtralSTTProvider.transcribe` — ``POST /v1/audio/transcriptions``
  for one finished clip, on Voxtral Mini Transcribe 3 or, more accurate,
  Voxtral Small Transcribe 3.
* :meth:`VoxtralSTTProvider.transcribe_stream` — the realtime WebSocket, on
  Voxtral Mini Transcribe Realtime 3: PCM in, text out while the person
  speaks, and the model's own turn detection.

How a stream ends depends on how the :class:`~roomkit.channels.VoiceChannel`
is set up, which the provider cannot see, so it is configuration:

* ``turn_detection`` set (default) — no pipeline VAD. The channel streams all
  audio (continuous mode) and the model decides where a turn ends; each turn
  is a final, and its start a speech start. Its words, as partials, can cut
  the agent off.
* ``turn_detection=None`` — a pipeline VAD delimits utterances. The channel
  opens one stream per utterance, and the provider answers one final once
  the stream ends.

The wrong choice loses words: behind a VAD with turn detection on, the
channel keeps the last turn of an utterance only; without a VAD and with it
off, no final comes until the audio stops.

Each stream is one session on the service. Voxtral 3 is in public preview
(October 2026). Install with ``pip install roomkit[mistral-voice]``.
"""

from __future__ import annotations

import contextlib
import logging
import mimetypes
import wave
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from roomkit.providers.utils import http_timeout, parse_data_uri
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.voxtral_protocol import (
    REALTIME_PATH,
    STREAMING_DELAYS_MS,
    http_error,
    language_code,
    rest_result,
    valid_streaming_delay,
)
from roomkit.voice.stt.voxtral_protocol import VoxtralSTTError as VoxtralSTTError
from roomkit.voice.stt.voxtral_realtime import RealtimeSession, SessionSetup, first_audio

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger("roomkit.voice.stt.voxtral")

DEFAULT_MODEL = "voxtral-mini-transcribe-3"
DEFAULT_REALTIME_MODEL = "voxtral-mini-transcribe-realtime-3"

_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})
"""Input rates sent as they are: the rates Mistral documents for the realtime
model. Any other is resampled to 16 kHz, the service's default."""
_DEFAULT_RATE = 16000


_LATE_TEXT_MARGIN_MS = 100
"""Added to the streaming delay to give the final wait. A turn's last words
followed its ``turn.end`` by up to 0.19 s at a 480 ms delay, 0.43 s at 480 ms
with Mistral's aggressive preset, 0.16 s at 240 ms and 0.75 s at 1200 ms
(measured 2026-10-21): never more than the delay."""


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for VoxtralSTTProvider.transcribe(). "
            "Install with: pip install roomkit[mistral-voice]"
        ) from exc
    return httpx


@dataclass(frozen=True)
class VoxtralTurnDetection:
    """When the realtime model says a turn starts and ends.

    The model scores, from 0 to 1, how likely the person is still speaking;
    the thresholds turn that score into turn events. The defaults are the
    service's own (an empty ``turn_detection`` object), and
    :meth:`aggressive` is Mistral's faster preset.

    Attributes:
        start_threshold: Score above which a turn starts or resumes. Raise it
            when noise starts turns.
        eager_end_threshold: Score below which the person has probably
            finished (an eager end, which RoomKit does not act on).
        end_threshold: Score below which the turn ends. Lower it to cut
            people off less often, at the cost of a later answer.
        min_silence_ms: Shortest pause before a turn can end.
        end_timeout_ms: Longest pause before a turn ends anyway.
    """

    start_threshold: float = 0.5
    eager_end_threshold: float = 0.5
    end_threshold: float = 0.33
    min_silence_ms: int = 500
    end_timeout_ms: int = 3000

    def __post_init__(self) -> None:
        if not 0.0 <= self.end_threshold < self.eager_end_threshold <= self.start_threshold <= 1:
            raise ValueError(
                "turn detection thresholds must keep their order, "
                "1 >= start_threshold >= eager_end_threshold > end_threshold >= 0"
            )
        if self.min_silence_ms < 0 or self.end_timeout_ms < self.min_silence_ms:
            raise ValueError("end_timeout_ms must be at least min_silence_ms, itself at least 0")

    @classmethod
    def aggressive(cls) -> VoxtralTurnDetection:
        """Mistral's preset for fast exchanges: turns end sooner, and cut more often."""
        return cls(eager_end_threshold=0.4, min_silence_ms=300, end_timeout_ms=2500)


@dataclass
class VoxtralSTTConfig:
    """Configuration for :class:`VoxtralSTTProvider`.

    Attributes:
        api_key: Mistral API key (sent as ``Bearer``).
        model: File transcription model: ``voxtral-mini-transcribe-3``, or
            ``voxtral-small-transcribe-3`` for the most accurate.
        realtime_model: Realtime transcription model.
        language: Language to transcribe, as a BCP-47 tag or an ISO 639-1
            code (``"fr-CA"`` is sent as ``"fr"``). ``None`` lets the model
            detect it.
        context_bias: Names and terms the model should spell right (a
            product, a person, a place), phrases included.
        streaming_delay_ms: How long the realtime model listens past a word
            before writing it: shorter is sooner, longer is more accurate.
            Any value from 80 to 1200, or 2400.
        turn_detection: The realtime model's turn detection, for a channel
            without a pipeline VAD; ``None`` behind one, where a turn ended
            inside an utterance would otherwise be lost (see the module
            docstring).
        final_wait_ms: How long a turn's final waits after the model ends the
            turn, for its last words, which the text brings up to a streaming
            delay later. ``None`` is the delay plus 100 ms (0.58 s with the
            defaults). Words later than it are dropped, with a warning.
        base_url: API root, without a version.
        realtime_url: Realtime WebSocket endpoint; ``None`` derives it from
            ``base_url``.
        timeout: Read budget of a file transcription, in seconds.
        connect_timeout: TCP connect budget, in seconds, apart from ``timeout``.
        handshake_timeout_s: How long a stream waits, in all, for the
            connection and for the session to accept its settings.
        final_timeout_s: How long a stream waits, once its audio has ended,
            for the end of the transcript.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    realtime_model: str = DEFAULT_REALTIME_MODEL
    language: str | None = None
    context_bias: list[str] = field(default_factory=list)
    streaming_delay_ms: int = 480
    turn_detection: VoxtralTurnDetection | None = field(default_factory=VoxtralTurnDetection)
    final_wait_ms: int | None = None
    base_url: str = "https://api.mistral.ai"
    realtime_url: str | None = None
    timeout: float = 60.0
    connect_timeout: float = 5.0
    handshake_timeout_s: float = 10.0
    final_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("VoxtralSTTConfig.api_key is required")
        if not self.api_key.isascii():
            # Refused here, not as a UnicodeEncodeError from the HTTP client.
            raise ValueError(
                "VoxtralSTTConfig.api_key holds a non-ASCII character (a placeholder such as "
                "'…' copied as is?): an HTTP Authorization header cannot carry it"
            )
        language_code(self.language)
        if not valid_streaming_delay(self.streaming_delay_ms):
            raise ValueError(
                f"streaming_delay_ms must be {STREAMING_DELAYS_MS}, got {self.streaming_delay_ms}"
            )
        if self.final_wait_ms is not None and self.final_wait_ms < 0:
            raise ValueError("final_wait_ms must not be negative")
        if self.handshake_timeout_s <= 0 or self.final_timeout_s <= 0:
            raise ValueError("VoxtralSTTConfig timeouts must be positive")

    @property
    def final_wait_s(self) -> float:
        """The wait for a turn's last words after its end, in seconds."""
        if self.final_wait_ms is not None:
            return self.final_wait_ms / 1000
        return (self.streaming_delay_ms + _LATE_TEXT_MARGIN_MS) / 1000

    @property
    def stream_url(self) -> str:
        """The realtime endpoint, the model in its query string."""
        base = self.realtime_url or self.base_url.rstrip("/") + REALTIME_PATH
        parts = urlsplit(base)
        scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
        query = urlencode([*parse_qsl(parts.query), ("model", self.realtime_model)])
        return urlunsplit((scheme, parts.netloc, parts.path, query, ""))


class VoxtralSTTProvider(STTProvider):
    """Speech-to-text on Mistral's Voxtral Transcribe 3, file and realtime."""

    def __init__(self, config: VoxtralSTTConfig) -> None:
        self._config = config
        self._resampler = LinearResamplerProvider()
        self._http: Any = None

    @property
    def name(self) -> str:
        return "VoxtralSTT"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_language_override(self) -> bool:
        return True

    def _supported_pcm(
        self, data: bytes, sample_rate: int, channels: int, rate: int, sample_width: int = 2
    ) -> bytes:
        """``data`` as mono 16-bit PCM at ``rate``, converted when it is not."""
        if sample_rate == rate and channels == 1 and sample_width == 2:
            return data
        frame = AudioFrame(
            data=data, sample_rate=sample_rate, channels=channels, sample_width=sample_width
        )
        return self._resampler.resample(frame, rate, 1, 2, stream="voxtral-stt").data

    # ------------------------------------------------------------------
    # File: POST /v1/audio/transcriptions
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe one finished clip over the file endpoint.

        PCM (an ``AudioChunk`` or ``AudioFrame``) is sent as a WAV, a
        ``data:`` URI as the file it carries, and an http(s) URL as
        ``file_url``: Mistral fetches it, not RoomKit. Mistral reports no
        confidence, and no language (``null`` in every answer measured).

        Raises:
            ValueError: ``language`` is not a language code.
            VoxtralSTTError: The service refused or failed the request.
        """
        code = language_code(language if language is not None else self._config.language)
        fields: dict[str, Any] = {"model": self._config.model}
        if code is not None:
            fields["language"] = code
        if self._config.context_bias:
            # One multipart field per entry, as the endpoint reads them.
            fields["context_bias"] = list(self._config.context_bias)
        part, value = self._clip_part(audio)
        response = await self._http_client().post(
            f"{self._config.base_url.rstrip('/')}/v1/audio/transcriptions",
            data=fields,
            files={part: value},
            headers={"Authorization": f"Bearer {self._config.api_key}"},
        )
        if response.status_code >= 400:
            raise http_error(response)
        return rest_result(response.json())

    def _clip_part(self, audio: AudioContent | AudioChunk | AudioFrame) -> tuple[str, Any]:
        """The multipart part carrying *audio*: the file, or the URL Mistral fetches."""
        if isinstance(audio, AudioFrame | AudioChunk):
            return "file", ("audio.wav", self._clip_wav(audio), "audio/wav")
        if audio.url.startswith("data:"):
            mime, payload = parse_data_uri(audio.url, fallback_mime=audio.mime_type)
            return "file", (f"audio{mimetypes.guess_extension(mime) or ''}", payload, mime)
        # AudioContent admits http(s) and data: URLs only.
        return "file_url", (None, audio.url)

    def _clip_wav(self, audio: AudioChunk | AudioFrame) -> bytes:
        """A clip's PCM as a mono 16-bit WAV, resampled when its rate is not one taken."""
        width = audio.sample_width if isinstance(audio, AudioFrame) else 2
        rate = audio.sample_rate if audio.sample_rate in _RATES else _DEFAULT_RATE
        pcm = self._supported_pcm(audio.data, audio.sample_rate, audio.channels, rate, width)
        buffer = BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(pcm)
        return buffer.getvalue()

    def _http_client(self) -> Any:
        if self._http is None:
            httpx = _import_httpx()
            self._http = httpx.AsyncClient(timeout=http_timeout(self._config))
        return self._http

    # ------------------------------------------------------------------
    # Realtime: wss://api.mistral.ai/v1/audio/transcriptions/realtime (voxtral_realtime)
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream audio to the realtime endpoint and yield its transcript.

        Yields the text so far as partials, and finals as the module docstring
        says: one per turn with turn detection, with a speech start when each
        turn begins; one once the stream ends without. The input rate is read
        off the first chunk: the documented rates go through as they are, any
        other is resampled to 16 kHz.

        Args:
            audio_stream: The audio; it ends when exhausted or on a chunk
                marked ``is_final``.
            language: Language for this stream, overriding the configured one.

        Raises:
            ValueError: ``language`` is not a language code.
            VoxtralSTTError: The service refused or failed the stream;
                ``retryable`` says whether a new stream may work.
        """
        code = language_code(language if language is not None else self._config.language)
        first = await first_audio(audio_stream)
        if first is None:
            return
        rate = first.sample_rate if first.sample_rate in _RATES else _DEFAULT_RATE
        if rate != first.sample_rate:
            logger.debug("Voxtral STT: resampling %d Hz to %d Hz", first.sample_rate, rate)
        session = await RealtimeSession.open(self._session_setup(rate, code), self._chunk_pcm)
        async with contextlib.aclosing(session.transcribe(first, audio_stream)) as results:
            async for result in results:
                yield result

    def _session_setup(self, rate: int, language: str | None) -> SessionSetup:
        """Every setting a session takes, sent in the one update before audio."""
        config = self._config
        settings: dict[str, Any] = {
            "audio_format": {"encoding": "pcm_s16le", "sample_rate": rate},
            "target_streaming_delay_ms": config.streaming_delay_ms,
        }
        if language is not None:
            settings["language"] = language
        if config.context_bias:
            settings["context_bias"] = list(config.context_bias)
        if config.turn_detection is not None:
            settings["turn_detection"] = asdict(config.turn_detection)
        return SessionSetup(
            url=config.stream_url,
            api_key=config.api_key,
            settings=settings,
            rate=rate,
            turns=config.turn_detection is not None,
            final_wait_s=config.final_wait_s,
            handshake_timeout_s=config.handshake_timeout_s,
            final_timeout_s=config.final_timeout_s,
        )

    def _chunk_pcm(self, chunk: AudioChunk, rate: int) -> bytes:
        return self._supported_pcm(chunk.data, chunk.sample_rate, chunk.channels, rate)

    async def close(self) -> None:
        """Release the REST client."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
