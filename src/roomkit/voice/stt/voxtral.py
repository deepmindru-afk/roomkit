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

``diarize=True`` labels who spoke, in a file and live; every final then names
its speaker (RFC §12.2.3). Live, the service cannot detect turns and label
speakers in one session, so ``turn_detection`` must be ``None``: a turn ends
on a change of speaker or a pause without new words (``diarized_pause_ms``),
and a ``VoiceChannel`` carries the labels in continuous mode only.

Each stream is one session on the service. Voxtral 3 is in public preview
(October 2026). Install with ``pip install roomkit[mistral-voice]``.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import mimetypes
import wave
from collections.abc import AsyncIterator
from dataclasses import asdict
from io import BytesIO
from typing import TYPE_CHECKING, Any

from roomkit.providers.utils import http_timeout, parse_data_uri
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.voxtral_config import (
    DEFAULT_MODEL as DEFAULT_MODEL,
)
from roomkit.voice.stt.voxtral_config import (
    DEFAULT_REALTIME_MODEL as DEFAULT_REALTIME_MODEL,
)
from roomkit.voice.stt.voxtral_config import (
    MAX_SPEAKER_REFERENCES as MAX_SPEAKER_REFERENCES,
)
from roomkit.voice.stt.voxtral_config import VoxtralSTTConfig as VoxtralSTTConfig
from roomkit.voice.stt.voxtral_config import VoxtralTurnDetection as VoxtralTurnDetection
from roomkit.voice.stt.voxtral_protocol import VoxtralSTTError as VoxtralSTTError
from roomkit.voice.stt.voxtral_protocol import http_error, language_code, rest_result
from roomkit.voice.stt.voxtral_realtime import RealtimeSession, SessionSetup, first_audio

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger("roomkit.voice.stt.voxtral")


_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})
"""Input rates sent as they are: the rates Mistral documents for the realtime
model. Any other is resampled to 16 kHz, the service's default."""
_DEFAULT_RATE = 16000


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for VoxtralSTTProvider.transcribe(). "
            "Install with: pip install roomkit[mistral-voice]"
        ) from exc
    return httpx


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

    @property
    def supports_diarization(self) -> bool:
        """True with ``diarize``: every final then names its speaker."""
        return self._config.diarize

    @property
    def keeps_stream_across_turns(self) -> bool:
        """With turn detection, one session carries turn after turn.

        A turn's last words trail its ``turn.end``, past the next
        ``turn.start`` at times: on a stream per turn, a turn started within
        the final wait would be cut in two.
        """
        return self._config.turn_detection is not None

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
        if self._config.diarize:
            # Speakers are labelled per segment: the endpoint refuses diarize without them.
            fields["diarize"] = "true"
            fields["timestamp_granularities"] = "segment"
        part, value = self._clip_part(audio)
        response = await self._http_client().post(
            f"{self._config.base_url.rstrip('/')}/v1/audio/transcriptions",
            data=fields,
            files={part: value},
            headers={"Authorization": f"Bearer {self._config.api_key}"},
        )
        if response.status_code >= 400:
            raise http_error(response)
        return rest_result(response.json(), diarized=self._config.diarize)

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
        turn begins; with ``diarize``, one per speaker's run of words, ended by
        another voice or a pause, each naming its speaker; one once the stream
        ends otherwise. The input rate is read
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
        if config.diarize:
            settings["diarize"] = True
        if config.speaker_references:
            settings["speaker_references"] = {
                name: base64.b64encode(clip).decode("ascii")
                for name, clip in config.speaker_references.items()
            }
        return SessionSetup(
            url=config.stream_url,
            api_key=config.api_key,
            settings=settings,
            rate=rate,
            turns=config.turn_detection is not None,
            diarize=config.diarize,
            pause_s=config.diarized_pause_ms / 1000,
            final_wait_s=config.final_wait_s,
            next_turn_wait_s=config.next_turn_wait_s,
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
