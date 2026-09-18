"""Google Gemini streaming speech-to-text over the Live API.

``gemini-3.5-transcribe-live`` is a dedicated recogniser, not a chat model
asked to write down what it hears: it takes a PCM stream over a WebSocket and
answers with interim and final transcripts as the caller speaks. That is the
opposite shape from :mod:`~roomkit.voice.stt.gemini`, which sends a complete
recording to a multimodal model and waits seconds for one answer, and the two
live apart for that reason rather than because they come from the same vendor.

Pick this one for live turn-taking; pick the batch one for a meeting recording,
where seeing the whole file at once is what buys speaker turns and timestamps
in a single pass.

Protocol facts, from Google's Live transcription guide:

* Audio is raw 16-bit PCM, 16 kHz, mono, little-endian, sent in short chunks.
* ``interim_input_transcription`` carries partials, ``input_transcription``
  the finals.
* Diarization and word-level timestamps are **not** available over the Live
  API, whatever the batch model offers. :attr:`GeminiTranscribeConfig` does
  not pretend otherwise: it has no knob for either.
* A session is capped at ten minutes, which is why :meth:`transcribe` states
  the bound instead of quietly truncating a long recording.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame

__all__ = ["GeminiTranscribeConfig", "GeminiTranscribeProvider"]

logger = logging.getLogger("roomkit.voice.stt.gemini_transcribe")

REQUIRED_SAMPLE_RATE = 16000
"""The only rate the Live transcription model documents. Audio at another rate
is still sent, with its true rate in the MIME type and a warning: guessing a
resample here would hide a pipeline that is misconfigured upstream."""

TRANSCRIPTION_MODES = frozenset({"VERBATIM", "SMART"})
"""``VERBATIM`` returns what was said. ``SMART`` removes disfluencies and
formats; Google documents it as incompatible with timestamps and diarization,
neither of which the Live API offers anyway."""

MAX_SESSION_SECONDS = 600
"""Google's cap on one Live transcription session. :meth:`transcribe` refuses a
recording longer than this up front: the socket would close on it part-way
through, and a partial transcript reads like a complete one."""


@dataclass
class GeminiTranscribeConfig:
    """Configuration for :class:`GeminiTranscribeProvider`.

    Attributes:
        api_key: API key for the Gemini Developer API.
        model: Live transcription model id.
        language_codes: BCP-47 hints, e.g. ``["fr-FR", "en-US"]``. Empty is
            not a missing value here: it is what asks the model to identify
            the language itself, across the 85+ locales it covers.
        custom_vocabulary: Terms to bias towards, up to 1000. Google reports
            the best results below about 100, so this is a place for the
            names a generic recogniser gets wrong, not a dictionary.
        mode: ``VERBATIM`` or ``SMART``.
        timeout: Read budget in seconds for the underlying HTTP client.
        connect_timeout: TCP connect timeout in seconds, kept apart from
            ``timeout`` so a host that no longer accepts connections is given
            up on in seconds rather than after the read budget.
    """

    api_key: str = field(repr=False)
    model: str = "gemini-3.5-transcribe-live"
    language_codes: list[str] = field(default_factory=list)
    custom_vocabulary: list[str] = field(default_factory=list)
    mode: str = "VERBATIM"
    timeout: float = 60.0
    connect_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        self.mode = self.mode.upper()
        if self.mode not in TRANSCRIPTION_MODES:
            raise ValueError(
                f"mode must be one of {sorted(TRANSCRIPTION_MODES)}, got {self.mode!r}"
            )
        if len(self.custom_vocabulary) > 1000:
            raise ValueError("custom_vocabulary takes at most 1000 terms")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(self.connect_timeout) or self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be a positive finite number")


class GeminiTranscribeProvider(STTProvider):
    """Streaming speech-to-text on ``gemini-3.5-transcribe-live``.

    The counterpart of :class:`~roomkit.voice.stt.gemini.GeminiSTTProvider`:
    that one transcribes a finished recording, this one transcribes a caller
    while they speak.
    """

    def __init__(self, config: GeminiTranscribeConfig) -> None:
        self._config = config
        self._client: Any = None
        self._http: Any = None
        self._warned_sample_rates: set[int] = set()

    @property
    def name(self) -> str:
        return "GeminiTranscribe"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_language_override(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Client and setup
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            built = build_genai_client(
                self._config,
                provider="GeminiTranscribeProvider",
                api_key=self._config.api_key,
            )
            self._client, self._http = built.client, built.http
        return self._client

    def _build_config(self, language: str | None) -> Any:
        from google.genai import types

        # A per-call language replaces the configured hints rather than
        # joining them: the caller naming one language means that language,
        # and appending it to a list of others would dilute the hint it is.
        codes = [language] if language else list(self._config.language_codes)

        transcription_kwargs: dict[str, Any] = {"mode": self._config.mode}
        if codes:
            transcription_kwargs["language_codes"] = codes
        if self._config.custom_vocabulary:
            transcription_kwargs["custom_vocabulary"] = list(self._config.custom_vocabulary)

        return types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(**transcription_kwargs),
        )

    def _blob(self, types: Any, chunk: AudioChunk) -> Any:
        rate = chunk.sample_rate or REQUIRED_SAMPLE_RATE
        if rate != REQUIRED_SAMPLE_RATE and rate not in self._warned_sample_rates:
            self._warned_sample_rates.add(rate)
            logger.warning(
                "%s documents %d Hz input; this stream is %d Hz. Sending it as is - "
                "resample upstream if the transcript comes back wrong.",
                self._config.model,
                REQUIRED_SAMPLE_RATE,
                rate,
            )
        return types.Blob(data=chunk.data, mime_type=f"audio/pcm;rate={rate}")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Yield interim and final transcripts as the audio arrives.

        ``language`` replaces the configured hints for this stream only. It is
        fixed for the life of the stream, because it is part of the session
        setup: the caller opens a new stream to change it.
        """
        from google.genai import types

        client = self._get_client()
        config = self._build_config(language)

        async with client.aio.live.connect(model=self._config.model, config=config) as session:
            # The send side runs on its own task: receive() must already be
            # draining when the first chunks go out, or a slow consumer
            # would stall the socket the sender is writing to.
            sender = asyncio.create_task(
                self._send_audio(session, types, audio_stream), name="gemini-transcribe-send"
            )
            try:
                async for response in session.receive():
                    content = getattr(response, "server_content", None)
                    if content is None:
                        continue

                    interim = getattr(content, "interim_input_transcription", None)
                    if interim is not None and interim.text:
                        yield TranscriptionResult(text=interim.text, is_final=False)

                    final = getattr(content, "input_transcription", None)
                    if final is not None and final.text:
                        yield TranscriptionResult(text=final.text, is_final=True)

                    # The audio is over and the server has closed its turn:
                    # nothing further is coming, and receive() alone would
                    # keep waiting on a socket that has said all it has to say.
                    if sender.done() and (
                        getattr(content, "turn_complete", None)
                        or getattr(content, "generation_complete", None)
                    ):
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                # Surface a send-side failure rather than ending the stream
                # clean: a caller that saw no exception assumes it got the
                # whole transcript.
                with_error = await asyncio.gather(sender, return_exceptions=True)
                error = with_error[0]
                if isinstance(error, Exception) and not isinstance(error, asyncio.CancelledError):
                    raise error

    async def _send_audio(
        self, session: Any, types: Any, audio_stream: AsyncIterator[AudioChunk]
    ) -> None:
        """Pump the caller's audio into the session, then close the input."""
        async for chunk in audio_stream:
            if not chunk.data:
                continue
            await session.send_realtime_input(audio=self._blob(types, chunk))
        await session.send_realtime_input(audio_stream_end=True)

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe one complete chunk through a streaming session.

        Convenience over the same socket rather than a second API: there is
        one client, one auth path and one config to keep correct. The Live
        session is capped at ten minutes, so a longer recording belongs to
        :class:`~roomkit.voice.stt.gemini.GeminiSTTProvider`, which also
        returns speaker turns and timestamps.
        """
        chunk = self._as_chunk(audio)
        seconds = len(chunk.data) / (chunk.sample_rate * chunk.channels * 2)
        if seconds > MAX_SESSION_SECONDS:
            raise ValueError(
                f"GeminiTranscribeProvider takes at most {MAX_SESSION_SECONDS // 60} minutes "
                f"of audio per session and this recording is {seconds:.0f} s long. "
                "GeminiSTTProvider transcribes a finished recording of any length."
            )

        async def one_chunk() -> AsyncIterator[AudioChunk]:
            yield chunk

        finals: list[str] = []
        async for result in self.transcribe_stream(one_chunk(), language=language):
            if result.is_final and result.text:
                finals.append(result.text)
        return TranscriptionResult(text=" ".join(finals).strip())

    @staticmethod
    def _as_chunk(audio: AudioContent | AudioChunk | AudioFrame) -> AudioChunk:
        """Narrow the ABC's three input shapes to the one the socket takes."""
        data = getattr(audio, "data", None)
        if not isinstance(data, bytes | bytearray):
            raise TypeError(
                "GeminiTranscribeProvider transcribes raw PCM: pass an AudioChunk or "
                "an AudioFrame. A URL-backed AudioContent belongs to GeminiSTTProvider."
            )
        return AudioChunk(
            data=bytes(data),
            sample_rate=getattr(audio, "sample_rate", REQUIRED_SAMPLE_RATE),
        )

    async def close(self) -> None:
        await close_genai_client(self._client, self._http)
        self._client = None
        self._http = None
