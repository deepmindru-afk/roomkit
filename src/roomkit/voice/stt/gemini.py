"""Google Gemini speech-to-text provider — batch transcription of recordings.

The API accepts a complete recording, not a stream, and answers in seconds,
which is why this provider is batch-only. For live turn-taking reach for
:mod:`~roomkit.voice.stt.gemini_transcribe`, which drives Google's dedicated
``gemini-3.5-transcribe-live`` recogniser over a WebSocket, or for
:mod:`~roomkit.voice.stt.deepgram`, :mod:`~roomkit.voice.stt.gradium` or a
local :mod:`~roomkit.voice.stt.sherpa_onnx` model.

What the batch shape buys is what a streaming recogniser structurally cannot
give: the model hears the whole recording before it answers, so one pass returns
the transcript, the speaker turns and the timestamps together — no diarization
stage, no merge. That makes this the provider for meeting recordings, voicemail
and imported audio files.

Two kinds of model, chosen by ``model``:

* A multimodal model (the default, ``gemini-3.8-flash``) is *instructed*: a
  prompt asks for the transcript and a JSON schema shapes it, with turn
  timestamps to the second. It takes long recordings.
* The dedicated recogniser, ``gemini-3.5-transcribe``, is configured instead:
  it refuses a prompt, answers about twice as fast, and times every word to
  100 ms — on up to 30 minutes when it labels speakers or times words. Its
  contract lives in :mod:`~roomkit.voice.stt.gemini_recogniser`.

Two input paths, both verified against the live API on 2026-08-07:

* Inline — raw PCM or an encoded clip carried in the request. Fast, and bounded
  by the request size limit, so this provider only inlines small recordings.
* Uploaded — the Files API returns a URI the interaction refers to. This is the
  path a real meeting recording takes; uploads are deleted after use rather
  than left to expire.

Both paths live in :mod:`~roomkit.voice.stt.gemini_audio`; the transcript
models in :mod:`~roomkit.voice.stt.gemini_transcript`. This module keeps the
config, the client, the prompt and the transcription call, and remains the
public import path for all of them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.voice.base import TranscriptionResult
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.gemini_audio import SUPPORTED_MIME_TYPES, audio_part, delete_upload
from roomkit.voice.stt.gemini_recogniser import (
    RECOGNISER_MODELS,
    TRANSCRIPTION_MODES,
    is_recogniser,
    recogniser_conflict,
    recogniser_transcript,
    transcription_config,
)
from roomkit.voice.stt.gemini_transcript import Transcript, TranscriptSegment, TranscriptWord

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import AudioChunk

__all__ = [
    "RECOGNISER_MODELS",
    "SUPPORTED_MIME_TYPES",
    "GeminiSTTConfig",
    "GeminiSTTProvider",
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
]

_MAX_VOCABULARY = 1000
"""Google's documented bound on ``custom_vocabulary`` terms."""

_MAX_INLINE_BYTES = 15 * 1024 * 1024
"""Default for ``GeminiSTTConfig.max_inline_bytes``, the bound the audio sources
apply: above it a file is uploaded instead of inlined — the request has a size
limit and a base64 payload is a third larger than the file it carries."""

_TRANSCRIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {
            "type": "string",
            "description": "BCP-47 code of the language spoken, e.g. 'fr-CA'.",
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "start": {"type": "string", "description": "MM:SS from the start"},
                    "end": {"type": "string", "description": "MM:SS from the start"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "start", "end", "text"],
            },
        },
    },
    "required": ["language", "segments"],
}


@dataclass
class GeminiSTTConfig:
    """Configuration for the Gemini batch STT provider.

    Args:
        api_key: Gemini API key (``GEMINI_API_KEY``).
        model: A text/multimodal Gemini model that accepts audio input — see
            :meth:`~roomkit.providers.gemini.ai.GeminiAIProvider.available_models`
            for the catalog roomkit keeps — or a dedicated recogniser from
            :data:`RECOGNISER_MODELS`. The default is the current flash model:
            transcription is not a reasoning task, flash is the cheapest way to
            buy the audio context window, and it takes recordings longer than
            the recogniser's 30 minutes with speakers.
        language: Optional BCP-47 hint (e.g. ``"fr-CA"``). Left unset, a
            multimodal model identifies the language itself and reports it on
            the transcript; the recogniser detects it too but reports nothing,
            so the transcript's ``language`` is then empty.
        diarize: Ask for speaker labels. Worth turning off for a single-speaker
            recording, where labelling costs tokens and invents distinctions.
            A conference recorded per participant track needs no diarization at
            all — transcribe each track and merge on the timestamps. On the
            recogniser it needs ``word_timestamps``, since the labels ride on
            the words.
        prompt: Extra instruction appended to the transcription request —
            formatting rules, anything the model should know before it listens.
            The recogniser takes no prompt: setting one there is refused.
        timeout: Per-request timeout in seconds. Generous by design: a model
            answering on an hour of audio is not answering in milliseconds.
        max_inline_bytes: Recordings larger than this are uploaded through the
            Files API instead of being inlined in the request.
        connect_timeout: TCP connect timeout in seconds, apart from ``timeout``.
        mode: ``"verbatim"`` (the default) or ``"smart"``, the recogniser's
            cleaned-up transcript: fillers dropped, self-corrections resolved,
            formatting applied. ``"smart"`` needs the recogniser and excludes
            ``diarize`` and ``word_timestamps``.
        custom_vocabulary: Terms to spell as written ("RoomKit"), up to 1000.
            The recogniser takes them natively and then excludes ``diarize``
            and ``word_timestamps``; a multimodal model reads them in its
            prompt.
        word_timestamps: Time every word (:attr:`Transcript.words`) and the
            turns built from them. Recogniser only — a multimodal model always
            times its turns to the second and never its words. Turning it off
            lifts the recogniser's limit from 30 minutes to an hour.
    """

    api_key: str = field(repr=False)
    model: str = "gemini-3.8-flash"
    language: str | None = None
    diarize: bool = True
    prompt: str | None = None
    timeout: float = 600.0
    max_inline_bytes: int = _MAX_INLINE_BYTES
    connect_timeout: float = 5.0
    mode: str = "verbatim"
    custom_vocabulary: list[str] = field(default_factory=list)
    word_timestamps: bool = True

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be blank when provided")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(self.connect_timeout) or self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be a positive finite number")
        if self.max_inline_bytes <= 0:
            raise ValueError("max_inline_bytes must be positive")
        self._check_recognition_options()

    def _check_recognition_options(self) -> None:
        """Refuse here what the recogniser would refuse with a 400 on the first call."""
        self.mode = self.mode.lower()
        if self.mode not in TRANSCRIPTION_MODES:
            raise ValueError(
                f"mode must be one of {sorted(TRANSCRIPTION_MODES)}, got {self.mode!r}"
            )
        if len(self.custom_vocabulary) > _MAX_VOCABULARY:
            raise ValueError(f"custom_vocabulary takes at most {_MAX_VOCABULARY} terms")
        if not is_recogniser(self.model):
            if self.mode == "smart":
                raise ValueError(f"mode='smart' needs a dedicated recogniser: {RECOGNISER_MODELS}")
            return
        conflict = recogniser_conflict(self)
        if conflict:
            raise ValueError(conflict)


class GeminiSTTProvider(STTProvider):
    """Google Gemini speech-to-text provider.

    Batch only — :attr:`supports_streaming` is ``False``, so a
    :class:`~roomkit.channels.voice.VoiceChannel` transcribes on ``SPEECH_END``
    rather than streaming partials. Given seconds of model latency, the honest
    placement is ``batch_mode=True`` (dictation, voicemail) or transcription of
    a finished recording, not live turn-taking.

    Two entry points:

    * :meth:`transcribe` — the ABC contract, returning flat text.
    * :meth:`transcribe_recording` — the whole :class:`Transcript`, with speaker
      turns and timestamps, and the one that accepts a file path.
    """

    def __init__(self, config: GeminiSTTConfig) -> None:
        self._config = config
        self._client: Any = None
        self._http: Any = None

    @property
    def name(self) -> str:
        return "GeminiSTT"

    @property
    def supports_streaming(self) -> bool:
        """The API takes a complete recording; there is no stream to open."""
        return False

    # ------------------------------------------------------------------
    # Client and prompt
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            # The client carries the connect/read split; see ``build_genai_client``
            # for why it cannot go on the request.
            built = build_genai_client(
                self._config, provider="GeminiSTTProvider", api_key=self._config.api_key
            )
            self._client, self._http = built.client, built.http
        return self._client

    def _build_prompt(self) -> str:
        lines = [
            "Transcribe the recording verbatim.",
            "Return only what is spoken: do not summarise, translate, or comment.",
            "Timestamps are MM:SS measured from the start of the recording.",
        ]
        if self._config.diarize:
            lines.append(
                'Label each speaker "Speaker 1", "Speaker 2", and so on, '
                "in order of first appearance, and keep a label attached to the "
                "same voice throughout."
            )
        else:
            lines.append('The recording has one speaker; label every segment "Speaker 1".')
        if self._config.language:
            lines.append(f"The recording is in {self._config.language}.")
        if self._config.custom_vocabulary:
            terms = ", ".join(self._config.custom_vocabulary)
            lines.append(f"Spell these terms exactly as written when they are spoken: {terms}.")
        if self._config.prompt:
            lines.append(self._config.prompt)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe a complete recording.

        Args:
            audio: Audio content (``data:`` URL, local path or Files API URI),
                raw audio chunk, or audio frame.

        Returns:
            TranscriptionResult whose ``text`` carries the spoken words and
            whose ``language`` carries what the model identified. Speaker turns
            and timestamps are dropped by this shape — call
            :meth:`transcribe_recording` for those.
        """
        transcript = await self.transcribe_recording(audio)
        return TranscriptionResult(
            text=transcript.plain_text,
            is_final=True,
            language=transcript.language or None,
        )

    async def transcribe_recording(self, source: Any) -> Transcript:
        """Transcribe a recording into speaker turns.

        Args:
            source: A path to a recording (``str`` or ``Path``), an
                ``AudioContent``, an ``AudioChunk`` or an ``AudioFrame``. Files
                larger than ``max_inline_bytes`` are uploaded through the Files
                API and deleted afterwards.

        Returns:
            The :class:`Transcript` — language, one segment per speaker turn,
            and, from the dedicated recogniser, the timed words.

        Raises:
            RuntimeError: The model answered without a usable transcript.
        """
        part, uploaded_name = await audio_part(
            source,
            max_inline_bytes=self._config.max_inline_bytes,
            upload_timeout=self._config.timeout,
            get_client=self._get_client,
        )
        try:
            interaction = await self._get_client().aio.interactions.create(
                model=self._config.model,
                **self._request(part),
                # No per-request ``timeout``: the SDK would flatten it to one
                # float; the connect/read split is on the client (``_get_client``).
            )
        finally:
            if uploaded_name is not None:
                await delete_upload(uploaded_name, get_client=self._get_client)

        if is_recogniser(self._config.model):
            return recogniser_transcript(interaction, language=self._config.language)
        return self._prompted_transcript(interaction)

    def _request(self, part: dict[str, Any]) -> dict[str, Any]:
        """The request the configured model takes around the audio *part*."""
        if is_recogniser(self._config.model):
            return {
                "input": [part],
                "generation_config": {"transcription_config": transcription_config(self._config)},
            }
        return {
            "input": [part, {"type": "text", "text": self._build_prompt()}],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": _TRANSCRIPT_SCHEMA,
            },
        }

    @staticmethod
    def _prompted_transcript(interaction: Any) -> Transcript:
        """Read the JSON transcript a multimodal model answered."""
        payload = getattr(interaction, "output_text", None)
        if not payload:
            raise RuntimeError(
                f"Gemini STT returned no transcript "
                f"(status={getattr(interaction, 'status', None)})"
            )
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini STT returned a transcript that is not JSON") from exc

        segments = [
            TranscriptSegment(
                speaker=str(item.get("speaker", "Speaker 1")),
                start=str(item.get("start", "")),
                end=str(item.get("end", "")),
                text=str(item.get("text", "")),
            )
            for item in parsed.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
        return Transcript(language=str(parsed.get("language", "")), segments=segments)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the genai client's connection pool and drop the reference."""
        client, self._client = self._client, None
        http, self._http = self._http, None
        await close_genai_client(client, http)
