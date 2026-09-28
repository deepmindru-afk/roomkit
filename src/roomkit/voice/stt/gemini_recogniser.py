"""The dedicated Gemini recogniser: request and answer for ``gemini-3.5-transcribe``.

:class:`~roomkit.voice.stt.gemini.GeminiSTTProvider` transcribes a recording
two ways. A multimodal model is asked in a prompt and answers a JSON
transcript; that is ``gemini.py``. A dedicated recogniser has no prompt to
write: it refuses one, and refuses a structured response too. It takes a
``transcription_config`` and answers the text and, when asked, one
``word_info`` annotation per word, which is where the speaker label and the
timing ride. This module builds that config and turns the words back into the
same :class:`~roomkit.voice.stt.gemini_transcript.Transcript`.

Measured against the live API on 2026-09-27:

* Speakers come only on the words: diarization without word timestamps
  answers no annotation at all, so ``diarize`` needs ``word_timestamps``.
* ``smart`` mode and ``custom_vocabulary`` each refuse diarization and word
  timestamps with a 400.
* ``custom_vocabulary`` without a language code answers the first sentence
  alone, every time; with one, the whole recording.
* The language is never reported back, not even when the model detected it.
* Raw ``audio/l16`` is refused however its rate is spelled; WAV is accepted,
  which is why :func:`~roomkit.voice.stt.gemini_audio.audio_part` sends PCM as
  WAV.

Google documents up to one hour of audio per request, thirty minutes with
diarization or word timestamps, and up to eight speakers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from roomkit.voice.stt.gemini_transcript import Transcript, TranscriptSegment, TranscriptWord

if TYPE_CHECKING:
    from roomkit.voice.stt.gemini import GeminiSTTConfig

RECOGNISER_MODELS: tuple[str, ...] = ("gemini-3.5-transcribe",)
"""Dedicated recognisers the unary API serves, verified against ``models.list``
2026-09-27. ``gemini-3.5-transcribe-live`` streams over the Live API instead,
which :class:`~roomkit.voice.stt.gemini_transcribe.GeminiTranscribeProvider`
drives."""

TRANSCRIPTION_MODES = frozenset({"verbatim", "smart"})


def is_recogniser(model: str) -> bool:
    """Whether *model* is a dedicated recogniser rather than a model to prompt.

    By family rather than exact id, so a later ``-transcribe`` release gets the
    contract its family has. The ``-live`` one speaks over a WebSocket and is
    not a model this provider can call.
    """
    return "-transcribe" in model and not model.endswith("-live")


def recogniser_conflict(config: GeminiSTTConfig) -> str | None:
    """The first option the recogniser would refuse, as a message saying what to set."""
    if config.prompt:
        return "the dedicated recogniser takes no prompt; bias spellings with custom_vocabulary"
    if config.diarize and not config.word_timestamps:
        return "diarize needs word_timestamps: the recogniser labels speakers on the words"
    if config.mode == "smart" and (config.diarize or config.word_timestamps):
        return "mode='smart' cannot be combined with diarize or word_timestamps; set both False"
    if config.custom_vocabulary and (config.diarize or config.word_timestamps):
        return (
            "custom_vocabulary cannot be combined with diarize or word_timestamps; set both False"
        )
    if config.custom_vocabulary and not config.language:
        return (
            "custom_vocabulary needs language: without one the recogniser stops after a sentence"
        )
    return None


def transcription_config(config: GeminiSTTConfig) -> dict[str, Any]:
    """The ``transcription_config`` for *config*, which ``__post_init__`` validated."""
    request: dict[str, Any] = {}
    if config.language:
        request["language_codes"] = [config.language]
    if config.custom_vocabulary:
        request["custom_vocabulary"] = list(config.custom_vocabulary)
    if config.mode == "smart":
        request["mode"] = "smart"
        return request
    verbatim: dict[str, Any] = {"type": "verbatim"}
    if config.diarize:
        verbatim["diarization_mode"] = "speaker"
    if config.word_timestamps:
        verbatim["timestamp_granularities"] = ["word"]
    request["mode"] = verbatim
    return request


def recogniser_transcript(interaction: Any, *, language: str | None) -> Transcript:
    """Build the :class:`Transcript` from a recogniser's answer.

    With words, a turn is a run of consecutive words from one speaker. Without
    them the whole text is one untimed segment, and silence is no segment.

    Raises:
        RuntimeError: The interaction did not complete, or a word's offset
            does not read as a duration.
    """
    status = getattr(interaction, "status", None)
    if status not in (None, "completed"):
        raise RuntimeError(f"Gemini STT returned no transcript (status={status})")
    words = _words(interaction)
    if words:
        segments = _turns(words)
    else:
        text = (getattr(interaction, "output_text", None) or "").strip()
        segments = [TranscriptSegment("Speaker 1", "", "", text)] if text else []
    return Transcript(language=language or "", segments=segments, words=words)


def _annotations(interaction: Any) -> Iterator[Any]:
    for step in getattr(interaction, "steps", None) or []:
        for content in getattr(step, "content", None) or []:
            yield from getattr(content, "annotations", None) or []


def _words(interaction: Any) -> list[TranscriptWord]:
    """The ``word_info`` annotations, speakers renamed in order of first appearance.

    The service labels speakers ``spk:0``, ``spk:1``; the prompted path says
    ``Speaker 1``, ``Speaker 2``. One vocabulary whichever model answered.

    A word's start is kept from going back before the previous word's: on a
    change of speaker the service has sent the first word's start ten seconds
    early (``"4.300s"`` for a word ending at ``"14.900s"``, right after one
    ending at ``"14.100s"``; measured 2026-09-27), which dragged its whole
    turn back. Such a start is taken as the previous word's end.
    """
    labels: dict[str, str] = {}
    words: list[TranscriptWord] = []
    for annotation in _annotations(interaction):
        if getattr(annotation, "type", None) != "word_info" or not annotation.text:
            continue
        speaker = annotation.speaker
        if speaker is not None:
            speaker = labels.setdefault(speaker, f"Speaker {len(labels) + 1}")
        start = _seconds(annotation.start_offset)
        end = _seconds(annotation.end_offset)
        if words and start < words[-1].start:
            start = min(words[-1].end, end)
        words.append(TranscriptWord(text=annotation.text, start=start, end=end, speaker=speaker))
    return words


def _turns(words: list[TranscriptWord]) -> list[TranscriptSegment]:
    runs: list[list[TranscriptWord]] = []
    for word in words:
        if runs and runs[-1][-1].speaker == word.speaker:
            runs[-1].append(word)
        else:
            runs.append([word])
    return [
        TranscriptSegment(
            speaker=run[0].speaker or "Speaker 1",
            start=_mmss(run[0].start),
            end=_mmss(run[-1].end),
            text=" ".join(word.text for word in run),
        )
        for run in runs
    ]


def _seconds(offset: str | None) -> float:
    """``"1.300s"`` to ``1.3``: the service writes protobuf durations."""
    try:
        return float(str(offset).removesuffix("s"))
    except ValueError as exc:
        raise RuntimeError(f"Gemini STT returned an unreadable word offset: {offset!r}") from exc


def _mmss(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"
