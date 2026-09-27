"""Transcript models for the Gemini STT provider.

What :meth:`~roomkit.voice.stt.gemini.GeminiSTTProvider.transcribe_recording`
hands back: a whole recording as speaker turns, with the timestamps as the
model read them. Kept beside the provider so ``gemini.py`` holds the
transcription alone; the public import path stays ``roomkit.voice.stt.gemini``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TranscriptSegment:
    """One speaker turn.

    Timestamps are ``MM:SS`` strings, as the model returns them. They are the
    model's reading of the recording, not a forced alignment: treat them as
    navigation, not as sync marks. They are empty when the model returned no
    timing at all (the dedicated recogniser with ``word_timestamps=False``).
    """

    speaker: str
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class TranscriptWord:
    """One recognised word with its timing, from the dedicated recogniser.

    Offsets are seconds from the start of the recording, at the 100 ms
    resolution the service reports.
    """

    text: str
    start: float
    end: float
    speaker: str | None = None
    """Label of the turn the word belongs to, or ``None`` without diarization."""


@dataclass(frozen=True)
class Transcript:
    """A whole recording, as speaker turns."""

    language: str
    """BCP-47 code of the language spoken. Empty when the model reported none:
    the dedicated recogniser only knows the language it was told."""
    segments: list[TranscriptSegment]
    words: list[TranscriptWord] = field(default_factory=list)
    """Word-level timing, when the model returned it (the dedicated recogniser
    with ``word_timestamps=True``); empty otherwise."""

    @property
    def text(self) -> str:
        """The turns joined into a readable transcript, one line per speaker."""
        return "\n".join(f"{s.speaker}: {s.text}" for s in self.segments)

    @property
    def plain_text(self) -> str:
        """The spoken words alone, without speaker labels."""
        return " ".join(s.text for s in self.segments)
