"""Conversation context handed to a TTS provider (RFC §12.2.2).

A TTS that synthesizes each sentence in isolation cannot carry prosody across a
conversation. Some can, when they are told what came before: their own previous
generations, the text of the dialogue, or its audio. The Voice Channel keeps
that history per voice session here and passes a snapshot to every synthesis
call of a provider that declares it can use it.

Context audio is a copy of what the user said. It lives in this store only:
in memory, bounded, dropped when the session goes (RFC §17.6).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

_PCM_SAMPLE_WIDTH = 2


class TTSContextLevel(StrEnum):
    """What conversation context a TTS provider consumes.

    Each level includes the ones above it.
    """

    NONE = "none"
    """Nothing: every call is a sentence in isolation (the default)."""

    SELF = "self"
    """Which of its own previous turns were played, and how far."""

    TEXT = "text"
    """The text of every turn, both roles."""

    AUDIO = "audio"
    """The text and the audio of every turn."""


@dataclass(frozen=True)
class ConversationTurn:
    """One turn of the dialogue heard on a voice session."""

    turn_id: str
    role: Literal["user", "assistant"]
    participant_id: str
    text: str
    """As synthesized, or as transcribed after ``ON_TRANSCRIPTION``."""
    audio: AudioFrame | None = None
    """The whole turn, only when the provider's level is AUDIO and audio is kept."""
    played_ms: int | None = None
    """Assistant turns: how much of the audio was actually played."""
    interrupted: bool = False
    """Assistant turns: playback was cancelled before the end."""


@dataclass(frozen=True)
class TTSContext:
    """The snapshot of a session's dialogue passed to one synthesis call."""

    context_id: str
    """The voice session the context belongs to."""
    turns: tuple[ConversationTurn, ...]
    """Oldest first."""
    next_turn_id: str
    """The ``turn_id`` the assistant turn of this call will get, if it is played."""


@dataclass
class TTSContextConfig:
    """How a Voice Channel keeps the TTS conversation context.

    Attributes:
        enabled: Keep a context at all (only for a provider whose
            ``context_level`` is not NONE).
        include_audio: Keep the audio of each turn. Off by default: audio is
            a copy of the user's voice, kept only when asked for.
        max_turns: Oldest turns are dropped past this.
        max_audio_seconds: Oldest audio is dropped past this; its text stays.
    """

    enabled: bool = True
    include_audio: bool = False
    max_turns: int = 20
    max_audio_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_audio_seconds < 0:
            raise ValueError("max_audio_seconds must not be negative")


class AssistantTurnRecorder:
    """Collects what one synthesis call produced, until the turn is recorded."""

    def __init__(
        self, session_id: str, turn_id: str, *, keep_audio: bool, max_seconds: float
    ) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self._keep_audio = keep_audio
        self._max_seconds = max_seconds
        self._audio = bytearray()
        self._sample_rate: int | None = None
        self._channels = 1

    def add(self, chunk: AudioChunk) -> None:
        """Keep the chunk's audio, when audio is kept and the format allows it."""
        if not self._keep_audio or not chunk.data:
            return
        if chunk.format != "pcm_s16le" or (
            self._sample_rate is not None
            and (chunk.sample_rate, chunk.channels) != (self._sample_rate, self._channels)
        ):
            # A turn whose audio cannot be cut by duration keeps its text only.
            self._keep_audio = False
            self._audio.clear()
            return
        self._sample_rate = chunk.sample_rate
        self._channels = chunk.channels
        frame_bytes = _PCM_SAMPLE_WIDTH * self._channels
        max_bytes = int(self._max_seconds * self._sample_rate) * frame_bytes
        room = max_bytes - len(self._audio)
        if room > 0:
            self._audio.extend(chunk.data[:room])

    def audio_until(self, played_ms: int) -> AudioFrame | None:
        """The audio collected, cut at *played_ms*."""
        if not self._keep_audio or not self._audio or self._sample_rate is None:
            return None
        frame_bytes = _PCM_SAMPLE_WIDTH * self._channels
        end = int(played_ms * self._sample_rate / 1000) * frame_bytes
        data = bytes(self._audio[:end])
        if not data:
            return None
        return AudioFrame(
            data=data,
            sample_rate=self._sample_rate,
            channels=self._channels,
            sample_width=_PCM_SAMPLE_WIDTH,
        )


class TTSContextStore:
    """The dialogue of every voice session, as a TTS provider may see it.

    One history per session, never shared with another session or channel.
    """

    def __init__(self, config: TTSContextConfig, level: TTSContextLevel) -> None:
        self._config = config
        self._level = level
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._dtmf_seen: set[str] = set()
        self._lock = threading.Lock()

    @property
    def keeps_audio(self) -> bool:
        return self._level == TTSContextLevel.AUDIO and self._config.include_audio

    def sessions(self) -> list[str]:
        with self._lock:
            return list(self._turns)

    def turns(self, session_id: str) -> tuple[ConversationTurn, ...]:
        with self._lock:
            return tuple(self._turns.get(session_id, ()))

    def audio_seconds(self, session_id: str) -> float:
        with self._lock:
            return _audio_seconds(self._turns.get(session_id, ()))

    def note_dtmf(self, session_id: str) -> None:
        """A DTMF tone was heard: the current user turn must not keep its audio."""
        with self._lock:
            self._dtmf_seen.add(session_id)

    def add_user_turn(
        self,
        session_id: str,
        participant_id: str,
        text: str,
        *,
        audio: bytes | None = None,
        sample_rate: int = 16000,
        text_changed: bool = False,
    ) -> None:
        """Record what the user said, as the transcription hooks left it.

        *audio* is the utterance as 16-bit mono PCM. It is dropped when a hook
        changed the text (it would still hold what the hook removed) and when
        DTMF was heard with redaction on (the tones carry the digits).
        """
        with self._lock:
            dtmf_seen = session_id in self._dtmf_seen
            self._dtmf_seen.discard(session_id)
        keep_audio = bool(audio) and self.keeps_audio and not text_changed and not dtmf_seen
        frame = (
            AudioFrame(data=audio, sample_rate=sample_rate, sample_width=_PCM_SAMPLE_WIDTH)
            if keep_audio and audio
            else None
        )
        self._append(
            session_id,
            ConversationTurn(
                turn_id=_new_turn_id(),
                role="user",
                participant_id=participant_id,
                text=text,
                audio=frame,
            ),
        )

    def begin_assistant_turn(self, session_id: str) -> tuple[TTSContext, AssistantTurnRecorder]:
        """Snapshot the context for a synthesis call, and open its turn."""
        turn_id = _new_turn_id()
        recorder = AssistantTurnRecorder(
            session_id,
            turn_id,
            keep_audio=self.keeps_audio,
            max_seconds=self._config.max_audio_seconds,
        )
        context = TTSContext(
            context_id=session_id, turns=self.turns(session_id), next_turn_id=turn_id
        )
        return context, recorder

    def commit_assistant_turn(
        self,
        recorder: AssistantTurnRecorder,
        participant_id: str,
        text: str,
        *,
        played_ms: int,
        interrupted: bool,
    ) -> None:
        """Record what the user heard of the call; nothing heard, no turn."""
        if played_ms <= 0 or not text:
            return
        self._append(
            recorder.session_id,
            ConversationTurn(
                turn_id=recorder.turn_id,
                role="assistant",
                participant_id=participant_id,
                text=text,
                audio=recorder.audio_until(played_ms),
                played_ms=played_ms,
                interrupted=interrupted,
            ),
        )

    def release(self, session_id: str) -> None:
        """Drop everything kept for the session."""
        with self._lock:
            self._turns.pop(session_id, None)
            self._dtmf_seen.discard(session_id)

    def _append(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            history = self._turns.setdefault(session_id, [])
            history.append(turn)
            del history[: max(0, len(history) - self._config.max_turns)]
            _trim_audio(history, self._config.max_audio_seconds)


def _new_turn_id() -> str:
    return uuid.uuid4().hex


def _duration_s(frame: AudioFrame) -> float:
    return len(frame.data) / (frame.sample_width * frame.channels * frame.sample_rate)


def _audio_seconds(turns: Iterable[ConversationTurn]) -> float:
    return sum(_duration_s(t.audio) for t in turns if t.audio is not None)


def _trim_audio(history: list[ConversationTurn], max_seconds: float) -> None:
    """Drop the oldest audio until the history fits; the text stays."""
    total = _audio_seconds(history)
    for i, turn in enumerate(history):
        if total <= max_seconds:
            return
        if turn.audio is not None:
            total -= _duration_s(turn.audio)
            history[i] = replace(turn, audio=None)
