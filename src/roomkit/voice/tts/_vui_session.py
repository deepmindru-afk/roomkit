"""Keep Vui's KV cache in step with the TTS conversation context (RFC §12.2.2).

Vui decodes each reply *inside* the conversation: the dialogue so far, the
user's audio included, lives in its KV cache. The Voice Channel hands the
provider that dialogue as a :class:`TTSContext`; this module works out what the
cache is missing and what it holds that the user never heard, and drives a
:class:`VuiCache` to match. It needs no GPU and no ``vui-tts``: the real cache
lives in :mod:`roomkit.voice.tts.vui`.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.vui")

FRAME_MS = 80.0  # Qwen3-TTS-12Hz codec: 12.5 frames per second


class VuiCache(Protocol):
    """The operations the session drives on one Vui row."""

    @property
    def offset(self) -> int:
        """Current KV position."""
        ...

    def restart(self, voice: str) -> None:
        """Empty the cache and prefill it with *voice*'s prompt."""
        ...

    def truncate(self, offset: int) -> None:
        """Move the KV position back to *offset*."""
        ...

    def add_user(self, text: str, audio: AudioFrame | None) -> None:
        """Write a closed user turn: its text, then its audio when given."""
        ...

    def generate(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
        """Speak *text* as the next agent turn, one PCM frame at a time."""
        ...


@dataclass
class _Generation:
    turn_id: str
    start_offset: int
    frame_offsets: list[int] = field(default_factory=list)


class VuiConversation:
    """One Vui row following one voice session's dialogue at a time.

    ``Engine(max_rows=1)`` gives a provider a single cache. The session it
    holds is kept incrementally; a call for another session (or another
    voice) restarts the cache from the prompt with the user turns awaiting a
    reply, and that session's earlier turns are not replayed.
    """

    def __init__(self, cache: VuiCache) -> None:
        self._cache = cache
        self._context_id: str | None = None
        self._voice: str | None = None
        self._applied: list[str] = []
        self._pending: _Generation | None = None

    @property
    def context_id(self) -> str | None:
        return self._context_id

    def speak(
        self, context: TTSContext | None, voice: str, text: str, cancel: threading.Event
    ) -> Generator[bytes, None, None]:
        """Bring the cache in step with *context*, then speak *text*."""
        if context is None:
            self._restart(None, voice)
        else:
            self._catch_up(context, voice)
        generation = _Generation(
            turn_id=context.next_turn_id if context else "",
            start_offset=self._cache.offset,
        )
        self._pending = generation if context is not None else None
        for pcm in self._cache.generate(text, cancel):
            generation.frame_offsets.append(self._cache.offset)
            yield pcm

    def release(self, context_id: str) -> None:
        """Forget *context_id*; the next call restarts from the prompt."""
        if context_id == self._context_id:
            self._context_id = None
            self._applied = []
            self._pending = None

    def _restart(self, context: TTSContext | None, voice: str) -> None:
        self._cache.restart(voice)
        self._context_id = context.context_id if context else None
        self._voice = voice
        self._pending = None
        self._applied = []
        if context is None:
            return
        # The user turns since the last reply are what this call answers, and
        # need no agent audio: they are written. Older history is not replayed.
        agent_turns = [i for i, t in enumerate(context.turns) if t.role == "assistant"]
        answered = agent_turns[-1] + 1 if agent_turns else 0
        for turn in context.turns[answered:]:
            self._cache.add_user(turn.text, turn.audio)
        self._applied = [t.turn_id for t in context.turns]

    def _catch_up(self, context: TTSContext, voice: str) -> None:
        if context.context_id != self._context_id or voice != self._voice:
            self._restart(context, voice)
            return
        self._settle_pending(context)
        turn_ids = [t.turn_id for t in context.turns]
        last = next(
            (i for i in range(len(turn_ids) - 1, -1, -1) if turn_ids[i] in self._applied), None
        )
        if last is None and self._applied:
            # Everything the cache holds fell out of the context window.
            self._restart(context, voice)
            return
        for turn in context.turns[(last + 1) if last is not None else 0 :]:
            if turn.role == "user":
                self._cache.add_user(turn.text, turn.audio)
                logger.debug("Vui cache: user turn added, now at %d", self._cache.offset)
            self._applied.append(turn.turn_id)

    def _settle_pending(self, context: TTSContext) -> None:
        """Cut the last generation back to what the user actually heard."""
        pending, self._pending = self._pending, None
        if pending is None:
            return
        turn = next((t for t in context.turns if t.turn_id == pending.turn_id), None)
        if turn is None:
            # Nothing of it was heard: it leaves no trace in the dialogue.
            self._cache.truncate(pending.start_offset)
            logger.debug("Vui cache: unheard reply dropped, back to %d", pending.start_offset)
            return
        heard = math.ceil((turn.played_ms or 0) / FRAME_MS)
        if turn.interrupted and heard < len(pending.frame_offsets):
            offset = pending.frame_offsets[heard - 1] if heard > 0 else pending.start_offset
            self._cache.truncate(offset)
            logger.debug(
                "Vui cache: reply cut to %d of %d frames heard, back to %d",
                heard,
                len(pending.frame_offsets),
                offset,
            )
        self._applied.append(turn.turn_id)
