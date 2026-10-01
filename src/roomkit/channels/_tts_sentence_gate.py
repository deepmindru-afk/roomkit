"""BEFORE_TTS on a streamed response: the hook judges each sentence (RFC §12.2 step 12s.b)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext

logger = logging.getLogger("roomkit.channels.voice")


class SentenceHookGate:
    """Runs the room's BEFORE_TTS hooks on each sentence before the TTS reads it.

    A streamed response cannot be held back whole, so the hook runs where the
    TTS reads: one sentence at a time, once for every session. A MODIFY
    replaces the sentence, a BLOCK drops it, and the engine's fail-closed rule
    for BEFORE_TTS — a hook that raises, times out or returns something
    unusable blocks — applies sentence by sentence. A sentence redacted to an
    empty string is not synthesized.
    """

    def __init__(self, hooks: HookEngine, room_id: str, context: RoomContext) -> None:
        self._hooks = hooks
        self._room_id = room_id
        self._context = context
        self._spoken: list[str] = []
        # Whether a hook replaced or dropped at least one sentence.
        self.changed = False

    async def run(self, sentences: AsyncIterator[str]) -> AsyncIterator[str]:
        """Yield each sentence as the hooks left it, skipping the dropped ones."""
        async for sentence in sentences:
            spoken = await self._judge(sentence)
            if spoken:
                self._spoken.append(spoken)
                yield spoken

    def text(self) -> str:
        """The response as the sessions were sent it."""
        return " ".join(s.strip() for s in self._spoken if s.strip())

    async def _judge(self, sentence: str) -> str:
        result = await self._hooks.run_sync_hooks(
            self._room_id,
            HookTrigger.BEFORE_TTS,
            sentence,
            self._context,
            skip_event_filter=True,
        )
        if not result.allowed:
            logger.info("TTS sentence blocked by hook: %s", result.reason)
            self.changed = True
            return ""
        spoken = result.event if isinstance(result.event, str) else sentence
        if spoken != sentence:
            self.changed = True
        return spoken
