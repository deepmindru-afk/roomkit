"""The frames of one turn, sent in order (RFC §10.6.4).

Text deltas the channel's stream yields without waiting between them cross as
one item: a model that delivers several chunks of one network read, or a
scripted stream, costs one frame per burst rather than one per delta. The
batch goes before any other frame of the turn, and as soon as the stream
waits; the item says how many deltas it carries, which the reader counts
against what it granted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from roomkit.turns.wire import Item

if TYPE_CHECKING:
    from roomkit.turns.wire import Frame

logger = logging.getLogger("roomkit.turns")


class FrameSender:
    """Sends a turn's frames one at a time, its text deltas by burst."""

    def __init__(self, send: Callable[[Frame], Awaitable[int]]) -> None:
        self._send = send
        self._lock = asyncio.Lock()
        self._text: list[str] = []
        self._flush: asyncio.Task[None] | None = None

    def add_text(self, text: str) -> None:
        """Keep *text* for the burst, sent once the stream waits; a text no
        frame can carry (a lone surrogate) raises ``ValueError`` here, before
        it spoils the burst."""
        text.encode()
        self._text.append(text)
        if self._flush is None:
            # Runs at the loop's next turn: once the stream's task waits.
            self._flush = asyncio.get_running_loop().create_task(self._flush_burst())

    async def send(self, frame: Frame) -> int:
        """Send *frame* after the text kept before it: its sequence number."""
        async with self._lock:
            await self._send_text()
            return await self._send(frame)

    async def _flush_burst(self) -> None:
        try:
            async with self._lock:
                # Text kept while a send suspended goes too: no flush was
                # scheduled for it.
                while self._text:
                    await self._send_text()
        except Exception:
            logger.debug("A burst of text deltas could not be sent", exc_info=True)
        finally:
            self._flush = None

    async def _send_text(self) -> None:
        if not self._text:
            return
        texts, self._text = self._text, []
        item: dict[str, object] = {"kind": "text", "text": "".join(texts)}
        if len(texts) > 1:
            item["deltas"] = len(texts)
        await self._send(Item(item=item))
