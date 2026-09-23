"""The reader of a streamed turn's response: a running tool outlives a stop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker

logger = logging.getLogger("roomkit.inbound")


class ResponseReader:
    """Reads a response stream, keeping a running tool's read out of a stop.

    While a tool call is open (its start marker read, not yet its end), the
    next read is where the tool executes. A barge-in cancels whoever is
    reading, and cancelling there would abort the tool halfway through a side
    effect and leave its row pending forever. That read runs as its own task
    instead: a cancelled reader leaves it running, and
    :meth:`finish_running_tools` lets it end (RFC §12.2 step 13s). A turn
    cancelled from outside still aborts it, through :meth:`abandon`.
    """

    def __init__(self, stream: AsyncIterator[Any]) -> None:
        self._stream = stream
        self._open: set[str] = set()
        self._pending: asyncio.Future[Any] | None = None

    async def next(self) -> Any:
        """The next item of the stream; ``StopAsyncIteration`` at its end."""
        if not self._open:
            item = await anext(self._stream)
        else:
            self._pending = asyncio.ensure_future(anext(self._stream))
            item = await asyncio.shield(self._pending)
            self._pending = None
        self._track(item)
        return item

    async def finish_running_tools(self) -> list[ToolCallEndMarker]:
        """Let the tools already running end, without starting another round.

        Returns their end markers. Reading stops as soon as no tool is open:
        one more read would ask the model for its next round.
        """
        ends: list[ToolCallEndMarker] = []
        while self._open:
            pending, self._pending = self._pending, None
            try:
                item = await (pending if pending is not None else anext(self._stream))
            except StopAsyncIteration:
                break
            except Exception:
                logger.exception("A running tool failed after the response was stopped")
                break
            self._track(item)
            if not isinstance(item, ToolCallEndMarker):
                break
            ends.append(item)
        return ends

    async def abandon(self) -> None:
        """Cancel a running tool's read (the turn itself was cancelled)."""
        pending, self._pending = self._pending, None
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    def _track(self, item: Any) -> None:
        if isinstance(item, ToolCallStartMarker):
            self._open.add(item.tool_id)
        elif isinstance(item, ToolCallEndMarker):
            self._open.discard(item.tool_id)
