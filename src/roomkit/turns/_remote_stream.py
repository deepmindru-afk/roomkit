"""The owner's copy of a turn's response stream (RFC §10.6.4).

Each read pulls one item from the executor; closing it closes the channel's
stream. A stream its reader drops unclosed is closed for it, as a collected
generator would be.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.turns._owner_turn import OwnerTurn

# The closes of streams their reader dropped, held until they end.
_DROPPED_CLOSES: set[asyncio.Task[None]] = set()


def _close_dropped(turn: OwnerTurn, loop: asyncio.AbstractEventLoop) -> None:
    """Close the stream of *turn*, which its reader dropped unclosed: the
    executor closes the channel's, as a collected generator would be."""
    if turn.over or loop.is_closed():
        return

    def start() -> None:
        task = loop.create_task(turn.close_stream())
        _DROPPED_CLOSES.add(task)
        task.add_done_callback(_DROPPED_CLOSES.discard)

    loop.call_soon_threadsafe(start)


class RemoteStream:
    """The owner's copy of a turn's response stream: each read pulls one
    item from the executor, closing it closes the channel's (RFC §10.6.4)."""

    def __init__(self, turn: OwnerTurn) -> None:
        self._turn = turn
        self._reading = False
        self._done = False
        # A stream its reader drops without closing it is closed for it.
        self._dropped = weakref.finalize(self, _close_dropped, turn, asyncio.get_running_loop())

    def __aiter__(self) -> RemoteStream:
        return self

    async def __anext__(self) -> Any:
        if self._done:
            raise StopAsyncIteration
        if self._reading:
            raise RuntimeError("anext(): asynchronous generator is already running")
        self._reading = True
        try:
            return await self._turn.read_item()
        except BaseException:
            self._done = True
            self._dropped.detach()
            raise
        finally:
            self._reading = False

    def drop(self) -> None:
        """Give the stream up unread: it is closed in the background, as its
        collection would close it."""
        if not self._done and not self._reading:
            self._done = True
            self._dropped.detach()
            _close_dropped(self._turn, asyncio.get_running_loop())

    async def aclose(self) -> None:
        if self._done:
            return
        if self._reading:
            raise RuntimeError("aclose(): asynchronous generator is already running")
        self._done = True
        self._dropped.detach()
        await self._turn.close_stream()
