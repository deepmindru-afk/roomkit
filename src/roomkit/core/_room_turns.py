"""A channel's turns in a room, one after the other (RFC §6.4).

A channel that declares ``serializes_room_turns`` has its turns in a room run
in the order they were asked, each once the one before it is over. The room's
owner keeps the order (:class:`RoomTurns`), per process, as an ACP agent's
channel keeps its own.

A waiting turn waits in its stream (:class:`QueuedTurn`), never in the room's
delivery lane: the previous turn's answer is delivered through that lane, so a
turn waiting there would wait on itself. It starts when its delivery first
reads it, on the room as it stood when it was asked plus what the channel
wrote since (:func:`room_as_asked`), and its place is given up once its reader
has written what it said. A cancelled delivery cancels its reader, so a turn
still waiting never starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from roomkit.core.lanes import anchor_index
from roomkit.models.channel import ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.providers.utils import _aclose_stream
from roomkit.turns.owner import drop_unread, read_as_trace


@dataclass(slots=True)
class _Line:
    """The turns of one channel in one room: the one running, then the waiting."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class RoomTurns:
    """The places of the turns of channels that take a room's turns one at a
    time, in the order they were asked: one line per channel and room, kept
    while a turn runs or waits in it."""

    def __init__(self) -> None:
        self._lines: dict[tuple[str, str], _Line] = {}

    async def take(self, room_id: str, channel_id: str) -> Callable[[], None]:
        """Wait until the turns of *channel_id* in *room_id* asked before this
        one are over; the call that gives the place up, once however often it
        is made. A wait cancelled holds no place."""
        key = (room_id, channel_id)
        line = self._lines.setdefault(key, _Line())
        line.users += 1
        try:
            await line.lock.acquire()
        except BaseException:
            self._leave(key, line)
            raise
        given = False

        def give_up() -> None:
            nonlocal given
            if not given:
                given = True
                line.lock.release()
                self._leave(key, line)

        return give_up

    def _leave(self, key: tuple[str, str], line: _Line) -> None:
        line.users -= 1
        if line.users == 0:
            del self._lines[key]


def room_as_asked(asked: RoomContext, now: RoomContext, channel_id: str) -> RoomContext:
    """The room a turn that waited reads: *asked*, as it stood when the turn
    was asked, then the events *channel_id* wrote since, read in *now* (its
    previous turns' answers). A message that came after the turn is not its to
    read: that message waits a turn of its own (RFC §6.4)."""
    horizon = max(
        (event.index for event in asked.recent_events),
        default=anchor_index(asked.room.latest_index),
    )
    since = [
        event
        for event in now.recent_events
        if event.index > horizon and event.source.channel_id == channel_id
    ]
    if not since:
        return asked
    return asked.model_copy(update={"recent_events": [*asked.recent_events, *since]})


async def _no_items() -> AsyncIterator[Any]:
    return
    yield  # pragma: no cover - makes this an async generator


class QueuedTurn:
    """The stream of a channel's turn that waits its place in the room.

    Nothing runs before it is read. Its first read waits for the place, starts
    the turn, then hands the turn's stream on item by item. Closing it ends the
    turn and gives the place up: its reader closes it once it has written what
    the turn said, so the next turn reads that answer. *record* is the turn's
    response metadata as its reader reads it, brought in line with the turn's
    own record at each item.
    """

    def __init__(
        self,
        take_place: Callable[[], Awaitable[Callable[[], None]]],
        start: Callable[[], Awaitable[ChannelOutput]],
        record: ResponseMetadata,
    ) -> None:
        self._take_place = take_place
        self._start = start
        self._record = record
        self._turn_record: ResponseMetadata | None = None
        self._stream: AsyncIterator[Any] | None = None
        self._give_up: Callable[[], None] | None = None
        self._as_trace = False
        self._closed = False

    def __aiter__(self) -> QueuedTurn:
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        if self._stream is None:
            self._stream = await self._begin()
        try:
            return await self._stream.__anext__()
        finally:
            self._mirror()

    async def _begin(self) -> AsyncIterator[Any]:
        """Wait for the place, then start the turn; its stream."""
        self._give_up = await self._take_place()
        try:
            output = await self._start()
        except BaseException:
            self._give_up()
            raise
        if output.error is not None:
            self._give_up()
            raise output.error
        self._turn_record = output.response_metadata
        stream = output.response_stream if output.response_stream is not None else _no_items()
        if self._as_trace:
            read_as_trace(stream)
        return stream

    def _mirror(self) -> None:
        """Bring the record the reader reads in line with the turn's own."""
        turn = self._turn_record
        if turn is None or turn is self._record:
            return
        for key in [key for key in self._record if key not in turn]:
            del self._record[key]
        self._record.update(turn)

    async def aclose(self) -> None:
        """End the turn, what it has not said yet discarded, and give its place up."""
        self._closed = True
        try:
            if self._stream is not None:
                await _aclose_stream(self._stream)
        finally:
            if self._give_up is not None:
                self._give_up()

    def drop(self) -> None:
        """Give the turn up unread: one never started holds no place."""
        self._closed = True
        if self._stream is not None:
            drop_unread(self._stream)
        if self._give_up is not None:
            self._give_up()

    def read_as_trace(self) -> None:
        """Read the turn as a delegated task's trace (RFC §23.3 step 5), from
        its start when it has not started yet."""
        self._as_trace = True
        if self._stream is not None:
            read_as_trace(self._stream)
