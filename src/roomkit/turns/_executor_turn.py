"""One turn on its executor (RFC §10.6.3, §10.6.4, §10.6.8).

The turn runs the channel's ``on_event``, sends its output, then advances its
stream one item per item the owner pulls. The channel changes some of what it
already yielded (its response-metadata record, a call's arguments once its
gate fixed them, a call's end before the end itself is read): after each step
the turn sends what changed, before the item it read, so the owner's copies
read as the channel's own.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import AsyncIterator, Awaitable, Coroutine
from typing import TYPE_CHECKING, Any

from pydantic_core import to_jsonable_python

from roomkit.channels._discussion_turn import DISCUSSION_TURN, issue_mark, retire_mark
from roomkit.core.exceptions import ChannelNotRegisteredError, TurnLinkError
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.turns._codec import encode_end, encode_item, encode_output, error_record
from roomkit.turns.base import heartbeat_tick
from roomkit.turns.wire import (
    Accepted,
    Cancel,
    Close,
    End,
    Error,
    Heartbeat,
    Item,
    Output,
    Pull,
    Record,
    ToolArgs,
    ToolEnded,
)

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.models.event import RoomEvent
    from roomkit.turns.base import ExecutorLink, TurnTimeouts
    from roomkit.turns.wire import Frame

logger = logging.getLogger("roomkit.turns")

_TURNS_RUN_UNDER: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "roomkit_turns_run_under", default=()
)
"""The turns the current code runs under, outermost first: a close called
from one of them must not cut it (the turns' :func:`held_runs`)."""


def turns_run_under() -> tuple[str, ...]:
    """The ids of the turns the current code runs under."""
    return _TURNS_RUN_UNDER.get()


def _enter_turn(turn_id: str) -> None:
    turns = _TURNS_RUN_UNDER.get()
    if turn_id not in turns:
        _TURNS_RUN_UNDER.set((*turns, turn_id))


def _snapshot(record: Any) -> dict[str, Any]:
    """A deep, JSON-ready copy of a record: a write into a nested map (an
    ACP turn's ``acp`` entry) is a change too."""
    return to_jsonable_python(dict(record), fallback=str)


async def _next_item(stream: AsyncIterator[Any]) -> Any:
    return await anext(stream)


async def _close_stream(stream: AsyncIterator[Any]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


class ExecutorTurn:
    """Runs one turn a link brought and answers it with frames."""

    def __init__(
        self,
        link: ExecutorLink,
        executor_id: str,
        channel: Awaitable[Channel | None],
        timeouts: TurnTimeouts,
    ) -> None:
        self._link = link
        self._request = link.request
        self._executor_id = executor_id
        self._channel = channel
        self._timeouts = timeouts
        self._credits = 0
        self._closing = False
        self._cut: str | None = None
        self._wake = asyncio.Event()
        self._heard = True
        self._work: asyncio.Task[Any] | None = None
        self._spared = False
        self._mark: dict[str, Any] | None = None
        self._record: dict[str, Any] = {}
        self._live_record: Any = None
        # The calls the stream started and has not ended, by their start's frame.
        self._open: dict[int, ToolCallStartMarker] = {}
        self._args_sent: dict[int, Any] = {}
        self._ends_sent: set[int] = set()

    async def run(self) -> None:
        """Run the turn to its end, which the owner is always told."""
        watch = asyncio.create_task(self._read_controls())
        beat = asyncio.create_task(self._beat())
        try:
            await self._send(Accepted(executor_id=self._executor_id))
            await self._turn()
        finally:
            if self._mark is not None:
                # The turn is over: the mark its notes rode reads as no mark.
                retire_mark(self._mark)
            watch.cancel()
            beat.cancel()
            await asyncio.gather(watch, beat, return_exceptions=True)
            await self._send(End())

    # -- The turn -----------------------------------------------------------

    async def _turn(self) -> None:
        """``on_event``, its output, then its stream as the owner pulls it."""
        channel = await self._channel
        if channel is None:
            missing = ChannelNotRegisteredError(
                f"No channel {self._request.channel_id!r} on executor {self._executor_id}"
            )
            await self._send(Error(error=error_record(missing)))
            return
        event = self._marked_event()
        try:
            output = await self._run_work(
                channel.on_event(event, self._request.binding, self._request.context),
                self._link.turn_context(),
            )
        except asyncio.CancelledError:
            if self._cut is None:
                raise
            return
        except Exception as exc:
            await self._send(Error(error=error_record(exc)))
            return
        self._live_record = output.response_metadata
        self._record = _snapshot(output.response_metadata)
        stream = output.response_stream
        await self._send(Output(output=encode_output(output), streams=stream is not None))
        if stream is not None:
            await self._serve_stream(stream)

    def _marked_event(self) -> RoomEvent:
        """The request's event, carrying a mark of this process for a
        discussion's turn so the channel reads its notes here."""
        event = self._request.event
        if not self._request.in_place:
            return event
        self._mark = issue_mark(self._request.notes)
        metadata = {**(event.metadata or {}), DISCUSSION_TURN: self._mark}
        return event.model_copy(update={"metadata": metadata})

    async def _serve_stream(self, stream: AsyncIterator[Any]) -> None:
        """Advance *stream* one item per pull, until its end, a close or a cut."""
        context: contextvars.Context | None = None
        while True:
            await self._wait_for_pull()
            if context is None:
                context = self._link.stream_context() or contextvars.Context()
            if self._closing or self._cut is not None:
                await self._close(stream, context)
                return
            self._credits -= 1
            try:
                item = await self._run_work(_next_item(stream), context)
            except StopAsyncIteration:
                await self._report_changes()
                await self._send(Item(item={"kind": "stream_end"}))
                return
            except asyncio.CancelledError:
                if self._cut is None:
                    raise
                await self._report_changes()
                return
            except Exception as exc:
                await self._report_changes()
                await self._send(Error(error=error_record(exc)))
                return
            await self._report_changes(before=item)
            await self._send_item(item)

    async def _close(self, stream: AsyncIterator[Any], context: contextvars.Context) -> None:
        """Close the stream as a local reader's close would."""
        try:
            await self._run_work(_close_stream(stream), context)
        except asyncio.CancelledError:
            if self._cut is None:
                raise
        except Exception:
            logger.exception("Closing the stream of turn %s failed", self._request.turn_id)
        await self._report_changes()

    async def _run_work(self, work: Coroutine[Any, Any, Any], context: contextvars.Context) -> Any:
        """Run one piece of the turn's work as a task the owner can cancel,
        the code under it knowing it runs under this turn."""
        context.run(_enter_turn, self._request.turn_id)
        self._work = asyncio.get_running_loop().create_task(
            work, context=context, name=f"roomkit-turn-{self._request.turn_id}-work"
        )
        try:
            return await self._work
        finally:
            self._work = None

    async def _wait_for_pull(self) -> None:
        while self._credits <= 0 and not self._closing and self._cut is None:
            self._wake.clear()
            await self._wake.wait()

    # -- What changed on the live objects ------------------------------------

    async def _report_changes(self, *, before: Any = None) -> None:
        """Send what the channel changed since the last item: its record, a
        call's fixed arguments, a call's end (but the one *before* is)."""
        if self._live_record is not None:
            snapshot = _snapshot(self._live_record)
            if snapshot != self._record:
                self._record = snapshot
                await self._send(Record(record=snapshot))
        for seq, start in self._open.items():
            if start.ran_with is not None and self._args_sent.get(seq) != start.ran_with:
                self._args_sent[seq] = start.ran_with
                await self._send(ToolArgs(start=seq, arguments=start.ran_with))
            ended = start.ended
            if ended is not None and ended is not before and seq not in self._ends_sent:
                self._ends_sent.add(seq)
                await self._send(ToolEnded(start=seq, end=encode_end(ended)))

    async def _send_item(self, item: Any) -> None:
        """Send one item, a call's end naming the frame of its start."""
        if isinstance(item, ToolCallEndMarker):
            start = self._start_of(item)
            await self._send(Item(item=encode_item(item, start=start)))
            return
        seq = await self._send(Item(item=encode_item(item)))
        if isinstance(item, ToolCallStartMarker):
            self._open[seq] = item

    def _start_of(self, end: ToolCallEndMarker) -> int | None:
        """The start *end* closes: the one it rides, else the earliest open
        under its id (two calls under one id are two calls, RFC §6.4)."""
        seq = next((s for s, start in self._open.items() if start.ended is end), None)
        if seq is None:
            seq = next(
                (s for s, start in self._open.items() if start.tool_id == end.tool_id), None
            )
        if seq is not None:
            del self._open[seq]
        return seq

    # -- The link -------------------------------------------------------------

    async def _send(self, frame: Frame) -> int:
        return await self._link.send(frame)

    async def _read_controls(self) -> None:
        """Apply the owner's control messages as they come."""
        while True:
            try:
                message = await self._link.receive()
            except TurnLinkError:
                self._cut_turn("owner_lost")
                return
            self._heard = True
            if isinstance(message, Pull):
                self._credits += message.count
            elif isinstance(message, Close):
                self._closing = True
            elif isinstance(message, Cancel):
                self._cut_turn(message.reason)
            self._wake.set()

    async def _beat(self) -> None:
        """Tell the owner the turn runs; cut it when the owner falls silent."""
        silent = 0
        while True:
            await heartbeat_tick(self._timeouts.heartbeat_interval)
            await self._send(Heartbeat())
            silent = 0 if self._heard else silent + 1
            self._heard = False
            if silent >= self._timeouts.silent_intervals:
                logger.warning("Owner of turn %s fell silent", self._request.turn_id)
                self._cut_turn("owner_lost")
                return

    @property
    def turn_id(self) -> str:
        return self._request.turn_id

    def spare(self) -> None:
        """Let the turn's work run on whatever ends its link: a close called
        from inside this turn must not cut the turn it runs under."""
        self._spared = True

    def _cut_turn(self, reason: str) -> None:
        if self._cut is None:
            self._cut = reason
        if not self._spared and self._work is not None and not self._work.done():
            self._work.cancel()
        self._wake.set()
