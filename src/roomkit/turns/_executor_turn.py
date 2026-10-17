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
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, NoReturn

from roomkit.channels._discussion_turn import DISCUSSION_TURN, issue_mark, retire_mark
from roomkit.channels._strategy_turn import strategy_turn
from roomkit.channels._tool_registry import set_turn_owner_tools
from roomkit.core._turn_scope import enter_turn
from roomkit.core.exceptions import ChannelNotRegisteredError, TurnLinkError
from roomkit.models.streaming import ThinkingDeltaMarker, ToolCallEndMarker, ToolCallStartMarker
from roomkit.tools._turn_inputs import set_turn_inputs
from roomkit.turns._codec import (
    encode_end,
    encode_item,
    encode_output,
    error_record,
    record_payload,
)
from roomkit.turns._executor_calls import ExecutorCalls
from roomkit.turns._executor_inputs import ExecutorInputs
from roomkit.turns.base import heartbeat_tick
from roomkit.turns.host import TurnHost, set_turn_host
from roomkit.turns.wire import (
    Accepted,
    Cancel,
    Close,
    End,
    Error,
    Heartbeat,
    HostResult,
    InputAnswer,
    Item,
    Output,
    Pull,
    Record,
    Steer,
    SteerResult,
    ToolArgs,
    ToolEnded,
    ToolResult,
)

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.models.channel import ChannelOutput
    from roomkit.models.event import RoomEvent
    from roomkit.turns.base import ExecutorLink, TurnTimeouts
    from roomkit.turns.wire import Control, Frame, TurnRequest

logger = logging.getLogger("roomkit.turns")


class _UncarriedError(Exception):
    """A frame of the turn could not cross: the owner was told, the turn ends."""


async def _reaction(channel: Channel, event: RoomEvent, request: TurnRequest) -> ChannelOutput:
    """The channel's ``on_event``, told whether the turn is a strategy's
    generation (RFC §10.6.1)."""
    with strategy_turn(request.strategy_turn):
        return await channel.on_event(event, request.binding, request.context)


def _takes(function: Callable[..., Any], name: str) -> bool:
    """Whether *function* takes the keyword argument *name*."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


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
        self._built: Channel | None = None
        self._spared = False
        self._mark: dict[str, Any] | None = None
        self._record: dict[str, Any] = {}
        self._live_record: Any = None
        # The calls the stream started and has not ended, by their start's frame.
        self._open: dict[int, ToolCallStartMarker] = {}
        self._args_sent: dict[int, Any] = {}
        self._ends_sent: set[int] = set()
        # The calls the turn makes to its owner: the tools it serves, and
        # the verbs of the turn's host (RFC §10.6.5).
        self._calls = ExecutorCalls(self._request, self._send_call)
        self._host = TurnHost(self._calls.host_call)
        self._inputs = ExecutorInputs(self._send_call, self._calls.host_call)

    async def run(self) -> None:
        """Run the turn to its end, which the owner is always told."""
        watch = asyncio.create_task(self._read_controls())
        beat = asyncio.create_task(self._beat())
        try:
            await self._send(Accepted(executor_id=self._executor_id))
            await self._turn()
        except _UncarriedError:
            pass
        finally:
            # The turn is over: its host answers no more verbs, the calls on
            # their way are answered first, and the mark its notes rode reads
            # as no mark.
            self._host._close()
            task = asyncio.current_task()
            if self._cut is not None or (task is not None and task.cancelling()):
                # Cut: nothing waits for an owner that may no longer answer.
                self._calls.fail_all(TurnLinkError(self._cut or "cancelled"))
            await self._calls.settle()
            if self._mark is not None:
                retire_mark(self._mark)
            watch.cancel()
            beat.cancel()
            await asyncio.gather(watch, beat, return_exceptions=True)
            await self._send(End())

    # -- The turn -----------------------------------------------------------

    async def _turn(self) -> None:
        """``on_event``, its output, then its stream as the owner pulls it."""
        try:
            channel = await self._channel
        except Exception as exc:
            await self._send(Error(error=error_record(exc)))
            return
        self._built = channel
        if channel is None:
            missing = ChannelNotRegisteredError(
                f"No channel {self._request.channel_id!r} on executor {self._executor_id}"
            )
            await self._send(Error(error=error_record(missing)))
            return
        event = self._marked_event()
        try:
            output = await self._run_work(
                _reaction(channel, event, self._request), self._link.turn_context()
            )
        except asyncio.CancelledError:
            if self._cut is None:
                raise
            return
        except Exception as exc:
            await self._send(Error(error=error_record(exc)))
            return
        self._live_record = output.response_metadata
        stream = output.response_stream
        try:
            wire = encode_output(output)
        except (TypeError, ValueError) as exc:
            if stream is not None:
                await self._close(stream, contextvars.Context())
            await self._refuse("output", exc)
        self._record = wire.response_metadata
        await self._send(Output(output=wire, streams=stream is not None))
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
        """Advance *stream* as the owner pulls it, until its end, a close or a
        cut; a stream whose next frame cannot cross is closed."""
        context: contextvars.Context | None = None
        while True:
            try:
                await self._wait_for_pull()
            except asyncio.CancelledError:
                await self._close(stream, context or contextvars.Context())
                raise
            if context is None:
                context = self._link.stream_context() or contextvars.Context()
            if self._closing or self._cut is not None:
                await self._close(stream, context)
                return
            try:
                if not await self._step(stream, context):
                    return
            except _UncarriedError:
                await self._close(stream, context)
                raise
            except asyncio.CancelledError:
                # The executor closes: the channel's stream is closed, as a
                # collected generator's would be, before the cut goes on.
                await self._close(stream, context)
                raise

    async def _step(self, stream: AsyncIterator[Any], context: contextvars.Context) -> bool:
        """Read one item and send it after what changed before it; whether
        the stream goes on."""
        self._credits -= 1
        last: Frame | None = None
        try:
            item = await self._run_work(_next_item(stream), context)
        except StopAsyncIteration:
            last = Item(item={"kind": "stream_end"})
        except asyncio.CancelledError:
            if self._cut is None:
                raise
        except Exception as exc:
            last = Error(error=error_record(exc))
        else:
            await self._report_changes(before=item)
            await self._send_item(item)
            if not isinstance(item, str | ThinkingDeltaMarker):
                # Past a call's start, a round's break or the loop's end, only
                # a later pull advances the stream (RFC §10.6.4).
                self._credits = 0
            return True
        await self._report_changes()
        if last is not None:
            await self._send(last)
        return False

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
        context.run(enter_turn, self._request.turn_id)
        context.run(set_turn_owner_tools, self._calls.owner_tools)
        context.run(set_turn_host, self._host)
        context.run(set_turn_inputs, self._inputs)
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
            snapshot = record_payload(self._live_record)
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
        start = self._start_of(item) if isinstance(item, ToolCallEndMarker) else None
        try:
            data = encode_item(item, start=start)
        except TypeError as exc:
            await self._refuse("stream item", exc)
        seq = await self._send(Item(item=data))
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
        """Send *frame*; one that cannot cross is refused to the owner as an
        error naming it. An owner lost hears nothing more."""
        if self._cut == "owner_lost":
            return 0
        try:
            return await self._link.send(frame)
        except (TypeError, ValueError) as exc:
            if isinstance(frame, Error):
                logger.exception("Turn %s could not report its error", self._request.turn_id)
                raise _UncarriedError from exc
            await self._refuse(frame.type, exc)

    async def _send_call(self, frame: Frame) -> int:
        """Send a call of the turn to its owner; an owner lost answers none."""
        if self._cut in ("owner_lost", "frame_lost"):
            raise TurnLinkError(self._cut)
        return await self._link.send(frame)

    async def _refuse(self, what: str, exc: Exception) -> NoReturn:
        """Tell the owner the turn's *what* cannot cross the backend, and end the turn."""
        logger.warning(
            "Turn %s: its %s cannot cross the turn backend: %s", self.turn_id, what, exc
        )
        refusal = TypeError(f"The turn's {what} cannot cross the turn backend: {exc}")
        await self._send(Error(error=error_record(refusal)))
        raise _UncarriedError from exc

    async def _read_controls(self) -> None:
        """Apply the owner's control messages as they come."""
        while True:
            try:
                message = await self._link.receive()
            except TurnLinkError as lost:
                self._cut_turn("frame_lost" if lost.reason == "frame_lost" else "owner_lost")
                return
            self._heard = True
            try:
                await self._apply_control(message)
            except _UncarriedError:
                raise
            except Exception:
                # One message failing to apply never stops the turn's controls.
                logger.exception(
                    "Turn %s: a %s control failed to apply", self.turn_id, message.type
                )
            self._wake.set()

    async def _apply_control(self, message: Control) -> None:
        """Apply one control message of the owner's."""
        if isinstance(message, Pull):
            self._credits += message.count
        elif isinstance(message, Close):
            self._closing = True
        elif isinstance(message, Cancel):
            self._cut_turn(message.reason)
        elif isinstance(message, ToolResult | HostResult):
            self._calls.resolve(message)
        elif isinstance(message, InputAnswer):
            self._inputs.answer(message)
        elif isinstance(message, Steer):
            await self._send(SteerResult(steer_id=message.steer_id, loops=self._steer(message)))

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

    def _steer(self, message: Steer) -> int | None:
        """Apply a steering directive to this turn's loops: how many it
        reached; ``None`` for a channel without steering, or not built yet,
        whose owner cuts the turn on a ``Cancel`` (RFC §10.6.8)."""
        steer = getattr(self._built, "steer", None)
        if steer is None:
            return None
        # A steer() of the channel's own from before turns were addressed
        # takes the turn's room instead.
        address = (
            {"turn_id": self._request.turn_id}
            if _takes(steer, "turn_id")
            else {"room_id": self._request.room_id}
        )
        try:
            reached = steer(message.directive, **address)
        except Exception:
            logger.exception("Turn %s: the channel's steer() failed", self.turn_id)
            return 0
        return reached if isinstance(reached, int) else 0

    def spare(self) -> None:
        """Let the turn's work run on whatever ends its link: a close called
        from inside this turn must not cut the turn it runs under."""
        self._spared = True

    def _cut_turn(self, reason: str) -> None:
        if self._cut is None:
            self._cut = reason
        if reason in ("owner_lost", "frame_lost"):
            self._calls.fail_all(TurnLinkError(reason))
        if not self._spared and self._work is not None and not self._work.done():
            self._work.cancel(reason)
        self._wake.set()
