"""One turn on its room owner (RFC §10.6.4, §10.6.8).

The owner reads the turn's frames back into the objects the pipeline reads
locally: a ``ChannelOutput``, a stream whose every read pulls one item, the
turn's one response-metadata record updated in place, and call markers whose
fixed arguments and early ends are set on the owner's own copies. The turn is
watched while it is read: an executor silent too long has lost the turn.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import TurnLinkError
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.turns._codec import decode_end, decode_item, decode_output, rebuild_error
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
    from roomkit.models.channel import ChannelOutput
    from roomkit.models.response_metadata import ResponseMetadata
    from roomkit.turns.base import TurnLink, TurnTimeouts
    from roomkit.turns.wire import Control, Frame

logger = logging.getLogger("roomkit.turns")

# The closes of streams their reader dropped, held until they end.
_DROPPED_CLOSES: set[asyncio.Task[None]] = set()


def _cancel_reason(cancelled: asyncio.CancelledError) -> str:
    """The reason a cancellation names, when its canceller gave one."""
    reason = cancelled.args[0] if cancelled.args else None
    return reason if isinstance(reason, str) and reason else "cancelled"


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


class OwnerTurn:
    """The owner's side of one turn, from its submission to its end."""

    def __init__(self, link: TurnLink, timeouts: TurnTimeouts) -> None:
        self._link = link
        self._timeouts = timeouts
        self._accepted = False
        self._over = False
        self._record: ResponseMetadata | None = None
        self._starts: dict[int, ToolCallStartMarker] = {}
        self._early_ends: dict[int, ToolCallEndMarker] = {}
        self._beat = asyncio.get_running_loop().create_task(self._beat_while_open())

    @property
    def over(self) -> bool:
        """Whether the turn is over for the owner."""
        return self._over

    def abandon(self) -> None:
        """Stop holding the turn now (the owner closes): later frames are
        dropped, and the executor, no longer hearing its owner, cuts it."""
        self._finish()

    # -- Output and stream ---------------------------------------------------

    async def output(self) -> ChannelOutput:
        """The channel's output, its stream reading this turn when it has one."""
        try:
            frame = await self._next_main()
        except asyncio.CancelledError as cancelled:
            await self._cancel(_cancel_reason(cancelled))
            raise
        if isinstance(frame, Output):
            output = self._decoded(decode_output, frame.output)
            self._record = output.response_metadata
            if frame.streams:
                output.response_stream = RemoteStream(self)
            else:
                self._finish()
            return output
        if isinstance(frame, Error):
            self._finish()
            raise rebuild_error(frame.error)
        self._finish()
        raise TurnLinkError("executor_lost", "The turn ended before its output")

    async def read_item(self) -> Any:
        """The stream's next item; ``StopAsyncIteration`` at its end."""
        await self._link.send(Pull())
        try:
            seq, frame = await self._next_main_with_seq()
        except asyncio.CancelledError as cancelled:
            await self._cancel(_cancel_reason(cancelled))
            raise
        if isinstance(frame, Item):
            if frame.item.get("kind") == "stream_end":
                await self._drain_to_end(None)
                raise StopAsyncIteration
            return self._decoded(self._item, seq, frame.item)
        if isinstance(frame, Error):
            await self._drain_to_end(None)
            raise rebuild_error(frame.error)
        self._finish()
        raise TurnLinkError("executor_lost", "The turn ended in the middle of its stream")

    async def close_stream(self) -> None:
        """Close the stream: the executor closes the channel's, a running
        call let finish (RFC §12.2 step 13s), and its end is waited for."""
        if self._over:
            return
        await self._link.send(Close())
        await self._drain_to_end(None)

    def _decoded(self, decode: Any, *args: Any) -> Any:
        """What *decode* builds of a frame's payload; a payload it cannot
        read fails the turn as an invalid frame."""
        try:
            return decode(*args)
        except (KeyError, TypeError, ValueError) as exc:
            self._finish()
            raise TurnLinkError(
                "frame_invalid", f"A frame of the turn could not be read: {exc}"
            ) from exc

    def _item(self, seq: int, data: dict[str, Any]) -> Any:
        """A new item, a call's end the same object as its start's ``ended``."""
        if data["kind"] != "tool_end":
            item = decode_item(data)
            if isinstance(item, ToolCallStartMarker):
                self._starts[seq] = item
            return item
        start_seq = data.get("start")
        end = self._early_ends.pop(start_seq, None) if start_seq is not None else None
        if end is None:
            end = decode_end(data)
        start = self._starts.pop(start_seq, None) if start_seq is not None else None
        if start is not None:
            start.ended = end
        return end

    # -- Frames ---------------------------------------------------------------

    async def _next_main(self) -> Frame:
        _, frame = await self._next_main_with_seq()
        return frame

    async def _next_main_with_seq(self) -> tuple[int, Frame]:
        """The next output, item, error or end; the frames that change what
        the owner holds applied on the way; the executor watched."""
        silent = 0
        while True:
            try:
                seq, frame = await asyncio.wait_for(
                    self._link.receive(), self._timeouts.heartbeat_interval
                )
            except TimeoutError:
                silent += 1
                if silent >= self._silence_limit():
                    raise await self._lost() from None
                continue
            except TurnLinkError:
                self._finish()
                raise
            silent = 0
            if not self._apply(seq, frame):
                return seq, frame

    def _apply(self, seq: int, frame: Frame) -> bool:
        """Apply a frame that changes what the owner holds; whether it was one."""
        if isinstance(frame, Accepted | Heartbeat):
            self._accepted = True
        elif isinstance(frame, Record):
            if self._record is not None:
                self._record.clear()
                self._record.update(frame.record)
        elif isinstance(frame, ToolArgs):
            if (start := self._starts.get(frame.start)) is not None:
                start.ran_with = frame.arguments
        elif isinstance(frame, ToolEnded):
            end = self._decoded(decode_end, frame.end)
            self._early_ends[frame.start] = end
            if (start := self._starts.get(frame.start)) is not None:
                start.ended = end
        else:
            return False
        return True

    def _silence_limit(self) -> int:
        """How many silent intervals lose the turn: the accept bound until an
        executor took it, the heartbeat bound after."""
        if self._accepted:
            return self._timeouts.silent_intervals
        return max(1, round(self._timeouts.accept / self._timeouts.heartbeat_interval))

    async def _lost(self) -> TurnLinkError:
        reason = "executor_lost" if self._link.taken or self._accepted else "no_executor"
        logger.warning("Turn %s failed: %s", self._link.request.turn_id, reason)
        await self._send_best_effort(Cancel(reason=reason))
        self._finish()
        return TurnLinkError(reason)

    # -- Ending ---------------------------------------------------------------

    async def _cancel(self, reason: str) -> None:
        """Cut the turn and wait for its end within the cancel bound; a
        second cancellation does not cut the wait short."""
        if self._over:
            return
        await self._send_best_effort(Cancel(reason=reason))
        drain = asyncio.get_running_loop().create_task(self._drain_to_end(self._timeouts.cancel))
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                continue
            except Exception:
                break

    async def _drain_to_end(self, bound: float | None) -> None:
        """Read the turn's last frames until its end, applying them; past
        *bound*, report incomplete drainage and drop the rest."""
        try:
            async with asyncio.timeout(bound):
                while not self._over:
                    frame = await self._next_main()
                    if isinstance(frame, End):
                        break
        except TimeoutError:
            logger.warning(
                "Turn %s did not end within %ss of its cancellation: drainage incomplete",
                self._link.request.turn_id,
                bound,
            )
        except TurnLinkError:
            pass
        finally:
            self._finish()

    def _finish(self) -> None:
        """The turn is over for the owner: stop beating, drop later frames."""
        if self._over:
            return
        self._over = True
        self._beat.cancel()
        self._link.release()

    async def _send_best_effort(self, message: Control) -> None:
        try:
            await self._link.send(message)
        except Exception:
            logger.debug("Could not send %s to turn %s", message.type, self._link.request.turn_id)

    async def _beat_while_open(self) -> None:
        """Tell the executor its owner still holds the turn."""
        while not self._over:
            await heartbeat_tick(self._timeouts.heartbeat_interval)
            if not self._over:
                await self._send_best_effort(Heartbeat())


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
