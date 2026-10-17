"""The in-memory turn backend: the default, in the owner's process (RFC §10.6).

Every message is serialized to JSON and parsed on the other side, as any
backend carries it, so the in-memory path is the distributed one on another
transport. What it alone carries is the owner's execution context: a turn's
work runs in a copy of the context of the task that submitted it, and its
stream advances in a copy of the context of the task that first read it, so
host code that calls the kit from a turn finds the reentrancy guards it finds
in the owner's own process (RFC §10.6.5).
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns._rooms import RoomRecords
from roomkit.turns.base import ExecutorLink, MessageOrder, TurnBackend, TurnLink
from roomkit.turns.wire import (
    ChannelCall,
    ChannelControlEnvelope,
    ChannelResult,
    ControlEnvelope,
    FrameEnvelope,
    Pull,
    RoomDetached,
    TurnRequest,
)

if TYPE_CHECKING:
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelControl, Control, Frame

# Put on a pipe when the backend closes: whoever waits on it learns the turn
# will not end.
_CLOSED = "\x00closed"


class _Pipe:
    """One direction of a turn's link: numbered JSON envelopes on a queue."""

    def __init__(self, turn_id: str, envelope: type[FrameEnvelope | ControlEnvelope]) -> None:
        self._turn_id = turn_id
        self._envelope = envelope
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._order = MessageOrder()
        self._sent = 0
        self.open = True

    def put(self, message: Frame | Control) -> int:
        """Queue *message*; its sequence number. A closed pipe drops it."""
        if not self.open:
            return self._sent
        # Serialized before it is numbered: a message that cannot cross
        # raises here and takes no number, so it is never a lost frame.
        envelope = self._envelope.model_validate(
            {"turn_id": self._turn_id, "seq": self._sent + 1, "message": message}
        )
        raw = envelope.model_dump_json()
        self._sent += 1
        self._queue.put_nowait(raw)
        return self._sent

    async def get(self) -> tuple[int, Any]:
        """The next message in order, parsed anew (a frame or a control
        message, by the pipe's envelope)."""
        while True:
            raw = await self._queue.get()
            if raw == _CLOSED:
                self._queue.put_nowait(_CLOSED)
                raise TurnLinkError("backend_closed")
            envelope = self._envelope.model_validate_json(raw)
            if self._order.admit(envelope.seq):
                return envelope.seq, envelope.message

    def shut(self) -> None:
        """Stop the pipe: its waiters raise, later messages are dropped."""
        self.open = False
        self._queue.put_nowait(_CLOSED)


class _MemoryTurn:
    """What both sides of one in-memory turn share: its two pipes, the
    request as JSON, and the owner's execution contexts."""

    def __init__(self, request: TurnRequest) -> None:
        self.request_json = request.model_dump_json()
        self.channel_id = request.channel_id
        self.frames = _Pipe(request.turn_id, FrameEnvelope)
        self.controls = _Pipe(request.turn_id, ControlEnvelope)
        self.turn_context = contextvars.copy_context()
        self.stream_context: contextvars.Context | None = None
        self.taken = False

    def shut(self) -> None:
        self.frames.shut()
        self.controls.shut()


class _OwnerLink(TurnLink):
    def __init__(self, turn: _MemoryTurn, request: TurnRequest, backend: InMemoryTurnBackend):
        self._turn = turn
        self._backend = backend
        self.request = request

    @property
    def taken(self) -> bool:
        return self._turn.taken

    async def receive(self) -> tuple[int, Frame]:
        seq, frame = await self._turn.frames.get()
        return seq, frame

    async def send(self, message: Control) -> None:
        if isinstance(message, Pull) and self._turn.stream_context is None:
            # The task that first reads the stream lends it its context.
            self._turn.stream_context = contextvars.copy_context()
        self._turn.controls.put(message)

    def release(self) -> None:
        self._turn.shut()
        self._backend._forget(self._turn)


class _ExecutorLink(ExecutorLink):
    def __init__(self, turn: _MemoryTurn) -> None:
        self._turn = turn
        self.request = TurnRequest.model_validate_json(turn.request_json)

    async def receive(self) -> Control:
        _, message = await self._turn.controls.get()
        return message

    async def send(self, frame: Frame) -> int:
        return self._turn.frames.put(frame)

    def turn_context(self) -> contextvars.Context:
        return self._turn.turn_context

    def stream_context(self) -> contextvars.Context | None:
        return self._turn.stream_context


class InMemoryTurnBackend(TurnBackend):
    """Turns carried in this process, serialized like any other (RFC §10.6).

    A room's turns of a channel go to the executor that took its last turn
    or channel call while it serves and declares the channel, else to the
    first that declares it (RFC §10.6.7). Every executor is in this process,
    so a turn none declares fails at once with ``no_executor``.
    """

    def __init__(self) -> None:
        self._executors: list[TurnExecutor] = []
        # Where each room's state of a channel is: the executor that took its
        # last turn or channel call.
        self._holders: RoomRecords[TurnExecutor] = RoomRecords()
        self._turns: set[_MemoryTurn] = set()
        self._closed = False

    async def submit(self, request: TurnRequest) -> TurnLink:
        if self._closed:
            raise TurnLinkError("backend_closed")
        executor = self._place(request.channel_id, request.room_id)
        turn = _MemoryTurn(request)
        self._turns.add(turn)
        turn.taken = True
        executor.take(_ExecutorLink(turn))
        return _OwnerLink(turn, request, self)

    async def serve(self, executor: TurnExecutor) -> None:
        if executor not in self._executors:
            self._executors.append(executor)

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        if self._closed:
            raise TurnLinkError("backend_closed")
        raw = ChannelControlEnvelope(message=message).model_dump_json()
        if isinstance(message, ChannelCall):
            executor = self._place(message.channel_id, message.room_id)
            answer = await executor.control(_crossed(raw))
            if answer is None:
                return ChannelResult()
            return ChannelResult.model_validate_json(answer.model_dump_json())
        if isinstance(message, RoomDetached):
            self._holders.pop(message.channel_id, message.room_id)
        for executor in [e for e in self._executors if e.declares(message.channel_id)]:
            await executor.control(_crossed(raw))
        return None

    async def close(self) -> None:
        self._closed = True
        for turn in list(self._turns):
            turn.shut()
        self._turns.clear()

    def _place(self, channel_id: str, room_id: str) -> TurnExecutor:
        """The executor a room's turn or channel call of *channel_id* goes to,
        recorded as the one holding the room's state from now."""
        holder = self._holders.get(channel_id, room_id)
        if holder is None or not holder.declares(channel_id):
            holder = next((e for e in self._executors if e.declares(channel_id)), None)
        if holder is None:
            raise TurnLinkError("no_executor", f"No executor serves channel {channel_id!r}")
        self._holders.put(channel_id, room_id, holder)
        return holder

    def _forget(self, turn: _MemoryTurn) -> None:
        self._turns.discard(turn)


def _crossed(raw: str) -> ChannelControl:
    """A channel control message as the executor parses it."""
    return ChannelControlEnvelope.model_validate_json(raw).message
