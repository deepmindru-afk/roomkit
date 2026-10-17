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
from collections import deque
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns.base import ExecutorLink, MessageOrder, TurnBackend, TurnLink
from roomkit.turns.wire import ControlEnvelope, FrameEnvelope, Pull, TurnRequest

if TYPE_CHECKING:
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import Control, Frame

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
        self._sent += 1
        envelope = self._envelope.model_validate(
            {"turn_id": self._turn_id, "seq": self._sent, "message": message}
        )
        self._queue.put_nowait(envelope.model_dump_json())
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

    A turn no executor declares waits for one to serve; the owner fails it
    once its accept timeout passes.
    """

    def __init__(self) -> None:
        self._executors: list[TurnExecutor] = []
        self._waiting: deque[_MemoryTurn] = deque()
        self._turns: set[_MemoryTurn] = set()
        self._closed = False

    async def submit(self, request: TurnRequest) -> TurnLink:
        if self._closed:
            raise TurnLinkError("backend_closed")
        turn = _MemoryTurn(request)
        self._turns.add(turn)
        if not self._hand_over(turn):
            self._waiting.append(turn)
        return _OwnerLink(turn, request, self)

    async def serve(self, executor: TurnExecutor) -> None:
        if executor not in self._executors:
            self._executors.append(executor)
        waiting, self._waiting = self._waiting, deque()
        for turn in waiting:
            if not self._hand_over(turn):
                self._waiting.append(turn)

    async def close(self) -> None:
        self._closed = True
        for turn in list(self._turns):
            turn.shut()
        self._turns.clear()
        self._waiting.clear()

    def _hand_over(self, turn: _MemoryTurn) -> bool:
        """Give *turn* to an executor that declares its channel, if one does."""
        link = _ExecutorLink(turn)
        executor = next((e for e in self._executors if e.declares(link.request.channel_id)), None)
        if executor is None:
            return False
        turn.taken = True
        executor.take(link)
        return True

    def _forget(self, turn: _MemoryTurn) -> None:
        self._turns.discard(turn)
        if turn in self._waiting:
            self._waiting.remove(turn)
