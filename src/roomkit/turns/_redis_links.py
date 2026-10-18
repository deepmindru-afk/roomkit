"""The two ends of a turn carried by Redis (RFC §10.6.3, §10.6.8).

Each end numbers what it sends and admits what it receives in order, as on
any backend: the messages are the protocol's envelopes, written to the other
process's inbox and routed to the turn's link by its id there.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, cast

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns.base import ExecutorLink, MessageOrder, TurnLink
from roomkit.turns.wire import ControlEnvelope, End, FrameEnvelope, TurnRequest

if TYPE_CHECKING:
    from roomkit.turns._redis_presence import LiveExecutors
    from roomkit.turns.wire import Control, Frame

logger = logging.getLogger("roomkit.turns")

Post = Callable[[str, str, str, str], Coroutine[Any, Any, None]]
"""Write to a node's inbox: the node, the kind, the body, the turn it is for."""

# Put on a link's queue when its turn is gone: its reader learns why.
_GONE = "\x00gone"


class _Inbound:
    """What reached one end of a turn, admitted in order."""

    def __init__(self, envelope: type[FrameEnvelope | ControlEnvelope]) -> None:
        self._envelope = envelope
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._order = MessageOrder()
        self._gone = "backend_closed"
        self.ended = False
        """Whether the other end's ``end`` came."""

    def deliver(self, body: str) -> None:
        self._queue.put_nowait(body)

    def end(self, reason: str) -> None:
        self._gone = reason
        self._queue.put_nowait(_GONE)

    async def next(self) -> tuple[int, Frame | Control]:
        while True:
            body = await self._queue.get()
            if body == _GONE:
                self._queue.put_nowait(_GONE)
                raise TurnLinkError(self._gone)
            envelope = self._envelope.model_validate_json(body)
            if self._order.admit(envelope.seq):
                if isinstance(envelope.message, End):
                    self.ended = True
                return envelope.seq, envelope.message


class _Outbound:
    """What one end of a turn sends: numbered once it serialized, so a
    message that cannot cross takes no number and is never a lost one."""

    def __init__(
        self, turn_id: str, envelope: type[FrameEnvelope | ControlEnvelope], post: Post
    ) -> None:
        self._turn_id = turn_id
        self._envelope = envelope
        self._post = post
        self._sent = 0
        self.open = True

    async def put(self, kind: str, node: str, message: Frame | Control) -> int:
        if not self.open:
            return self._sent
        body = self._envelope.model_validate(
            {"turn_id": self._turn_id, "seq": self._sent + 1, "message": message}
        ).model_dump_json()
        self._sent += 1
        await self._post(node, kind, body, self._turn_id)
        return self._sent


class RedisTurnLink(TurnLink):
    """The owner's end of a turn an executor in another process took."""

    def __init__(
        self,
        request: TurnRequest,
        executor_id: str,
        executor_node: str,
        post: Post,
        live: LiveExecutors,
        forget: Callable[[RedisTurnLink], None],
        spawn: Callable[[Coroutine[Any, Any, None]], None],
    ) -> None:
        self.request = request
        self.executor_id = executor_id
        self._node = executor_node
        self._frames = _Inbound(FrameEnvelope)
        self._controls = _Outbound(request.turn_id, ControlEnvelope, post)
        self._post = post
        self._live = live
        self._forget = forget
        self._spawn = spawn
        self._beat_seen = live.last_beat(executor_id)

    @property
    def taken(self) -> bool:
        # Placed on an executor that was alive: a silence is its loss.
        return True

    def deliver(self, body: str) -> None:
        self._frames.deliver(body)

    def end(self, reason: str) -> None:
        """The backend closed: the reader learns the turn is gone."""
        self._frames.end(reason)

    async def receive(self) -> tuple[int, Frame]:
        seq, frame = await self._frames.next()
        return seq, cast("Frame", frame)

    async def send(self, message: Control) -> None:
        await self._controls.put("control", self._node, message)

    def release(self) -> None:
        if not self._controls.open:
            return
        self._controls.open = False
        ended = self._frames.ended
        self._frames.end("backend_closed")
        self._forget(self)
        if not ended:
            # Given up before its end: the executor stops it, as one whose
            # owner went.
            self._spawn(self._tell_released())

    async def _tell_released(self) -> None:
        turn_id = self.request.turn_id
        try:
            await self._post(self._node, "released", turn_id, turn_id)
        except Exception:
            logger.debug("Telling the executor of turn %s it was released failed", turn_id)

    def beat_heard(self) -> bool:
        beat = self._live.last_beat(self.executor_id)
        if beat is None or beat == self._beat_seen:
            return False
        self._beat_seen = beat
        return True


class RedisExecutorLink(ExecutorLink):
    """The executor's end of a turn a room owner in another process submitted."""

    def __init__(
        self,
        request: TurnRequest,
        owner_node: str,
        post: Post,
        forget: Callable[[RedisExecutorLink], None],
    ) -> None:
        self.request = request
        self.owner_node = owner_node
        self._controls = _Inbound(ControlEnvelope)
        self._frames = _Outbound(request.turn_id, FrameEnvelope, post)
        self._forget = forget

    def deliver(self, body: str) -> None:
        self._controls.deliver(body)

    def end(self, reason: str) -> None:
        """The owner released the turn, or the backend closed."""
        self._frames.open = False
        self._controls.end(reason)

    async def receive(self) -> Control:
        _, message = await self._controls.next()
        return cast("Control", message)

    async def send(self, frame: Frame) -> int:
        seq = await self._frames.put("frame", self.owner_node, frame)
        if isinstance(frame, End):
            # The turn's last frame: nothing reaches it any more.
            self._frames.open = False
            self._forget(self)
        return seq
