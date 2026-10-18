"""What reaches a process's inbox on the Redis turn backend, and where it goes.

Each kind of entry has its route: a frame to the owner's end of its turn, a
control message to the executor's, a request to the executor it names, a
channel control to that executor, an answer to the question waiting for it,
and the work a turn began after it ended to the room owner that submitted the
turn (RFC §10.6.3), whichever process that owner runs in.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns._codec import error_record
from roomkit.turns._redis_links import RedisExecutorLink, RedisTurnLink
from roomkit.turns.base import RoomOwner
from roomkit.turns.wire import (
    ChannelControlEnvelope,
    ChannelResult,
    End,
    Error,
    FrameEnvelope,
    HostResult,
    OwnerMessageEnvelope,
    RoomCall,
    RoomPublication,
    TurnRequest,
)

if TYPE_CHECKING:
    from roomkit.turns._redis_inbox import Envelope
    from roomkit.turns._redis_presence import LiveExecutors
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelControl

logger = logging.getLogger("roomkit.turns")

Post = Callable[..., Coroutine[Any, Any, None]]

# How many turns' owners are remembered for the work they began (RFC §10.6.3).
_TURN_OWNERS = 10_000
# How long an executor waits for its room owner to answer a verb made after
# its turn: owners beat in no registry.
_OWNER_ANSWER_TIMEOUT = 30.0


class Routes:
    """The ends of this process's turns, what it waits for, and the routes
    of what its inbox receives.

    Args:
        node: This process's node id.
        post: Writes to a node's inbox: ``post(node, kind, body, target, ref)``.
        spawn: Runs a coroutine on a task the backend holds.
        live: The executors alive, as the owner reads them.
        local_owner: What the owner of this process is reached by for a turn
            it submitted to an executor here.
    """

    def __init__(
        self,
        node: str,
        post: Post,
        spawn: Callable[[Coroutine[Any, Any, None]], None],
        live: LiveExecutors,
        local_owner: RoomOwner,
    ) -> None:
        self._node = node
        self._post = post
        self._spawn = spawn
        self._live = live
        self._local_owner = local_owner
        self.shared: dict[str, TurnExecutor] = {}
        """The executors of this process other processes reach, by id."""
        self.owner: RoomOwner | None = None
        """The room owner of this process, once it attends."""
        self._owner_links: dict[str, RedisTurnLink] = {}
        self._executor_links: dict[str, RedisExecutorLink] = {}
        self._answers: dict[str, asyncio.Future[str]] = {}
        self._turn_owners: OrderedDict[str, str] = OrderedDict()
        self._routes: dict[str, Callable[[Envelope], None]] = {
            "frame": self._frame,
            "control": self._control,
            "released": self._released,
            "request": self._request,
            "channel": lambda envelope: self._spawn(self._channel(envelope)),
            "owner": lambda envelope: self._spawn(self._owner_message(envelope)),
            "channel_result": self._answer,
            "host_result": self._answer,
        }
        self.port = _OwnerPort(self)
        """What this process's executors reach a room owner by outside a turn."""

    def route(self, envelope: Envelope) -> None:
        """Hand one inbox entry on, in order, without waiting."""
        route = self._routes.get(envelope.kind)
        if route is None:
            logger.warning("A turn inbox entry of an unknown kind %r was dropped", envelope.kind)
            return
        route(envelope)

    def owner_link(self, request: TurnRequest, executor_id: str, node: str) -> RedisTurnLink:
        """The owner's end of a turn submitted to *executor_id* at *node*."""
        link = RedisTurnLink(
            request,
            executor_id,
            node,
            self._post,
            self._live,
            self._forget_owner_link,
            self._spawn,
        )
        self._owner_links[request.turn_id] = link
        return link

    async def ask(
        self, node: str, kind: str, body: str, target: str, *, while_alive: str | None = None
    ) -> str:
        """Post a question to *node* and wait for its answer's body: while the
        executor *while_alive* beats, else within the owner answer bound."""
        ref = uuid4().hex
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._answers[ref] = answer
        try:
            await self._post(node, kind, body, target, ref)
            if while_alive is None:
                return await asyncio.wait_for(answer, _OWNER_ANSWER_TIMEOUT)
            return await self._answered_while_alive(answer, while_alive)
        finally:
            self._answers.pop(ref, None)

    def close(self) -> None:
        """The backend closes: every end of a turn and every question learns it."""
        for link in list(self._owner_links.values()):
            link.end("backend_closed")
        for executor_link in list(self._executor_links.values()):
            executor_link.end("backend_closed")
        for answer in self._answers.values():
            if not answer.done():
                answer.set_exception(TurnLinkError("backend_closed"))

    def turn_owner(self, turn_id: str) -> str | None:
        """The node of the owner that submitted *turn_id* to an executor here."""
        return self._turn_owners.get(turn_id)

    def _forget_owner_link(self, link: RedisTurnLink) -> None:
        self._owner_links.pop(link.request.turn_id, None)

    def _forget_executor_link(self, link: RedisExecutorLink) -> None:
        self._executor_links.pop(link.request.turn_id, None)

    async def _answered_while_alive(self, answer: asyncio.Future[str], executor_id: str) -> str:
        while True:
            done, _ = await asyncio.wait({answer}, timeout=self._live.interval)
            if done:
                return answer.result()
            if executor_id not in await self._live.declared():
                raise TurnLinkError("executor_lost", f"Executor {executor_id} stopped beating")

    # -- Routes ---------------------------------------------------------------------

    def _frame(self, envelope: Envelope) -> None:
        if (link := self._owner_links.get(envelope.target)) is not None:
            link.deliver(envelope.body)

    def _control(self, envelope: Envelope) -> None:
        if (link := self._executor_links.get(envelope.target)) is not None:
            link.deliver(envelope.body)

    def _released(self, envelope: Envelope) -> None:
        if (link := self._executor_links.get(envelope.target)) is not None:
            link.end("backend_closed")

    def _answer(self, envelope: Envelope) -> None:
        if (answer := self._answers.get(envelope.ref)) is not None and not answer.done():
            answer.set_result(envelope.body)

    def _request(self, envelope: Envelope) -> None:
        """A request of an owner elsewhere, for an executor of this process; one
        that no longer serves answers at once, as no executor would."""
        request = TurnRequest.model_validate_json(envelope.body)
        executor = self.shared.get(envelope.target)
        if executor is None or executor.closed:
            self._spawn(self._refuse(request, envelope.sender, envelope.target))
            return
        link = RedisExecutorLink(
            request,
            envelope.sender,
            self._post,
            self._forget_executor_link,
        )
        self._executor_links[request.turn_id] = link
        self._turn_owners[request.turn_id] = envelope.sender
        while len(self._turn_owners) > _TURN_OWNERS:
            self._turn_owners.popitem(last=False)
        executor.take(link)

    async def _refuse(self, request: TurnRequest, owner_node: str, executor_id: str) -> None:
        refusal = TurnLinkError("no_executor", f"Executor {executor_id} no longer serves")
        for seq, frame in enumerate((Error(error=error_record(refusal)), End()), start=1):
            body = FrameEnvelope(turn_id=request.turn_id, seq=seq, message=frame)
            await self._post(owner_node, "frame", body.model_dump_json(), request.turn_id)

    async def _channel(self, envelope: Envelope) -> None:
        executor = self.shared.get(envelope.target)
        if executor is None:
            logger.warning(
                "A channel control came for executor %s, which no longer serves", envelope.target
            )
            return
        answer = await executor.control(_crossed(envelope.body))
        if envelope.ref:
            result = answer if answer is not None else ChannelResult()
            await self._post(
                envelope.sender, "channel_result", result.model_dump_json(), "", envelope.ref
            )

    async def _owner_message(self, envelope: Envelope) -> None:
        """Work an executor elsewhere began in a turn of this owner (RFC §10.6.3)."""
        message = OwnerMessageEnvelope.model_validate_json(envelope.body).message
        owner = self.owner
        if isinstance(message, RoomPublication):
            if owner is not None:
                await _apart(owner.room_publication(message))
            return
        if owner is None:
            refusal = RuntimeError("No room owner is reached outside a turn on this backend")
            answer = HostResult(call_id=message.call_id, error=error_record(refusal))
        else:
            answer = await _apart(owner.room_call(message))
        await self._post(
            envelope.sender, "host_result", answer.model_dump_json(), "", envelope.ref
        )


class _OwnerPort(RoomOwner):
    """What an executor served here reaches a room owner by outside a turn:
    the owner that submitted the turn the work began in, here or elsewhere."""

    def __init__(self, routes: Routes) -> None:
        self._routes = routes

    async def room_call(self, message: RoomCall) -> HostResult:
        node = self._routes.turn_owner(message.turn_id)
        if node is None:
            return await self._routes._local_owner.room_call(message)
        body = OwnerMessageEnvelope(message=message).model_dump_json()
        try:
            answer = await self._routes.ask(node, "owner", body, "")
        except (TimeoutError, TurnLinkError) as exc:
            return HostResult(call_id=message.call_id, error=error_record(exc))
        return HostResult.model_validate_json(answer)

    async def room_publication(self, message: RoomPublication) -> None:
        node = self._routes.turn_owner(message.turn_id)
        if node is None:
            await self._routes._local_owner.room_publication(message)
            return
        body = OwnerMessageEnvelope(message=message).model_dump_json()
        await self._routes._post(node, "owner", body, "")


def _crossed(raw: str) -> ChannelControl:
    return ChannelControlEnvelope.model_validate_json(raw).message


async def _apart[T](work: Coroutine[Any, Any, T]) -> T:
    """Run *work* in a context of its own: the owner's side shares no context
    variable with the message's arrival."""
    return await asyncio.get_running_loop().create_task(work, context=contextvars.Context())
