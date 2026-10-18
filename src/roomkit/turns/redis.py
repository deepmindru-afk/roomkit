"""Turns carried between processes by Redis (RFC §10.6).

The room owner's process submits each turn to one executor, chosen from what
the executors alive declare (RFC §10.6.7): the one holding the room's state
for the channel while it lives, else one that takes it. A turn on an executor
of the owner's own process stays in the process, carried as the in-memory
backend carries it; one on an executor elsewhere crosses Redis, each message
the protocol's envelope written to the other process's inbox. The executors
of a process beat from a thread of their own, and the owner counts those
beats with the turn's messages (RFC §10.6.8).

Requires ``pip install roomkit[redis]`` and a store every process shares
(RFC §10.6.9). Redis is the trust boundary: whoever writes to it can act as
any executor or owner, so restrict it (ACL, TLS) as the rooms it carries.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core.exceptions import TurnLinkError
from roomkit.core.task_utils import cancel_and_wait
from roomkit.turns._redis_inbox import Envelope, Inbox
from roomkit.turns._redis_placement import Elsewhere, Placement
from roomkit.turns._redis_presence import BeatThread, Declaration, LiveExecutors, Registry
from roomkit.turns._redis_routes import Routes
from roomkit.turns.base import TurnBackend, TurnTimeouts
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.turns.wire import (
    ChannelCall,
    ChannelControlEnvelope,
    ChannelResult,
    CommittedEvent,
    RoomDetached,
)

if TYPE_CHECKING:
    from roomkit.turns.base import RoomOwner, TurnLink
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelControl, TurnRequest

logger = logging.getLogger("roomkit.turns")


class RedisTurnBackend(TurnBackend):
    """Turns carried between processes by Redis (RFC §10.6).

    One backend per process, given to the kit (``RoomKit(turn_backend=...)``)
    and to the executors that process serves (``await backend.serve(executor)``).
    An executor whose channels are given as a collection
    (``TurnExecutor(build, hosts={"meteo"})``) takes turns from every
    process; one given a predicate, as the kit's own, from its process only.

    Args:
        url: The Redis server; the thread an executor beats from connects
            to it too.
        client: A ``redis.asyncio`` client to use instead of connecting.
        key_prefix: The prefix of every key the backend writes.
        timeouts: The heartbeat bounds (RFC §10.6.8): the kit and the
            executors served here keep them, and every process must agree.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        client: Any = None,
        key_prefix: str = "roomkit:turns",
        timeouts: TurnTimeouts | None = None,
        _loop_back: bool = False,
    ) -> None:
        try:
            import redis as _redis
            import redis.asyncio as _aioredis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisTurnBackend. "
                "Install it with: pip install roomkit[redis]"
            ) from exc
        self._owns_client = client is None
        self._client = client if client is not None else _aioredis.from_url(url)
        self.node_id = uuid4().hex[:12]
        """This process's id among the backend's processes."""
        self.timeouts = timeouts or TurnTimeouts()
        self._inbox_ttl_ms = int(self.timeouts.heartbeat_timeout * 3000)
        self._inbox = Inbox(
            self._client, key_prefix, self.node_id, self._route, self._inbox_ttl_ms
        )
        self._registry = Registry(self._client, key_prefix)
        self._live = LiveExecutors(
            self._registry, self.timeouts.heartbeat_interval, self.timeouts.heartbeat_timeout
        )
        self._beat = BeatThread(
            lambda: _redis.Redis.from_url(url),
            self._registry.alive_key,
            self._inbox.key(self.node_id),
            self.timeouts.heartbeat_interval,
            self._inbox_ttl_ms,
        )
        # Turns on executors of this process are carried in it (RFC §10.6.7);
        # the tests' loop back sends them across Redis too.
        self._here = InMemoryTurnBackend()
        self._loop_back = _loop_back
        self._placement = Placement(
            self._client, key_prefix, self.node_id, self._live, loop_back=_loop_back
        )
        self._tasks: set[asyncio.Task[Any]] = set()
        self._routes = Routes(self.node_id, self._post, self._spawn, self._live, self._here._port)
        self._started = False
        self._closed = False

    async def submit(self, request: TurnRequest) -> TurnLink:
        self._start()
        placed = await self._placement.place(
            request.channel_id, request.room_id, request.room_hooks
        )
        if not isinstance(placed, Elsewhere):
            return self._here._submit_to(placed, request)
        link = self._routes.owner_link(request, placed.executor_id, placed.node)
        await self._post(placed.node, "request", request.model_dump_json(), placed.executor_id)
        return link

    async def serve(self, executor: TurnExecutor) -> None:
        self._start()
        if executor not in self._placement.here:
            self._placement.here.append(executor)
        await self._here.serve(executor)
        if self._loop_back:
            self._routes.shared[executor.executor_id] = executor
        if executor.hosted is not None:
            await self._share(executor)
        # Work a turn began reaches the owner of that turn, here or elsewhere.
        executor.reach_owner(self._routes.port)

    async def attend(self, owner: RoomOwner) -> None:
        self._start()
        await self._here.attend(owner)
        if self._routes.owner is None:
            self._routes.owner = owner

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        self._start()
        raw = ChannelControlEnvelope(message=message).model_dump_json()
        if isinstance(message, ChannelCall):
            placed = await self._placement.place(message.channel_id, message.room_id)
            return await self._channel_call(placed, raw)
        if isinstance(message, CommittedEvent):
            held = await self._placement.hold(
                message.channel_id, message.room_id, message.room_hooks
            )
            await self._carry(held, raw)
            return None
        if isinstance(message, RoomDetached):
            await self._placement.forget(message.channel_id, message.room_id)
        for here in [e for e in self._placement.here if e.declares(message.channel_id)]:
            await here.control(_crossed(raw))
        for elsewhere in await self._placement.declaring_elsewhere(message.channel_id):
            await self._post(elsewhere.node, "channel", raw, elsewhere.executor_id)
        return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._routes.close()
        for executor_id in list(self._routes.shared):
            try:
                await self._registry.withdraw(executor_id)
            except Exception:
                logger.debug("Withdrawing executor %s failed", executor_id, exc_info=True)
        await asyncio.to_thread(self._beat.stop)
        await self._live.close()
        await self._inbox.close()
        await self._here.close()
        if self._tasks:
            await cancel_and_wait(*self._tasks, log_errors_to=logger)
        if self._owns_client:
            await self._client.aclose()

    async def _share(self, executor: TurnExecutor) -> None:
        """Have other processes reach *executor*: declared, then alive from
        its first beat, which this task makes; the thread makes the others."""
        assert executor.hosted is not None
        if executor.store is None or executor.store.is_process_local:
            logger.warning(
                "Executor %s takes turns from other processes without a store they "
                "share (TurnExecutor(store=...)): its channels read the rooms there "
                "(RFC §10.6.9)",
                executor.executor_id,
            )
        self._routes.shared[executor.executor_id] = executor
        declaration = Declaration(
            self.node_id, executor.hosted, executor.hooks.defined_room_hooks()
        )
        await self._registry.declare(executor.executor_id, declaration)
        await self._registry.beat(
            self._routes.shared, self._inbox.key(self.node_id), self._inbox_ttl_ms
        )
        self._beat.beat_for(self._routes.shared)

    async def _channel_call(self, placed: TurnExecutor | Elsewhere, raw: str) -> ChannelResult:
        if not isinstance(placed, Elsewhere):
            answer = await placed.control(_crossed(raw))
            if answer is None:
                return ChannelResult()
            return ChannelResult.model_validate_json(answer.model_dump_json())
        body = await self._routes.ask(
            placed.node, "channel", raw, placed.executor_id, while_alive=placed.executor_id
        )
        return ChannelResult.model_validate_json(body)

    async def _carry(self, placed: TurnExecutor | Elsewhere, raw: str) -> None:
        if not isinstance(placed, Elsewhere):
            await placed.control(_crossed(raw))
            return
        await self._post(placed.node, "channel", raw, placed.executor_id)

    def _start(self) -> None:
        if self._closed:
            raise TurnLinkError("backend_closed")
        if not self._started:
            self._started = True
            self._inbox.start()
            self._live.start()

    def _route(self, envelope: Envelope) -> None:
        self._routes.route(envelope)

    async def _post(self, node: str, kind: str, body: str, target: str, ref: str = "") -> None:
        await self._inbox.send(node, Envelope(kind, self.node_id, body, target=target, ref=ref))

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        """Run *work* on a task the backend holds until it closes."""
        if self._closed:
            work.close()
            return
        task = asyncio.get_running_loop().create_task(work, context=contextvars.Context())
        self._tasks.add(task)
        task.add_done_callback(self._settled)

    def _settled(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("A turn backend message failed", exc_info=error)


def _crossed(raw: str) -> ChannelControl:
    return ChannelControlEnvelope.model_validate_json(raw).message
