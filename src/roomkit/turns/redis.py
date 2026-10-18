"""Turns carried between processes by Redis (RFC §10.6).

The room owner's process submits each turn to one executor, chosen from what
the executors alive declare (RFC §10.6.7): the one holding the room's state
for the channel while it lives, else one that takes it. A turn on an executor
of the owner's own process stays in the process, carried as the in-memory
backend carries it; one on an executor elsewhere crosses Redis, each message
the protocol's envelope written to the other process's inbox, in order, and
acknowledged where it must be: a request is taken or refused (and placed
again), a channel control is applied before the owner goes on. Each process
beats from a thread of its own, and an owner counts a turn's executor alive
while it beats and its process still runs the turn (RFC §10.6.8).

Requires ``pip install roomkit[redis]`` and a store every process shares
(RFC §10.6.9). Redis is the trust boundary: whoever writes to it can act as
any executor or owner, so restrict it (ACL, TLS) as the rooms it carries.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core.exceptions import TurnLinkError
from roomkit.core.task_utils import cancel_and_wait
from roomkit.turns._redis_beat import BeatPayload, BeatThread, LiveExecutors
from roomkit.turns._redis_inbox import Envelope, Inbox
from roomkit.turns._redis_outbox import Failed, Outbox
from roomkit.turns._redis_placement import Elsewhere, Placement
from roomkit.turns._redis_presence import Keys, Registry
from roomkit.turns._redis_routes import REFUSED, Routes
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
    from roomkit.turns._redis_links import RedisTurnLink
    from roomkit.turns.base import RoomOwner, TurnLink
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelControl, TurnRequest

logger = logging.getLogger("roomkit.turns")

_DEFAULT_URL = "redis://localhost:6379"
# The connections one process uses at most besides its inbox's: its writes
# go through one writer per destination, so few are in use at once.
_MAX_CONNECTIONS = 32
_POOL_WAIT = 20.0


class RedisTurnBackend(TurnBackend):
    """Turns carried between processes by Redis (RFC §10.6).

    One backend per process, given to the kit (``RoomKit(turn_backend=...)``)
    and to the executors that process serves (``await backend.serve(executor)``).
    An executor whose channels are given as a collection
    (``TurnExecutor(build, hosts={"meteo"})``) takes turns from every
    process; one given a predicate, as the kit's own, from its process only.

    Args:
        url: The Redis server (``redis://localhost:6379`` by default). With
            *client*, still required: the process's beat thread connects
            with it.
        client: A ``redis.asyncio`` client to use for the backend's writes
            and reads instead of connecting.
        key_prefix: The prefix of every key the backend writes.
        timeouts: The heartbeat bounds (RFC §10.6.8), kept by the kit given
            this backend: give every process, and every executor's
            ``TurnExecutor(timeouts=...)``, the same.

    Raises:
        ValueError: *client* without *url*.
    """

    def __init__(
        self,
        url: str | None = None,
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
        if client is not None and url is None:
            raise ValueError(
                "RedisTurnBackend(client=...) needs url= too: its beat connects with it"
            )
        address = url or _DEFAULT_URL
        self._owns_client = client is None
        self._client = client if client is not None else _pooled(_aioredis, address)
        self.node_id = uuid4().hex[:12]
        """This process's id among the backend's processes."""
        self._timeouts = timeouts or TurnTimeouts()
        self.timeouts = self._timeouts
        self._ttl_ms = int(self._timeouts.heartbeat_timeout * 3000)
        keys = Keys(key_prefix)
        self._outbox = Outbox(self._client, keys.inbox, self._ttl_ms)
        self._inbox = Inbox(
            _aioredis.from_url(address), keys.inbox(self.node_id), self._route, owns_client=True
        )
        self._registry = Registry(self._client, keys)
        self._live = LiveExecutors(
            self._registry, self._timeouts.heartbeat_interval, self._timeouts.heartbeat_timeout
        )
        self._beat = BeatThread(
            lambda: _redis.Redis.from_url(address),
            keys.beat(self.node_id),
            self._timeouts.heartbeat_interval,
            self._ttl_ms,
            lambda: self._payload(),
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
        self._payload = BeatPayload(self.node_id, self._routes)
        self._started = False
        self._closed = False

    async def submit(self, request: TurnRequest) -> TurnLink:
        self._start()
        refused: set[str] = set()
        while True:
            placed = await self._placement.place(
                request.channel_id, request.room_id, request.room_hooks, excluded=refused
            )
            if not isinstance(placed, Elsewhere):
                return self._here._submit_to(placed, request)
            answer, link = await self._hand(request, placed)
            if answer != REFUSED:
                return link
            # An executor that closed: the turn goes to another, never twice.
            self._routes.forget_owner_link(link)
            refused.add(placed.member)

    async def _hand(self, request: TurnRequest, placed: Elsewhere) -> tuple[str, RedisTurnLink]:
        """Hand *request* to *placed*'s process: its answer (taken or
        refused) and the owner's end of the turn. A process that does not
        answer within the accept bound has lost it: the turn is released
        there, never handed to another (it may have begun)."""
        link = self._routes.owner_link(request, placed.member, placed.executor_id, placed.node)
        body = request.model_dump_json()
        try:
            answer = await self._routes.ask(
                placed.node, "request", body, placed.executor_id, bound=self._timeouts.accept
            )
        except TimeoutError as exc:
            link.release()
            raise TurnLinkError(
                "executor_lost", f"Executor {placed.member} did not answer the turn"
            ) from exc
        except BaseException:
            link.release()
            raise
        return answer, link

    async def serve(self, executor: TurnExecutor) -> None:
        self._start()
        if executor not in self._placement.here:
            self._placement.here.append(executor)
        await self._here.serve(executor)
        if executor.timeouts != self._timeouts:
            logger.warning(
                "Executor %s keeps other timeouts than its turn backend: give both the same "
                "TurnTimeouts (RFC §10.6.8)",
                executor.executor_id,
            )
        if self._loop_back or executor.hosted is not None:
            self._routes.shared[executor.executor_id] = executor
        if executor.hosted is not None:
            if executor.store is None or executor.store.is_process_local:
                logger.warning(
                    "Executor %s takes turns from other processes without a store they "
                    "share (TurnExecutor(store=...)): its channels read the rooms there "
                    "(RFC §10.6.9)",
                    executor.executor_id,
                )
            self._payload.executors = (*self._payload.executors, executor)
            await self._beat_now()
        # Work a turn began reaches the owner of that turn, here or elsewhere.
        executor.reach_owner(self._routes.port)

    async def attend(self, owner: RoomOwner) -> None:
        self._start()
        await self._here.attend(owner)
        if self._routes.owner is None:
            self._routes.owner = owner
            await self._beat_now()

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        self._start()
        raw = ChannelControlEnvelope(message=message).model_dump_json()
        if isinstance(message, ChannelCall):
            placed = await self._placement.place(message.channel_id, message.room_id)
            return await self._apply(placed, raw)
        if isinstance(message, CommittedEvent):
            held = await self._placement.hold(
                message.channel_id, message.room_id, message.room_hooks
            )
            await self._apply(held, raw)
            return None
        if isinstance(message, RoomDetached):
            await self._placement.forget(message.channel_id, message.room_id)
        for here in [e for e in self._placement.here if e.declares(message.channel_id)]:
            await here.control(_crossed(raw))
        elsewhere = await self._placement.declaring_elsewhere(message.channel_id)
        results = await asyncio.gather(
            *(self._apply(executor, raw) for executor in elsewhere), return_exceptions=True
        )
        for executor, result in zip(elsewhere, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "Executor %s did not apply %s: %s", executor.member, message.type, result
                )
        return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._routes.close()
        try:
            await self._registry.withdraw(self._payload.members)
        except Exception:
            logger.debug("Withdrawing this process's executors failed", exc_info=True)
        await asyncio.to_thread(self._beat.stop)
        await self._live.close()
        await self._inbox.close()
        await self._outbox.close()
        await self._here.close()
        if self._tasks:
            await cancel_and_wait(*self._tasks, log_errors_to=logger)
        if self._owns_client:
            await self._client.aclose()

    async def _apply(self, placed: TurnExecutor | Elsewhere, raw: str) -> ChannelResult:
        """Apply a channel control on *placed*, here or across Redis: its
        result once applied, waited for while the executor lives."""
        if not isinstance(placed, Elsewhere):
            answer = await placed.control(_crossed(raw))
            if answer is None:
                return ChannelResult()
            return ChannelResult.model_validate_json(answer.model_dump_json())
        body = await self._routes.ask(
            placed.node, "channel", raw, placed.executor_id, alive=self._lives(placed)
        )
        return ChannelResult.model_validate_json(body)

    def _lives(self, placed: Elsewhere) -> Callable[[], bool]:
        """Whether *placed* still lives: by its beat, or (the tests' loop
        back) as an executor of this process."""
        if placed.node == self.node_id:
            local = self._routes.shared.get(placed.executor_id)
            return lambda: local is not None and not local.closed
        return lambda: self._live.alive(placed.member)

    def _start(self) -> None:
        if self._closed:
            raise TurnLinkError("backend_closed")
        if not self._started:
            self._started = True
            self._inbox.start()
            self._live.start()

    async def _beat_now(self) -> None:
        """Beat once from this task (alive from now), then from the thread."""
        await self._registry.beat(self.node_id, self._payload(), self._ttl_ms)
        self._beat.start()

    def _route(self, envelope: Envelope) -> None:
        self._routes.route(envelope)

    def _post(
        self,
        node: str,
        kind: str,
        body: str,
        target: str,
        ref: str = "",
        *,
        failed: Failed | None = None,
    ) -> None:
        self._outbox.post(node, Envelope(kind, self.node_id, body, target=target, ref=ref), failed)

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


def _pooled(aioredis: Any, url: str) -> Any:
    """A client whose callers wait for a free connection instead of failing
    when every one is in use."""
    pool = aioredis.BlockingConnectionPool.from_url(
        url, max_connections=_MAX_CONNECTIONS, timeout=_POOL_WAIT
    )
    return aioredis.Redis(connection_pool=pool)


def _crossed(raw: str) -> ChannelControl:
    return ChannelControlEnvelope.model_validate_json(raw).message
