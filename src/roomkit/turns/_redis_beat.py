"""How a process of the Redis turn backend beats, and reads the others' beats
(RFC §10.6.8).

The beat runs on a thread of its own with a connection of its own, so a turn
that keeps the event loop busy never stalls it; what it writes is built from
what the event loop replaces whole and never changes in place. The view of
the others is read again each heartbeat interval on the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.turns._redis_presence import BEAT_SCRIPT, Declaration, Registry, View, member_of

if TYPE_CHECKING:
    from roomkit.turns._redis_routes import Routes
    from roomkit.turns.executor import TurnExecutor

logger = logging.getLogger("roomkit.turns")


class BeatPayload:
    """What this process's next beat writes: what each executor it serves
    declares (its room hooks read live), the ones that closed, the turns it
    runs for owners elsewhere, and whether it is a room owner."""

    def __init__(self, node: str, routes: Routes) -> None:
        self._node = node
        self._routes = routes
        self.executors: tuple[TurnExecutor, ...] = ()
        """The executors it beats for: replaced, never changed."""
        self._declared: dict[str, str] = {}

    @property
    def members(self) -> list[str]:
        return [member_of(self._node, executor.executor_id) for executor in self.executors]

    def __call__(self) -> str:
        declared: dict[str, str] = {}
        gone: list[str] = []
        for executor in self.executors:
            member = member_of(self._node, executor.executor_id)
            if executor.closed:
                gone.append(member)
            else:
                declared[member] = self._declaration(member, executor)
        payload = {
            "node": self._node,
            "declared": declared,
            "gone": gone,
            "running": sorted(self._routes.running),
            "owner": self._routes.owner is not None,
        }
        return json.dumps(payload)

    def _declaration(self, member: str, executor: TurnExecutor) -> str:
        """*executor*'s declaration now, the last one kept when the event
        loop changed its room hooks while this thread read them."""
        assert executor.hosted is not None
        try:
            hooks = executor.hooks.defined_room_hooks()
        except RuntimeError:
            return self._declared.get(member, "")
        dumped = Declaration(self._node, executor.executor_id, executor.hosted, hooks).dump()
        self._declared[member] = dumped
        return dumped


class LiveExecutors:
    """A process's view of the processes alive, read each heartbeat interval:
    the executors and what they declare, the turns each process runs, the
    owners alive."""

    def __init__(self, registry: Registry, interval: float, timeout: float) -> None:
        self._registry = registry
        self.interval = interval
        """How often the view is read again."""
        self._timeout_ms = int(timeout * 1000)
        self._view = View()
        self._refresher: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()

    def start(self) -> None:
        if self._refresher is None:
            self._refresher = asyncio.get_running_loop().create_task(
                self._refresh_each_interval(), name="roomkit-turns-live-executors"
            )

    async def current(self) -> dict[str, Declaration]:
        """The executors alive and what they declare, by member, read now."""
        await self.refresh()
        return self._alive_declared()

    async def declared(self) -> dict[str, Declaration]:
        """The executors alive and what they declare, as last read (read now
        the first time)."""
        if not self._ready.is_set():
            await self.refresh()
        return self._alive_declared()

    def last_beat(self, member: str) -> int | None:
        """The server time of *member*'s last beat seen, in ms."""
        return self._view.beats.get(member)

    def alive(self, member: str) -> bool:
        """Whether *member* beat within the timeout, as last read."""
        beat = self._view.beats.get(member)
        return beat is not None and self._view.now - beat < self._timeout_ms

    def runs(self, node: str, turn_id: str) -> bool:
        """Whether *node* still ran *turn_id* when it last beat."""
        return turn_id in self._view.running.get(node, frozenset())

    def owner_alive(self, node: str) -> bool:
        """Whether the owner process *node* beat within the timeout."""
        beat = self._view.owners.get(node)
        return beat is not None and self._view.now - beat < self._timeout_ms

    async def refresh(self) -> None:
        self._view = await self._registry.view(forget_after_ms=10 * self._timeout_ms)
        self._ready.set()

    async def close(self) -> None:
        if self._refresher is not None:
            self._refresher.cancel()
            await asyncio.gather(self._refresher, return_exceptions=True)
            self._refresher = None

    def _alive_declared(self) -> dict[str, Declaration]:
        view = self._view
        return {
            member: declared
            for member, declared in view.declared.items()
            if view.now - view.beats.get(member, 0) < self._timeout_ms
        }

    async def _refresh_each_interval(self) -> None:
        while True:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Reading the turn processes alive failed", exc_info=True)
            await asyncio.sleep(self.interval)


class BeatThread:
    """Beats for this process from a thread of its own, on a connection of
    its own (RFC §10.6.8): each beat is what *payload* builds then."""

    def __init__(
        self,
        connect: Callable[[], Any],
        keys: list[str],
        interval: float,
        ttl_ms: int,
        payload: Callable[[], str],
    ) -> None:
        self._connect = connect
        self._keys = keys
        self._interval = interval
        self._ttl_ms = ttl_ms
        self._payload = payload
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="roomkit-turns-beat", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 6.0)
            self._thread = None

    def _run(self) -> None:
        client = self._connect()
        beat = client.register_script(BEAT_SCRIPT)
        try:
            while True:
                try:
                    beat(keys=self._keys, args=[self._ttl_ms, self._payload()])
                except Exception:
                    logger.warning("A turn process beat failed", exc_info=True)
                if self._stop.wait(self._interval):
                    return
        finally:
            client.close()
