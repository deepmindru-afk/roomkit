"""How a process of the Redis turn backend beats, and reads the others' beats
(RFC §10.6.8).

The beat runs on a thread of its own with a connection of its own, so a turn
that keeps the event loop busy never stalls it; what it writes is built from
what the event loop replaces whole and never changes in place. The view of
the others is read again each heartbeat interval on the event loop, with
whether the processes running this one's turns still run them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
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
    runs for owners elsewhere, and whether it is a room owner. A process
    closing beats for its executors without declaring them: none is placed
    there any more, while what they were doing is answered."""

    def __init__(self, node: str, routes: Routes) -> None:
        self._node = node
        self._routes = routes
        # The executors it beats for and whether the process closes, numbered
        # each time they change: a beat built from an older state never
        # undoes one built from a newer.
        self._served: tuple[int, tuple[TurnExecutor, ...], bool] = (0, (), False)
        self._declared: dict[str, str] = {}

    @property
    def executors(self) -> tuple[TurnExecutor, ...]:
        return self._served[1]

    @executors.setter
    def executors(self, executors: tuple[TurnExecutor, ...]) -> None:
        # Replaced whole, never changed, so the beat thread reads it whole.
        generation, _, closing = self._served
        self._served = (generation + 1, executors, closing)

    def close(self) -> None:
        """Beat from now without declaring the executors."""
        generation, executors, _ = self._served
        self._served = (generation + 1, executors, True)

    @property
    def members(self) -> list[str]:
        return [member_of(self._node, executor.executor_id) for executor in self.executors]

    def __call__(self) -> str:
        generation, executors, closing = self._served
        serving = [member_of(self._node, e.executor_id) for e in executors if not e.closed]
        declared = (
            {}
            if closing
            else {
                member_of(self._node, e.executor_id): self._declaration(e)
                for e in executors
                if not e.closed
            }
        )
        # A closed executor's member another of its id now serves stays.
        closed = (member_of(self._node, e.executor_id) for e in executors if e.closed)
        gone = [member for member in closed if member not in serving]
        payload = {
            "node": self._node,
            "generation": generation,
            "declared": declared,
            "undeclared": serving if closing else [],
            "gone": gone,
            "running": sorted(self._routes.running),
            "owner": self._routes.owner is not None,
        }
        return json.dumps(payload)

    def _declaration(self, executor: TurnExecutor) -> str:
        """*executor*'s declaration now, the last one kept when the event
        loop changed its room hooks while this thread read them."""
        assert executor.hosted is not None
        member = member_of(self._node, executor.executor_id)
        try:
            hooks = executor.hooks.defined_room_hooks()
        except RuntimeError:
            return self._declared.get(member, "")
        dumped = Declaration(self._node, executor.executor_id, executor.hosted, hooks).dump()
        self._declared[member] = dumped
        return dumped


class LiveExecutors:
    """A process's view of the processes alive, read each heartbeat interval:
    the executors and what they declare, the owners alive, and which of the
    turns this process reads their processes still run.

    Args:
        registry: The processes' beats and declarations.
        interval: How often the view is read again.
        timeout: How long a silent process stays alive (RFC §10.6.8).
        open_turns: The turns this process reads, by the node running each.
    """

    def __init__(
        self,
        registry: Registry,
        interval: float,
        timeout: float,
        open_turns: Callable[[], dict[str, list[str]]],
    ) -> None:
        self._registry = registry
        self.interval = interval
        """How often the view is read again."""
        self._timeout = timeout
        self._timeout_ms = int(timeout * 1000)
        self._open_turns = open_turns
        self._view = View()
        self._running: dict[str, frozenset[str]] = {}
        # The reads failing in a row: how many, from the first one's failure
        # to the last one's.
        self._failures = 0
        self._failing_since: float | None = None
        self._failed_at = 0.0
        self._refresher: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._refresher is None:
            self._refresher = asyncio.get_running_loop().create_task(
                self._refresh_each_interval(), name="roomkit-turns-live-executors"
            )

    async def current(self) -> dict[str, Declaration]:
        """The executors alive and what they declare, by member, read now."""
        await self.refresh()
        return self._alive_declared()

    def last_beat(self, member: str) -> int | None:
        """The server time of *member*'s last beat seen, in ms."""
        return self._view.beats.get(member)

    def alive(self, member: str) -> bool:
        """Whether *member* beat within the timeout, as last read."""
        beat = self._view.beats.get(member)
        return not self._unread() and beat is not None and self._view.now - beat < self._timeout_ms

    def runs(self, node: str, turn_id: str) -> bool:
        """Whether *node* still ran *turn_id* when it last beat, as last asked."""
        return turn_id in self._running.get(node, frozenset())

    def owner_alive(self, node: str) -> bool:
        """Whether the owner process *node* beat within the timeout."""
        beat = self._view.owners.get(node)
        return not self._unread() and beat is not None and self._view.now - beat < self._timeout_ms

    async def refresh(self) -> None:
        self._view = await self._registry.view(forget_after_ms=10 * self._timeout_ms)
        self._failures = 0
        self._failing_since = None

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

    def _unread(self) -> bool:
        """Whether the reads of the others kept failing (or hanging) over the
        heartbeat timeout, two of them at least: none of them is then known
        alive. Only the reads this process made count, so a stall of its own
        event loop (one read at most) is not the others' silence."""
        failing = self._failing_since
        return (
            self._failures >= 2
            and failing is not None
            and self._failed_at - failing >= self._timeout
        )

    async def _refresh_each_interval(self) -> None:
        while True:
            try:
                # A read that hangs fails as one that raises.
                async with asyncio.timeout(self._timeout):
                    await self.refresh()
                    self._running = await self._registry.running(self._open_turns())
            except asyncio.CancelledError:
                raise
            except Exception:
                self._failed()
                logger.warning("Reading the turn processes alive failed", exc_info=True)
            await asyncio.sleep(self.interval)

    def _failed(self) -> None:
        """A read failed: counted from when it failed, so neither a stall of
        this process nor a read that succeeded before it counts."""
        self._failures += 1
        self._failed_at = time.monotonic()
        if self._failing_since is None:
            self._failing_since = self._failed_at


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
