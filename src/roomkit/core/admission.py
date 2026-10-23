"""Off-lock checks: admission tickets for ``needs_lock=False`` hooks (RFC §9.5.1).

A ``BEFORE_BROADCAST`` check that reads only the event (a PII scan, a
moderation call) runs before the room lock is taken, so the I/O of one
message's check does not hold the other events of the room. Arrival order,
which holding the lock across the check would give, comes from a per-room
ticket: checks of successive events overlap, but each event waits for every
earlier ticket of the room to be released before it takes the lock, so they
commit in arrival order.

Neither existing ordering mechanism fits. The room lock (:class:`RoomLockManager`)
is a mutex: holding it across the check is exactly the serialization being
removed. The delivery lanes (:mod:`roomkit.core.lanes`) order by committed
index, and at this point no index exists yet.

The ticket lives in process memory: arrival order holds within one process.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from roomkit.core._reentrance import admission_order
from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core._reentrance import ReentryBudget
    from roomkit.core.hooks import HookEngine, SyncPipelineResult
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent


class _Ticket:
    """One event's place in its room's admission order."""

    __slots__ = ("_ahead", "_released")

    def __init__(self, ahead: list[_Ticket]) -> None:
        self._ahead = ahead
        self._released = asyncio.Event()

    async def wait_turn(self) -> None:
        """Return once every ticket taken before this one is released."""
        for ticket in self._ahead:
            await ticket._released.wait()
        self._ahead = []

    def release(self) -> None:
        self._released.set()


class RoomAdmission:
    """Per-room admission tickets and the off-lock check they order.

    *report_errors* announces the hooks of a check that produced no usable
    result (``hook_error``, RFC §8.2), as soon as the check has run.
    """

    def __init__(
        self,
        hook_engine: HookEngine,
        build_context: Callable[[str], Awaitable[RoomContext]],
        report_errors: Callable[[str, RoomEvent, SyncPipelineResult], Awaitable[None]]
        | None = None,
    ) -> None:
        self._hook_engine = hook_engine
        self._build_context = build_context
        self._report_errors = report_errors
        self._queues: dict[str, deque[_Ticket]] = {}

    def _take(self, room_id: str) -> _Ticket:
        queue = self._queues.setdefault(room_id, deque())
        ticket = _Ticket(list(queue))
        queue.append(ticket)
        return ticket

    def _release(self, room_id: str, ticket: _Ticket) -> None:
        # Synchronous on purpose: it runs from ``finally`` on every exit path,
        # cancellation included, and must not be able to fail half-way.
        ticket.release()
        queue = self._queues.get(room_id)
        if queue is None:
            return
        queue.remove(ticket)
        if not queue:
            del self._queues[room_id]

    async def _report(self, room_id: str, event: RoomEvent, outcome: SyncPipelineResult) -> None:
        """Report the check's unusable results now, once: the locked pass
        that reuses the outcome finds none left to report."""
        if self._report_errors is not None and outcome.hook_errors:
            await self._report_errors(room_id, event, outcome)
        outcome.hook_errors = []

    def pending(self, room_id: str) -> int:
        """How many tickets of *room_id* are held (for tests and metrics)."""
        return len(self._queues.get(room_id, ()))

    @asynccontextmanager
    async def admitted(
        self,
        room_id: str,
        event: RoomEvent,
        context: RoomContext | None,
        *,
        budget: ReentryBudget,
        adopted: bool,
    ) -> AsyncGenerator[SyncPipelineResult | None, None]:
        """Run *event*'s off-lock check, then hold its turn until exit.

        Yields ``None`` — nothing taken, every hook left to the locked pass —
        when no off-lock hook matches the event, or when the event is
        *adopted* by the room's work (RFC §9.5.1 Reentrance): made under the
        room lock (a locked hook, or code under the lock) or in the room's
        admission order (an off-lock hook's body, a message's parse), it
        would otherwise wait for the ticket its own caller holds, released
        only once the caller returns.

        The check's hooks stand in the room's admission order for *event*:
        what their bodies send into the room is adopted beside it, on its
        cascade's *budget*.

        Otherwise yields the off-lock outcome once every earlier ticket of the
        room has been released. The caller takes the room lock and commits inside the
        ``async with``; the ticket is released on exit, whatever the path —
        committed, blocked, refused, failed, timed out or cancelled. A hook
        of the check that produced no usable result is reported before the
        wait, whatever the gates under the lock decide next (RFC §9.5.1); its
        timeout was reported as it expired.
        """
        if adopted or not self._hook_engine.has_off_lock_hooks(room_id, event):
            yield None
            return

        ticket = self._take(room_id)
        try:
            if context is None:
                context = await self._build_context(room_id)
            with admission_order(room_id, budget):
                outcome = await self._hook_engine.run_sync_hooks(
                    room_id, HookTrigger.BEFORE_BROADCAST, event, context, needs_lock=False
                )
            await self._report(room_id, event, outcome)
            await ticket.wait_turn()
            yield outcome
        finally:
            self._release(room_id, ticket)
