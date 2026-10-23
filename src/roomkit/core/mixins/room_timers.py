"""RoomTimersMixin — a room's timer transitions, applied when they fall due (RFC §5.1).

Every locked section that reads a room's status (the gate of §10.1 step 6,
an attach or a binding change, a direct mutation, a record's commit) first
applies the transitions due at its instant (:meth:`_apply_due_timers`), so an
event that arrives after its room fell due is refused as if the timer had run
on time. A room nobody writes to transitions when the host checks its timers
(:meth:`check_room_timers`, :meth:`check_all_timers`).

Each transition takes the instant it fell due, all of them in one write; each
emits its framework event once stored, before anything the section decides
next, and fires its hook once the section has released the room lock.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.core._room_timers import DueTransition, due_transitions, transitioned
from roomkit.core.locks import _has_room_lock, unlocked_context
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.task_utils import start_named
from roomkit.models.enums import EventType, HookTrigger, RoomStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.room import Room

logger = logging.getLogger("roomkit.framework")


def _report(room_id: str, transition: DueTransition) -> tuple[HookTrigger, str, str]:
    """The trigger, the code and the message of *transition*."""
    if transition.status is RoomStatus.PAUSED:
        trigger, verb = HookTrigger.ON_ROOM_PAUSED, "paused"
    else:
        trigger, verb = HookTrigger.ON_ROOM_CLOSED, "closed"
    return trigger, f"room_{verb}_by_timer", f"Room {room_id} {verb} by timer"


def _log_failed_announcement(task: asyncio.Task[None]) -> None:
    if task.cancelled() or (error := task.exception()) is None:
        return
    logger.error("A room's timer hooks failed", exc_info=error)


class RoomTimersMixin(HelpersMixin):
    """Room timer transitions: the host's check and every locked section's.

    Host contract: :class:`~roomkit.core.mixins.helpers.FrameworkHelpers`,
    with the room lock manager and the room recordings.
    """

    _lock_manager: RoomLockManager

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # see RoomLifecycleMixin
    _stop_room_recordings: Any  # see RoomRecordingMixin

    async def check_room_timers(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Apply the room's due timer transitions (RFC §5.1), for a room
        nobody writes to.

        An ACTIVE room pauses once ``inactive_after_seconds`` passed since its
        last activity, or since its creation before any; a PAUSED room
        closes once ``closed_after_seconds`` passed since ``paused_at``. Each
        transition takes the instant it fell due, however late the check:
        one long after both pauses the room at the first instant and closes
        it at the second, in one write. Each emits ``room_paused`` or
        ``room_closed`` (``reason = timer``), then fires ``ON_ROOM_PAUSED``
        or ``ON_ROOM_CLOSED`` once the room lock is released, before this
        returns.

        Returns the room as stored once its due transitions are applied.
        *organization_id* scopes the check to one tenant (RFC §17.2).
        """
        hooks: list[asyncio.Task[None]] = []
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id, organization_id=organization_id)
            due = await self._apply_due_timers(room, hooks=hooks)
            if due is not room:
                room = await self.get_room(room_id)
        await self._await_timer_hooks(room_id, hooks)
        return room

    async def check_all_timers(self, *, organization_id: str | None = None) -> list[Room]:
        """Check the timers of every ACTIVE and PAUSED room
        (:meth:`check_room_timers`) and return the rooms that transitioned.

        *organization_id* limits the check to one tenant's rooms; with none,
        every room is checked.
        """
        rooms = [
            room
            for status in (RoomStatus.ACTIVE, RoomStatus.PAUSED)
            for room in await self._store.find_rooms(
                organization_id=organization_id, status=status.value
            )
        ]
        transitioned_rooms: list[Room] = []
        for room in rooms:
            checked = await self.check_room_timers(room.id, organization_id=organization_id)
            if checked.status != room.status:
                transitioned_rooms.append(checked)
        return transitioned_rooms

    async def _apply_due_timers(
        self, room: Room, *, hooks: list[asyncio.Task[None]] | None = None
    ) -> Room:
        """*room* with the timer transitions due now applied (RFC §5.1); call
        under its lock, before the section reads its status. *room* itself
        when nothing is due, which costs no I/O.

        The transitions are written in one write, each at the instant it fell
        due; a close stops the room's recordings first. Each then has its
        lifecycle record and its framework event, before anything the
        section decides next; their hooks fire once the lock is released,
        from a task the kit holds, which *hooks* collects for a caller that
        waits for it.
        """
        now = datetime.now(UTC)
        due = due_transitions(room, now)
        if not due:
            return room
        after = transitioned(room, due, now)
        if after.status is RoomStatus.CLOSED:
            await self._stop_room_recordings(room.id)
            await self._store_refusing_room(after)
        else:
            await self._store.update_room(after)
        for transition in due:
            await self._announce_transition(room.id, transition)
        context = await self._build_context(room.id)
        task = self._fire_timer_hooks(room.id, due, context)
        if hooks is not None:
            hooks.append(task)
        return after

    async def _announce_transition(self, room_id: str, transition: DueTransition) -> None:
        """*transition*'s lifecycle record and its framework event, once the
        room is stored (RFC §5.1, §8.2)."""
        _, code, message = _report(room_id, transition)
        logger.info(message, extra={"room_id": room_id})
        await self._emit_system_event(
            room_id,
            EventType.SYSTEM,
            code=code,
            message=message,
            data={"threshold": transition.threshold, "at": transition.at.isoformat()},
            # The record of the transition itself, written once the status
            # has flipped: the one write a CLOSED room still owes its
            # timeline (RFC §5.1).
            records_transition=True,
        )
        name = "room_paused" if transition.status is RoomStatus.PAUSED else "room_closed"
        await self._emit_framework_event(name, room_id=room_id, data={"reason": "timer"})

    def _fire_timer_hooks(
        self, room_id: str, due: list[DueTransition], context: RoomContext
    ) -> asyncio.Task[None]:
        """Fire *due*'s hooks, in order, with *context*, the room as the
        transitions left it, once the section that applied them has released
        the room lock (RFC §5.1, §9.3).

        A task started in a context that holds no lock, which asks for the
        lock as any later entry and waits for the section, and any ahead of
        it, to release it. The kit holds it with its other background work.
        """

        async def fire() -> None:
            async with self._lock_manager.locked(room_id):
                pass
            for transition in due:
                trigger, code, message = _report(room_id, transition)
                await self._fire_lifecycle_hook(
                    room_id,
                    trigger,
                    EventType.SYSTEM,
                    code=code,
                    message=message,
                    data={"threshold": transition.threshold, "at": transition.at.isoformat()},
                    context=context,
                )

        task = start_named(
            fire(), name=f"roomkit-timer-hooks-{room_id}", context=unlocked_context()
        )
        task.add_done_callback(_log_failed_announcement)
        task.add_done_callback(self._pending_hook_tasks.discard)
        self._pending_hook_tasks.add(task)
        return task

    async def _await_timer_hooks(self, room_id: str, hooks: list[asyncio.Task[None]]) -> None:
        """Wait, once the caller's section released the lock, for the timer
        hooks it started; not under a lock held from before it, which their
        task waits for."""
        if hooks and not _has_room_lock(room_id, self._lock_manager):
            await asyncio.shield(asyncio.gather(*hooks))

    async def _room_due(self, room_id: str, *, organization_id: str | None = None) -> Room | None:
        """The room, read under its lock with its due timer transitions
        applied (RFC §5.1), or ``None`` when the store holds none. Scoped to
        *organization_id*, the read raises as :meth:`get_room` does."""
        if organization_id is not None:
            room: Room | None = await self.get_room(room_id, organization_id=organization_id)
        else:
            room = await self._store.get_room(room_id)
        return None if room is None else await self._apply_due_timers(room)

    async def _due_context(
        self, context: RoomContext, rebuild: Callable[[], Awaitable[RoomContext]]
    ) -> RoomContext:
        """*context*, read under the room lock, once the room's due timer
        transitions are applied: rebuilt by *rebuild* when one was, since
        they wrote the room and its records."""
        if not due_transitions(context.room, datetime.now(UTC)):
            return context
        await self._apply_due_timers(context.room)
        return await rebuild()
