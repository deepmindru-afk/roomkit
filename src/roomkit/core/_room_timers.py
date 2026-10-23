"""A room's timer transitions: which fall due, and when (RFC §5.1).

An ACTIVE room with ``inactive_after_seconds`` pauses once that long has
passed since its last activity, or since its creation before any. A PAUSED
room with ``closed_after_seconds`` closes once that long has passed since it
paused. A transition takes the instant it fell due, however late it is
applied: a check long after both fell due pauses the room at the first
instant and closes it at the second.

Pure: no I/O, so every locked section can ask at no cost whether anything
is due before it reads the room's status.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from roomkit.models.enums import RoomStatus
from roomkit.models.room import Room


@dataclass(frozen=True, slots=True)
class DueTransition:
    """A timer's transition: the status it leads to (PAUSED or CLOSED), the
    instant it fell due and the threshold that elapsed."""

    status: RoomStatus
    at: datetime
    threshold: int


def due_transitions(room: Room, now: datetime) -> list[DueTransition]:
    """The transitions *room*'s timers make due by *now*, in the order they
    fell due: none, a pause, a close, or a pause then a close."""
    due: list[DueTransition] = []
    timers = room.timers
    status, paused_at = room.status, room.paused_at
    if status is RoomStatus.ACTIVE and timers.inactive_after_seconds is not None:
        since = timers.last_activity_at or room.created_at
        at = since + timedelta(seconds=timers.inactive_after_seconds)
        if at > now:
            return due
        due.append(DueTransition(RoomStatus.PAUSED, at, timers.inactive_after_seconds))
        status, paused_at = RoomStatus.PAUSED, at
    if status is RoomStatus.PAUSED and timers.closed_after_seconds is not None:
        # A room stored PAUSED without the instant (before paused_at existed)
        # counts from its last modification, which the pause was.
        since = paused_at or room.updated_at
        at = since + timedelta(seconds=timers.closed_after_seconds)
        if at <= now:
            due.append(DueTransition(RoomStatus.CLOSED, at, timers.closed_after_seconds))
    return due


def transitioned(room: Room, due: list[DueTransition], now: datetime) -> Room:
    """*room* once *due* is applied, in one write at *now*: a pause records
    ``paused_at``, a close ``closed_at`` and clears ``paused_at``, each at
    the instant it fell due."""
    update: dict[str, object] = {"updated_at": now}
    for transition in due:
        update["status"] = transition.status
        if transition.status is RoomStatus.PAUSED:
            update["paused_at"] = transition.at
        else:
            update["closed_at"] = transition.at
            update["paused_at"] = None
    return room.model_copy(update=update)
