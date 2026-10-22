"""What the framework knows of the tasks it runs for each room (RFC §23.4).

A room's ``cancel_task`` answers from it, never from what the StatusBus lists
alone: a task running for the room and how it can end, or the status a task
the framework ran there ended with, even before the bus shows that end.
"""

from __future__ import annotations

import asyncio
from typing import Literal, NamedTuple

TaskReach = Literal["tool", "strategy", "turn"]
"""How a task of a room can end before its own end (RFC §23.4): ``tool``, a
background task the room's ``cancel_task`` cancels; ``strategy``, a task a
strategy runs without posting it, which the strategy follows (the host may
cancel one that runs in the background); ``turn``, an inline delegation,
which ends with the turn that asked for it."""

AGENT_REACH: frozenset[TaskReach] = frozenset({"tool"})
"""The tasks of its room an agent's ``cancel_task`` may cancel."""

HOST_REACH: frozenset[TaskReach] = frozenset({"tool", "strategy", "turn"})
"""The tasks of a room its host may ask to cancel: any the framework runs
for it, the runner cancelling those it holds."""

ENDED_ROOM_TASKS_KEPT = 1024
"""How many ended tasks the record keeps, the latest to end, all rooms
together."""

_WHY_NOT_CANCELLED: dict[TaskReach, str] = {
    "turn": "it runs inside the turn that asked for it, and ends with it",
    "strategy": "the strategy that runs it follows it",
}


class RoomTask(NamedTuple):
    """A task the framework runs for a room, and how it can end."""

    room_id: str
    reach: TaskReach


class RoomTaskRecord:
    """The tasks the framework runs for each room, by id: those running, with
    how each can end, and the latest ended, with the status each ended with."""

    def __init__(self) -> None:
        self._running: dict[str, RoomTask] = {}
        self._ended: dict[str, tuple[RoomTask, str]] = {}

    @property
    def running(self) -> list[str]:
        """The tasks running, by id, in the order they started."""
        return list(self._running)

    def start(self, task_id: str, room_id: str, reach: TaskReach) -> None:
        self._running[task_id] = RoomTask(room_id, reach)

    def end(self, task_id: str, status: str) -> None:
        """*task_id* ended with *status*: kept among the latest ended."""
        task = self._running.pop(task_id, None)
        if task is None:
            return
        self._ended[task_id] = (task, status)
        if len(self._ended) > ENDED_ROOM_TASKS_KEPT:
            del self._ended[next(iter(self._ended))]

    def reach(self, task_id: str, room_id: str) -> TaskReach | None:
        """How *task_id*, running for *room_id*, can end; ``None`` for any
        other task: another room's, one ended, one the framework never ran."""
        task = self._running.get(task_id)
        return task.reach if task is not None and task.room_id == room_id else None

    def ended_with(self, task_id: str, room_id: str) -> str | None:
        """The status *task_id*, run for *room_id*, ended with; ``None`` for
        any other task, or one ended before the latest kept."""
        ended = self._ended.get(task_id)
        return ended[1] if ended is not None and ended[0].room_id == room_id else None

    def ended_reach(self, task_id: str, room_id: str) -> TaskReach | None:
        """How *task_id*, run for *room_id* and ended, could have ended before
        its own end; ``None`` for any other task."""
        ended = self._ended.get(task_id)
        return ended[0].reach if ended is not None and ended[0].room_id == room_id else None


class TaskCancel:
    """A ``cancel_task`` cancel under way for one task: the channels whose
    calls asked for it, none of which is handed the cancellation back,
    whether it cancelled the task, which a call asking meanwhile is answered,
    and whether the task's end has decided whom it hands its end back to
    (*sealed*): a call asking after that is no longer one of its callers
    (RFC §23.4)."""

    def __init__(self, channel_id: str | None) -> None:
        self.callers: set[str] = {channel_id} if channel_id else set()
        self.cancelled: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.sealed = False


def why_not_cancelled(reach: TaskReach | None) -> str | None:
    """Why a task running for a room is not that room's to cancel, by how it
    can end; ``None`` when nothing keeps it from being cancelled there."""
    return _WHY_NOT_CANCELLED.get(reach) if reach is not None else None
