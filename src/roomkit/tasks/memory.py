"""In-memory task runner using asyncio.create_task()."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.channels._shielded import shielded
from roomkit.core._failure_log import log_failure
from roomkit.core.task_utils import cancel_and_wait, log_task_exception
from roomkit.models.enums import TaskStatus
from roomkit.tasks.base import OnCompleteCallback, TaskRunner
from roomkit.tasks.models import (
    DelegatedTask,
    DelegatedTaskResult,
    cancelled_task_fields,
    finished_task_fields,
)

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.tasks")


class InMemoryTaskRunner(TaskRunner):
    """Default task runner — executes tasks as ``asyncio.Task`` instances."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._handles: dict[str, DelegatedTask] = {}
        # How each task ends once cancelled (RFC §23.3).
        self._cancelled_ends: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}

    async def submit(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        bg = asyncio.create_task(
            self._execute(kit, task, context=context, on_complete=on_complete),
            name=f"delegate:{task.id}",
        )
        bg.add_done_callback(log_task_exception)
        self._tasks[task.id] = bg
        self._handles[task.id] = task
        fields = cancelled_task_fields(context)
        self._cancelled_ends[task.id] = partial(
            self._finish, kit, task, fields, time.monotonic(), on_complete
        )

    async def cancel(self, task_id: str) -> bool:
        handle = self._handles.get(task_id)
        bg = self._tasks.get(task_id)
        if handle is None or bg is None:
            return False
        await cancel_and_wait(bg)
        end = self._cancelled_ends.pop(task_id, None)
        if end is not None and handle.result is None:
            # It ends as any task does, cancelled, its completion run to its
            # end even if this call is cancelled meanwhile (RFC §23.3).
            await shielded(end())
        self._tasks.pop(task_id, None)
        self._handles.pop(task_id, None)
        return True

    async def close(self) -> None:
        for task_id in list(self._tasks):
            await self.cancel(task_id)

    async def _execute(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        start = time.monotonic()
        task.status = TaskStatus.IN_PROGRESS
        # Cancelled (``cancel``, ``close``), it is ended by ``cancel``.
        fields = await self._run(kit, task, context)
        await self._finish(kit, task, fields, start, on_complete)

    async def _run(
        self, kit: RoomKit, task: DelegatedTask, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Run the worker in the task's child room: the task's outcome."""
        agent_response: str | None = None
        failure: Exception | None = None
        try:
            # Update child room status
            room = await kit.get_room(task.child_room_id)
            if room is None:
                logger.warning(
                    "Task %s: child room %s not found",
                    task.id,
                    task.child_room_id,
                )
                # No worker ran.
                fields = finished_task_fields(None, None, context)
                return {**fields, "error": f"Child room {task.child_room_id} not found"}
            await kit.store.update_room(
                room.model_copy(
                    update={
                        "metadata": {
                            **room.metadata,
                            "task_status": TaskStatus.IN_PROGRESS,
                        },
                    }
                )
            )
            # Lazy import to avoid circular dependency
            from roomkit.core.mixins.delegation import run_agent_in_child_room

            agent_response = await run_agent_in_child_room(kit, task.child_room_id, task.task)
        except Exception as exc:
            log_failure(logger, exc, f"Task {task.id}")
            failure = exc
        return finished_task_fields(agent_response, failure, context)

    async def _finish(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        fields: dict[str, Any],
        start: float,
        on_complete: OnCompleteCallback | None,
    ) -> None:
        """End *task* with its outcome *fields*: its child room's status, its
        completion callback, then its waiters."""
        result = DelegatedTaskResult(
            task_id=task.id,
            child_room_id=task.child_room_id,
            parent_room_id=task.parent_room_id,
            agent_id=task.agent_id,
            duration_ms=(time.monotonic() - start) * 1000,
            **fields,
        )

        # Update child room metadata
        try:
            room = await kit.get_room(task.child_room_id)
            if room is not None:
                completed = result.status == TaskStatus.COMPLETED
                await kit.store.update_room(
                    room.model_copy(
                        update={
                            "metadata": {
                                **room.metadata,
                                "task_status": result.status,
                                "task_result": result.output if completed else None,
                            },
                        }
                    )
                )
        except Exception:
            logger.exception("Task %s: failed to update child room metadata", task.id)

        # Run on_complete BEFORE setting result so hooks fire before waiters unblock
        if on_complete:
            try:
                await on_complete(result)
            except Exception:
                logger.exception("on_complete callback failed for task %s", task.id)

        # ALWAYS set result — callers of wait() depend on this
        task._set_result(result)

        self._tasks.pop(task.id, None)
        self._handles.pop(task.id, None)
        self._cancelled_ends.pop(task.id, None)
