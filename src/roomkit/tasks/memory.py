"""In-memory task runner using asyncio.create_task()."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.core._failure_log import log_failure
from roomkit.core.task_utils import (
    cancel_and_wait,
    log_task_exception,
    shielded,
    spared_by_close,
    start_named,
    wait_for_unspared,
)
from roomkit.models.enums import TaskStatus
from roomkit.models.response_metadata import TurnEntries
from roomkit.tasks._child_status import record_task_end
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
        # How each task ends cancelled. Whoever takes a task's entry ends the
        # task, so it ends once: the task itself once its work ran, or
        # ``cancel`` (RFC §23.3).
        self._cancelled_ends: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}
        # The end ``cancel`` took of each task, on a run of its own until it
        # is over: it outlives the task's work, so a cancel asking meanwhile,
        # the runner's close's among them, waits for it.
        self._ends: dict[str, asyncio.Task[None]] = {}

    async def submit(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        turns: TurnEntries = {}
        end = partial(
            self._end_cancelled, kit, task, context, turns, time.monotonic(), on_complete
        )
        bg = start_named(
            self._execute(kit, task, context=context, on_complete=on_complete, turns=turns),
            name=f"delegate:{task.id}",
        )
        bg.add_done_callback(log_task_exception)
        self._tasks[task.id] = bg
        self._handles[task.id] = task
        self._cancelled_ends[task.id] = end

    async def cancel(self, task_id: str) -> bool:
        bg = self._tasks.get(task_id)
        if task_id not in self._handles or bg is None:
            return False
        end = self._cancelled_ends.pop(task_id, None)
        if end is None:
            # It is ending as it stands: by itself once its work ran, or by
            # the cancel that took its end, once that work ended. Waited for,
            # that end included, but not once it closes the kit (an
            # on_complete joining the close), which would wait for itself.
            await wait_for_unspared(lambda: self._runs(task_id) - spared_by_close())
            return False
        # Its end is held from now on, and runs once its work has ended.
        work_ended = asyncio.Event()
        ending = self._hold_end(task_id, _once_set(work_ended, end))
        try:
            await cancel_and_wait(bg)
        finally:
            # It ends cancelled, to its end even if this call is cancelled.
            work_ended.set()
            await asyncio.shield(ending)
        return True

    async def close(self) -> None:
        # A task's end may delegate again: that task is cancelled in turn.
        # The ones this close runs under (a worker's tool closing the kit)
        # are spared: cancelling one would wait for itself.
        while running := self._unspared():
            for task_id in running:
                await self.cancel(task_id)

    def _unspared(self) -> list[str]:
        """The tasks running but the ones the current close runs under: their
        work, or the end a cancel took (an on_complete closing the kit). One
        whose end was taken and whose runs are all over has nothing left to
        wait for, whatever is left of its entry: listed, it would be listed
        again at every pass, the close never yielding."""
        spared = spared_by_close()
        return [
            task_id
            for task_id in self._tasks
            if self._still_runs(task_id) and not self._runs(task_id) & spared
        ]

    def _still_runs(self, task_id: str) -> bool:
        """Whether *task_id* has its end still to take, or a run not over."""
        if task_id in self._cancelled_ends:
            return True
        return any(not run.done() for run in self._runs(task_id))

    def _runs(self, task_id: str) -> set[asyncio.Task[None]]:
        """The runs of *task_id*: its work and, once a cancel took it, its
        end, which outlives that work."""
        runs = (self._tasks.get(task_id), self._ends.get(task_id))
        return {run for run in runs if run is not None}

    def _hold_end(self, task_id: str, end: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Run *end*, the end of *task_id* a cancel took, as a run of its own
        held until it is over (:meth:`_runs`). Named in
        :func:`~roomkit.core.task_utils.held_runs`, so a ``close()`` from
        under it (an on_complete closing the kit) spares it rather than wait
        for itself."""
        ending = start_named(end, name=f"delegate-end:{task_id}")
        self._ends[task_id] = ending
        ending.add_done_callback(lambda _: self._ends.pop(task_id, None))
        return ending

    async def _execute(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
        turns: TurnEntries,
    ) -> None:
        start = time.monotonic()
        task.status = TaskStatus.IN_PROGRESS
        try:
            fields = await self._run(kit, task, context, turns)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            # Raised outside its turn's outcome: it ends failed, once (RFC §23.3).
            failed = finished_task_fields(None, exc, context, turns)
            await self._end_ran(kit, task, failed, start, on_complete)
            raise
        await self._end_ran(kit, task, fields, start, on_complete)

    async def _end_ran(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        fields: dict[str, Any],
        start: float,
        on_complete: OnCompleteCallback | None,
    ) -> None:
        """End *task*, whose run is over, with its outcome *fields*, to its
        end whatever cancels it now; unless ``cancel`` took its end first."""
        if self._cancelled_ends.pop(task.id, None) is None:
            # ``cancel`` took its end: it ends the task, cancelled.
            return
        await shielded(self._finish(kit, task, fields, start, on_complete))

    async def _end_cancelled(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        context: dict[str, Any] | None,
        turns: TurnEntries,
        start: float,
        on_complete: OnCompleteCallback | None,
    ) -> None:
        """End a task cancelled from outside: cancelled, with no output, its
        worker's turn under ``turns`` once its run had begun (RFC §23.3)."""
        began = task.status == TaskStatus.IN_PROGRESS
        fields = cancelled_task_fields(context, turns, worker=task.agent_id if began else None)
        await self._finish(kit, task, fields, start, on_complete)

    async def _run(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        context: dict[str, Any] | None,
        turns: TurnEntries,
    ) -> dict[str, Any]:
        """Run the worker in the task's child room, its turn's record into
        *turns*: the task's outcome."""
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
            # Its key alone: a full room write would undo what was written
            # since the room was read, the room's register of authors among it.
            await kit.store.patch_room_metadata(
                task.child_room_id, {"task_status": TaskStatus.IN_PROGRESS}
            )
            # Lazy import to avoid circular dependency
            from roomkit.core.mixins.delegation import run_agent_in_child_room

            agent_response = await run_agent_in_child_room(
                kit, task.child_room_id, task.description, turns=turns
            )
        except Exception as exc:
            log_failure(logger, exc, f"Task {task.id}")
            failure = exc
        return finished_task_fields(agent_response, failure, context, turns)

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
        await record_task_end(kit, result)

        # Run on_complete BEFORE setting result so hooks fire before waiters unblock
        if on_complete:
            await _complete(kit, task.id, on_complete, result)

        # ALWAYS set result — callers of wait() depend on this
        task._set_result(result)

        self._tasks.pop(task.id, None)
        self._handles.pop(task.id, None)


async def _once_set(ready: asyncio.Event, end: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run *end* once *ready* is set: a cancelled task's end, once its work
    has ended (its worker's turn recorded)."""
    await ready.wait()
    await end()


async def _complete(
    kit: RoomKit, task_id: str, on_complete: OnCompleteCallback, result: DelegatedTaskResult
) -> None:
    """Run a task's completion callback as the kit's work, on a task of its
    own: the kit's close gives it the grace, then cuts it, and names it in
    its log when it outlasts its unwind (RFC §12.10.4, §23.3). The task's end
    goes on whatever becomes of it."""

    async def callback() -> None:
        await on_complete(result)

    run = start_named(callback(), name=f"roomkit-task-on-complete-{task_id}")
    kit._hold_channel_work(run)
    await asyncio.wait({run})
    if not run.cancelled() and (error := run.exception()) is not None:
        logger.error("on_complete callback failed for task %s", task_id, exc_info=error)
