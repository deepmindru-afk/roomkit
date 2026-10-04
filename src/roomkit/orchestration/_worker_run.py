"""A strategy's worker delegation: one task handed to one agent, waited for
within the strategy's bound, and followed on the status bus (RFC §19.7.3,
§19.7.4, §23.3).

One sequence for the supervisor's workers (sequential, parallel, supervised,
per-worker) and the Loop's producer and reviewers: the worker is posted
pending, its task delegated and waited for, then one terminal entry is posted
however the delegation ends: completed, failed, timed out, raised or
cancelled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from roomkit.models.enums import TaskStatus
from roomkit.orchestration.status_bus import StatusLevel, post_agent_lifecycle
from roomkit.tasks.models import task_work

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult


def task_output(result: Any) -> str:
    """The text a strategy reads of a delegated task result: its work.

    A failed task reads as failed, never with its error, which is an
    exception's message for the logs and hooks, not for a model (RFC §9.3),
    nor with the narration a worker cut short keeps as its output (RFC
    §23.3). Accepts a real result, a duck-typed object, or ``None``.
    """
    if result is None:
        return ""
    if work := task_work(result):
        return work
    if getattr(result, "status", None) == TaskStatus.CANCELLED:
        return "The task was cancelled."
    failed = getattr(result, "error", None) or getattr(result, "status", None) == TaskStatus.FAILED
    return "The task failed." if failed else ""


def task_completed(result: Any) -> bool:
    """Whether a delegated task result reports ``completed`` status."""
    return getattr(result, "status", None) == TaskStatus.COMPLETED


@dataclass(frozen=True)
class WorkerOutcome:
    """How a worker's delegation ended.

    Attributes:
        output: What the strategy reads of it: the task's work, or that it
            failed, was cancelled or timed out (never an error's message).
        completed: Whether the task completed.
        task: The delegated task; ``None`` when it timed out.
    """

    output: str
    completed: bool
    task: DelegatedTask | None = None

    @property
    def result(self) -> DelegatedTaskResult | None:
        """The task's result; ``None`` when it timed out."""
        return self.task.result if self.task is not None else None


class WorkerEnd(NamedTuple):
    """A worker delegation's terminal entry: its level, its detail, and what
    it adds to the status's metadata."""

    level: StatusLevel
    detail: str
    metadata: dict[str, Any] | None = None


def _completed_or_failed(outcome: WorkerOutcome) -> WorkerEnd:
    """A worker's terminal entry: completed or failed, with its output."""
    return WorkerEnd(
        StatusLevel.COMPLETED if outcome.completed else StatusLevel.FAILED, outcome.output
    )


@dataclass(frozen=True)
class WorkerStatus:
    """How a worker's delegation shows on the status bus.

    Attributes:
        metadata: Posted with every entry (the room, the strategy, the
            worker's role); a task that ran adds its ``task_id`` to its
            terminal entry.
        action: The entries' action.
        ended: The terminal entry of a delegation that returned, from its
            outcome; one that raised or was cancelled is posted failed.
    """

    metadata: dict[str, Any]
    action: str = "task"
    ended: Callable[[WorkerOutcome], WorkerEnd] = _completed_or_failed


async def run_worker(
    kit: RoomKit,
    room_id: str,
    worker_id: str,
    task: str,
    *,
    timeout: float | None,
    status: WorkerStatus | None = None,
    **delegation: Any,
) -> WorkerOutcome:
    """Delegate *task* to *worker_id* from *room_id* and wait for it.

    Bounded by *timeout* (``None``: unbounded): a task past it is cancelled
    and its delegation reads as failed. Posted on the status bus as *status*
    says, pending then one terminal entry whatever ends it; nothing is
    posted without one (a supervisor's own pass). *delegation* goes to
    :meth:`RoomKit.delegate` (``share_channels``,
    ``require_structured_result``, ``result_tool``).
    """
    post = _Posts(kit, worker_id, status)
    post(StatusLevel.PENDING, task)
    try:
        outcome = await _delegate(kit, room_id, worker_id, task, timeout, delegation)
    except asyncio.CancelledError:
        post(StatusLevel.FAILED, "cancelled")
        raise
    except Exception as exc:
        post(StatusLevel.FAILED, str(exc))
        raise
    if status is not None:
        end = status.ended(outcome)
        post(end.level, end.detail, task=outcome.task, metadata=end.metadata)
    return outcome


async def _delegate(
    kit: RoomKit,
    room_id: str,
    worker_id: str,
    task: str,
    timeout: float | None,
    delegation: dict[str, Any],
) -> WorkerOutcome:
    """The delegation itself, inline, within *timeout*."""
    try:
        delegated = await asyncio.wait_for(
            kit.delegate(room_id, worker_id, task, wait=True, **delegation), timeout=timeout
        )
    except TimeoutError:
        if timeout is None:
            raise
        return WorkerOutcome(f"The task timed out after {timeout:g}s.", False)
    result = delegated.result
    return WorkerOutcome(task_output(result), task_completed(result), delegated)


class _Posts:
    """A worker delegation's entries on the status bus, as its status says."""

    def __init__(self, kit: RoomKit, worker_id: str, status: WorkerStatus | None) -> None:
        self._kit = kit
        self._worker_id = worker_id
        self._status = status

    def __call__(
        self,
        level: StatusLevel,
        detail: str,
        *,
        task: DelegatedTask | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        status = self._status
        if status is None:
            return
        metadata = {**status.metadata, **(metadata or {})}
        if task is not None:
            metadata["task_id"] = task.id
        post_agent_lifecycle(
            self._kit,
            self._worker_id,
            level,
            action=status.action,
            detail=detail,
            metadata=metadata,
        )
