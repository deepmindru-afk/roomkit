"""TaskRunner ABC for pluggable background task execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

OnCompleteCallback = Callable[[DelegatedTaskResult], Awaitable[None]]


class TaskRunner(ABC):
    """ABC for executing delegated tasks in the background.

    Follows the same pluggable-backend pattern as ``ConversationStore``
    and ``RoomLockManager``.

    A task ends exactly once, through *on_complete*, whatever ends it (RFC
    §23.1, §23.3): run to its end, it ends completed or failed; cancelled
    (:meth:`cancel`, :meth:`close`, even before it ran), it ends with a
    ``cancelled`` result (:func:`~roomkit.tasks.models.cancelled_task_fields`);
    its run raising outside its turn's outcome, it ends failed with what was
    raised and the record of the turn it ran (``turns``,
    :func:`~roomkit.tasks.models.finished_task_fields`, RFC §23.3 step 6).
    The framework closes the task's span, fires ``ON_TASK_COMPLETED`` and
    hands the result back from that call.

    The worker is given ``task.description``, what it reads (the brief,
    else the task, with the room's conversation when the delegation hands
    it over), not ``task.task``, the label the task is named by (RFC §23.3
    step 4).
    """

    @abstractmethod
    async def submit(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        """Start background execution of *task*.

        A submit that raises has not taken the task: the framework ends it
        failed itself, so the runner must never end it too (RFC §23.3).

        Args:
            kit: The RoomKit instance (used to interact with rooms/channels).
            task: The delegated task handle.
            context: Optional context passed to the agent.
            on_complete: Callback invoked when the task finishes.
        """

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Cancel a task and end it ``cancelled`` through its *on_complete*,
        run to its end even if this call is cancelled. A task that already
        ran to its end ends as it stands. Returns True if the task was found
        and cancelled."""

    @abstractmethod
    async def close(self) -> None:
        """Shutdown the runner: every task it holds ends cancelled, one
        submitted meanwhile too."""
