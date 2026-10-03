"""How a delegated task's end is recorded on its child room (RFC §23.3)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.models.enums import TaskStatus

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTaskResult

logger = logging.getLogger("roomkit.tasks")


async def record_task_end(kit: RoomKit, result: DelegatedTaskResult) -> None:
    """Stamp the child room with how its task ended, and the worker's answer
    when it completed; a failure to write it is logged, never raised."""
    try:
        room = await kit.get_room(result.child_room_id)
        if room is None:
            return
        completed = result.status == TaskStatus.COMPLETED
        metadata = {
            **room.metadata,
            "task_status": result.status,
            "task_result": result.output if completed else None,
        }
        await kit.store.update_room(room.model_copy(update={"metadata": metadata}))
    except Exception:
        logger.exception("Task %s: failed to update child room metadata", result.task_id)
