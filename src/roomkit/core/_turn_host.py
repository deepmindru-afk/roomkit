"""The verbs of a turn's ``TurnHost``, run by the kit that owns the room
(RFC §10.6.5).

Each verb arrives as data, validated against the record of its arguments
(``roomkit.turns.host.VERBS``) before it reaches the kit, and answers with
data. It runs in the kit's own code, in the context the owner's task reading
the turn lent it, so its reentrancy is that task's.

The tasks the host delegates are kept until ``wait_task`` read their result,
or, once ended, until more than :data:`ENDED_TASKS_KEPT` ended after them.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from pydantic_core import to_jsonable_python

from roomkit.models.enums import TaskStatus
from roomkit.turns.base import OwnerHost
from roomkit.turns.host import (
    VERBS,
    AddObservation,
    Delegate,
    PostStatus,
    SendEvent,
    UpdateRoomMetadata,
    WaitTask,
)

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask
    from roomkit.turns.wire import TurnRequest

ENDED_TASKS_KEPT = 1024
"""How many ended tasks a host keeps for a ``wait_task`` that has not come."""


class KitHost(OwnerHost):
    """The kit's side of its turns' hosts."""

    def __init__(self, kit: RoomKit) -> None:
        self._kit = kit
        self._tasks: OrderedDict[str, DelegatedTask] = OrderedDict()
        self._verbs: dict[str, Callable[[Any], Awaitable[Any]]] = {
            "send_event": self._send_event,
            "delegate": self._delegate,
            "wait_task": self._wait_task,
            "post_status": self._post_status,
            "update_room_metadata": self._update_room_metadata,
            "add_observation": self._add_observation,
        }

    async def call(self, request: TurnRequest, verb: str, arguments: dict[str, Any]) -> Any:
        record = VERBS.get(verb)
        serve = self._verbs.get(verb)
        if record is None or serve is None:
            raise ValueError(f"No verb {verb!r} on a turn's host")
        try:
            args: BaseModel = record.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"The arguments of verb {verb!r} are not valid: {exc}") from exc
        return to_jsonable_python(await serve(args), fallback=str)

    async def _send_event(self, args: SendEvent) -> Any:
        options = args.model_dump(exclude={"content"})
        event = await self._kit.send_event(content=args.content, **options)
        return event.model_dump(mode="json")

    async def _delegate(self, args: Delegate) -> str:
        task = await self._kit.delegate(**args.model_dump())
        self._keep(task)
        return task.id

    async def _wait_task(self, args: WaitTask) -> Any:
        task = self._tasks.get(args.task_id)
        if task is None:
            return None
        try:
            result = await task.wait(args.timeout)
        except TimeoutError:
            return None
        self._tasks.pop(args.task_id, None)
        return result.model_dump(mode="json")

    async def _post_status(self, args: PostStatus) -> None:
        self._kit.status_bus.post(
            args.agent_id, args.action, args.status, detail=args.detail, metadata=args.metadata
        )

    async def _update_room_metadata(self, args: UpdateRoomMetadata) -> Any:
        room = await self._kit.update_room_metadata(
            args.room_id, args.metadata, organization_id=args.organization_id
        )
        return room.model_dump(mode="json")

    async def _add_observation(self, args: AddObservation) -> None:
        await self._kit.store.add_observation(args.observation)

    def _keep(self, task: DelegatedTask) -> None:
        """Keep *task* for ``wait_task``, letting go of the oldest ended ones
        past the bound."""
        self._tasks[task.id] = task
        ended = [tid for tid, kept in self._tasks.items() if _ended(kept)]
        for task_id in ended[: max(0, len(ended) - ENDED_TASKS_KEPT)]:
            del self._tasks[task_id]


def _ended(task: DelegatedTask) -> bool:
    return task.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
