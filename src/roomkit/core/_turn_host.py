"""The verbs of a turn's ``TurnHost``, run by the kit that owns the room
(RFC §10.6.5).

Each verb arrives as data, validated against the record of its arguments
(``roomkit.turns.host.VERBS``) before it reaches the kit, and answers with
data. It runs in the kit's own code, under what the turn gave the call (its
chain depth, who asked) and in the context the owner's task reading the turn
lent it, so its reentrancy is that task's.

A host acts for its turn's room, in the room's organization: a verb naming
another room, or another organization, is refused, and an event a turn sends
is at least as deep as the turn's response (RFC §8.3).

The tasks the room's hosts delegate are kept for ``wait_task`` as long as
they run, and once ended until :data:`ENDED_TASKS_KEPT` more ended after them.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from roomkit.models.enums import TaskStatus
from roomkit.tools.context import _current_turn_chain_depth
from roomkit.turns._codec import error_record
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
from roomkit.turns.wire import TurnRequest

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask

ENDED_TASKS_KEPT = 1024
"""How many ended tasks the hosts keep for a ``wait_task`` that has not come."""

Verb = Callable[[TurnRequest, Any], Awaitable[Any]]


class KitHost(OwnerHost):
    """The kit's side of its turns' hosts."""

    def __init__(self, kit: RoomKit) -> None:
        self._kit = kit
        self._tasks: OrderedDict[str, DelegatedTask] = OrderedDict()
        self._verbs: dict[str, Verb] = {
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
            args = record.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"The arguments of verb {verb!r} are not valid: {exc}") from exc
        return to_jsonable_python(await serve(request, args), fallback=str)

    async def _send_event(self, request: TurnRequest, args: SendEvent) -> Any:
        options = args.model_dump(exclude={"content", "chain_depth", "organization_id"})
        return await self._kit.send_event(
            content=args.content,
            chain_depth=max(args.chain_depth, _current_turn_chain_depth()),
            organization_id=_organization(request, args.room_id, args.organization_id),
            **options,
        )

    async def _delegate(self, request: TurnRequest, args: Delegate) -> str:
        _same_room(request, args.room_id)
        task = await self._kit.delegate(**args.model_dump())
        self._keep(task)
        return task.id

    async def _wait_task(self, request: TurnRequest, args: WaitTask) -> Any:
        task = self._tasks.get(args.task_id)
        if task is None or task.parent_room_id != request.room_id:
            return None
        try:
            result = await task.wait(args.timeout)
        except TimeoutError:
            return None
        failure = result.exception
        return {
            "result": result,
            "exception": None if failure is None else error_record(failure),
        }

    async def _post_status(self, request: TurnRequest, args: PostStatus) -> None:
        self._kit.status_bus.post(
            args.agent_id, args.action, args.status, detail=args.detail, metadata=args.metadata
        )

    async def _update_room_metadata(self, request: TurnRequest, args: UpdateRoomMetadata) -> Any:
        return await self._kit.update_room_metadata(
            args.room_id,
            args.metadata,
            organization_id=_organization(request, args.room_id, args.organization_id),
        )

    async def _add_observation(self, request: TurnRequest, args: AddObservation) -> None:
        _same_room(request, args.observation.room_id)
        await self._kit.store.add_observation(args.observation)

    def _keep(self, task: DelegatedTask) -> None:
        """Keep *task* for ``wait_task``, letting go of the oldest ended ones
        past the bound."""
        self._tasks[task.id] = task
        ended = [tid for tid, kept in self._tasks.items() if _ended(kept)]
        for task_id in ended[: max(0, len(ended) - ENDED_TASKS_KEPT)]:
            del self._tasks[task_id]


def _same_room(request: TurnRequest, room_id: str) -> None:
    """Refuse a verb for another room than its turn's."""
    if room_id != request.room_id:
        raise ValueError(
            f"A turn's host acts for its room {request.room_id!r}, not for {room_id!r}"
        )


def _organization(request: TurnRequest, room_id: str, named: str | None) -> str | None:
    """The organization a verb runs in: its turn's room's; one the verb names
    must be that one."""
    _same_room(request, room_id)
    organization = request.context.room.organization_id
    if named is not None and named != organization:
        raise ValueError(f"A turn's host acts in its room's organization, not in {named!r}")
    return organization


def _ended(task: DelegatedTask) -> bool:
    return task.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
