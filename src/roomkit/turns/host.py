"""How code on a turn's executor reaches a room: :class:`TurnHost` (RFC §10.6.5).

A host's tool, or a hook the executor fires, runs where the turn runs, which
may be another process than the room owner's. It reads the room from the
store, and changes it only through the turn's host, read with
:func:`current_turn_host`: inside a tool call of a turn and inside a hook the
executor fires during a turn; ``None`` anywhere else, between two turns
included. Each verb is one ``host_call`` on the turn's link, every argument
is data, and the owner runs it as its own code would, with the reentrancy of
its task reading the turn: a ``send_event`` that would wait on the room's own
delivery from a task that delivery needs returns with its completion
detached.

The verbs are enumerated here, each with the record its arguments cross as;
the owner validates what it receives against it.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from roomkit.models.enums import EventType, Visibility
from roomkit.models.event import EventContent, RoomEvent
from roomkit.models.room import Room
from roomkit.models.task import DelegatedTaskResult, Observation

HostCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""Sends one verb and its arguments to the room owner; its result, as data."""


class SendEvent(BaseModel):
    """``send_event``: the arguments of ``RoomKit.send_event``."""

    room_id: str
    channel_id: str
    content: EventContent
    event_type: EventType = EventType.MESSAGE
    chain_depth: int = 0
    participant_id: str | None = None
    metadata: dict[str, Any] | None = None
    visibility: str = Visibility.ALL
    provider: str | None = None
    response_visibility: str | None = None
    created_at: datetime | None = None
    parent_event_id: str | None = None
    idempotency_key: str | None = None
    addressed_to: list[str] | None = None
    organization_id: str | None = None
    standalone: bool = False


class Delegate(BaseModel):
    """``delegate``: the arguments of ``RoomKit.delegate`` that are data;
    the task runs in the background, its id answered."""

    room_id: str
    agent_id: str
    task: str
    context: dict[str, Any] | None = None
    share_channels: list[str] | None = None
    notify: str | None = None
    require_structured_result: bool = False
    max_result_retries: int = 3
    post_status: bool = True
    brief: str | None = None
    conversation: int = 0
    hand_back: str = "relay"


class WaitTask(BaseModel):
    """``wait_task``: a task this host delegated, and how long to wait."""

    task_id: str
    timeout: float | None = None


class PostStatus(BaseModel):
    """``post_status``: an entry on the kit's status bus (RFC §19.8)."""

    agent_id: str
    action: str
    status: str
    detail: str = ""
    metadata: dict[str, Any] | None = None


class UpdateRoomMetadata(BaseModel):
    """``update_room_metadata``: keys merged into a room's metadata."""

    room_id: str
    metadata: dict[str, Any]
    organization_id: str | None = None


class AddObservation(BaseModel):
    """``add_observation``: an observation stored for its room (RFC §5.11)."""

    observation: Observation


VERBS: dict[str, type[BaseModel]] = {
    "send_event": SendEvent,
    "delegate": Delegate,
    "wait_task": WaitTask,
    "post_status": PostStatus,
    "update_room_metadata": UpdateRoomMetadata,
    "add_observation": AddObservation,
}
"""Every verb of a turn's host, by its name, with the record of its arguments."""


class TurnHost:
    """What a host's tool or hook on a turn's executor may ask of the room
    owner, each verb one call across the turn backend (RFC §10.6.5)."""

    def __init__(self, call: HostCaller) -> None:
        self._call = call
        self._open = True

    @property
    def open(self) -> bool:
        """Whether the turn it serves still runs."""
        return self._open

    def close(self) -> None:
        """The turn ended: the host answers no more verbs."""
        self._open = False

    async def send_event(
        self, room_id: str, channel_id: str, content: EventContent, **options: Any
    ) -> RoomEvent:
        """Send an event into a room, as ``RoomKit.send_event`` does: the
        same options, each a value the backend carries."""
        args = SendEvent(room_id=room_id, channel_id=channel_id, content=content, **options)
        return RoomEvent.model_validate(await self._verb("send_event", args))

    async def delegate(self, room_id: str, agent_id: str, task: str, **options: Any) -> str:
        """Delegate *task* to *agent_id* from *room_id*, in the background, as
        ``RoomKit.delegate`` does: the task's id, which :meth:`wait_task` waits."""
        args = Delegate(room_id=room_id, agent_id=agent_id, task=task, **options)
        return str(await self._verb("delegate", args))

    async def wait_task(
        self, task_id: str, timeout: float | None = None
    ) -> DelegatedTaskResult | None:
        """The result of a task this host delegated, once it ends; ``None``
        when *timeout* passes first, or for a task it does not know."""
        data = await self._verb("wait_task", WaitTask(task_id=task_id, timeout=timeout))
        return None if data is None else DelegatedTaskResult.model_validate(data)

    async def post_status(
        self,
        agent_id: str,
        action: str,
        status: str,
        *,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post an entry on the kit's status bus."""
        args = PostStatus(
            agent_id=agent_id, action=action, status=status, detail=detail, metadata=metadata
        )
        await self._verb("post_status", args)

    async def update_room_metadata(
        self, room_id: str, metadata: dict[str, Any], *, organization_id: str | None = None
    ) -> Room:
        """Merge *metadata* into the room's: the room as stored after."""
        args = UpdateRoomMetadata(
            room_id=room_id, metadata=metadata, organization_id=organization_id
        )
        return Room.model_validate(await self._verb("update_room_metadata", args))

    async def add_observation(self, observation: Observation) -> None:
        """Store *observation* for its room."""
        await self._verb("add_observation", AddObservation(observation=observation))

    async def _verb(self, verb: str, args: BaseModel) -> Any:
        if not self._open:
            raise RuntimeError(f"The turn this host served has ended: {verb} is not sent")
        return await self._call(verb, args.model_dump(mode="json"))


_TURN_HOST: contextvars.ContextVar[TurnHost | None] = contextvars.ContextVar(
    "roomkit_turn_host", default=None
)


def current_turn_host() -> TurnHost | None:
    """The host of the turn the current code runs in (RFC §10.6.5): inside a
    tool call of a turn and inside a hook its executor fires during it;
    ``None`` anywhere else, after the turn ended included."""
    host = _TURN_HOST.get()
    return host if host is not None and host.open else None


def set_turn_host(host: TurnHost | None) -> None:
    """Set the host of the turn the current context runs: an executor sets
    it for every turn it runs."""
    _TURN_HOST.set(host)
