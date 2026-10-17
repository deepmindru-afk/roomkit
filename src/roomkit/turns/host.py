"""How code on a turn's executor reaches a room: :class:`TurnHost` (RFC §10.6.5).

A host's tool, or a hook the executor fires, runs where the turn runs, which
may be another process than the room owner's. It reads the room from the
store, and changes it only through the turn's host, read with
:func:`current_turn_host`: inside a call of a host tool during a turn and
inside a hook the executor fires during it; ``None`` anywhere else, between
two turns and on the owner included. Each verb is one ``host_call`` on the
turn's link, every argument is data, and the owner runs it as its own code
would, under what the turn gave the call (its room, its chain depth, who
asked) and with the reentrancy of its task reading the turn: a
``send_event`` that would wait on the room's own delivery from a task that
delivery needs returns with its completion detached.

A host acts for its turn's room: a verb naming another room, or another
organization than the room's, is refused. The verbs are enumerated here, each
with the record its arguments cross as; the owner validates what it receives
against it, and refuses an argument the record does not name.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from roomkit.models.enums import EventType, Visibility
from roomkit.models.event import EventContent, RoomEvent
from roomkit.models.room import Room
from roomkit.models.task import DelegatedTaskResult, Observation
from roomkit.turns._codec import rebuild_error
from roomkit.turns.wire import ErrorRecord

HostCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""Sends one verb and its arguments to the room owner; its result, as data."""


class _Verb(BaseModel):
    """A verb's arguments: an argument no record names is refused."""

    model_config = ConfigDict(extra="forbid")


class SendEvent(_Verb):
    """``send_event``: the arguments of ``RoomKit.send_event``. The event is
    at least as deep as the response of the turn that sends it (RFC §8.3)."""

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


class Delegate(_Verb):
    """``delegate``: the arguments of ``RoomKit.delegate`` a background task
    reads, each data; the task's id answered."""

    room_id: str
    agent_id: str
    task: str
    context: dict[str, Any] | None = None
    share_channels: list[str] | None = None
    notify: str | None = None
    post_status: bool = True
    brief: str | None = None
    conversation: int = 0
    hand_back: str = "relay"


class WaitTask(_Verb):
    """``wait_task``: a task a host of the room delegated, and how long to wait."""

    task_id: str
    timeout: float | None = None


class PostStatus(_Verb):
    """``post_status``: an entry on the kit's status bus (RFC §19.8)."""

    agent_id: str
    action: str
    status: str
    detail: str = ""
    metadata: dict[str, Any] | None = None


class UpdateRoomMetadata(_Verb):
    """``update_room_metadata``: keys merged into a room's metadata."""

    room_id: str
    metadata: dict[str, Any]
    organization_id: str | None = None


class AnswerInput(_Verb):
    """``answer_input``: an answer the channel's human-input handler received
    for a request of the turn, which the owner records."""

    request_id: str
    value: str


class RejectInput(_Verb):
    """``reject_input``: a rejection the channel's handler received."""

    request_id: str
    reason: str = ""


class AddObservation(_Verb):
    """``add_observation``: an observation stored for its room (RFC §5.11)."""

    observation: Observation


VERBS: dict[str, type[_Verb]] = {
    "send_event": SendEvent,
    "delegate": Delegate,
    "wait_task": WaitTask,
    "post_status": PostStatus,
    "update_room_metadata": UpdateRoomMetadata,
    "add_observation": AddObservation,
    "answer_input": AnswerInput,
    "reject_input": RejectInput,
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

    def _close(self) -> None:
        """The turn ended: the host answers no more verbs. The executor's."""
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
        """The result of a task a host of this room delegated, once it ends,
        its failure kept (``exception``, RFC §23.3); ``None`` when *timeout*
        passes first, or for a task the room's hosts did not delegate."""
        data = await self._verb("wait_task", WaitTask(task_id=task_id, timeout=timeout))
        if data is None:
            return None
        result = DelegatedTaskResult.model_validate(data["result"])
        if data.get("exception") is not None:
            failure = rebuild_error(
                ErrorRecord.model_validate(data["exception"]), where="the room owner"
            )
            if isinstance(failure, Exception):
                result.exception = failure
        return result

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

    async def _verb(self, verb: str, args: _Verb) -> Any:
        if not self._open:
            raise RuntimeError(f"The turn this host served has ended: {verb} is not sent")
        return await self._call(verb, args.model_dump(mode="json"))


_TURN_HOST: contextvars.ContextVar[TurnHost | None] = contextvars.ContextVar(
    "roomkit_turn_host", default=None
)


def current_turn_host() -> TurnHost | None:
    """The host of the turn the current code runs in (RFC §10.6.5): inside a
    call of a host tool during a turn and inside a hook its executor fires
    during it; ``None`` anywhere else, after the turn ended and in a tool the
    room owner serves included."""
    host = _TURN_HOST.get()
    return host if host is not None and host.open else None


def set_turn_host(host: TurnHost | None) -> None:
    """Set the host of the turn the current context runs: an executor sets
    it for every turn it runs."""
    _TURN_HOST.set(host)
