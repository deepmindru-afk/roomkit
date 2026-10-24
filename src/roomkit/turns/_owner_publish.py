"""What the room owner publishes of what a turn's channel published on its
executor (RFC §10.6.2, §10.6.3): during the turn, from its ``ephemeral`` and
``framework_event`` frames; after it, from a ``room_publication``. Only what
names the turn's room and channel, and only the framework events a turn's
executor emits, are published; anything else is dropped."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.core.visibility import answers_reach
from roomkit.models.enums import Visibility
from roomkit.realtime.base import EphemeralEvent
from roomkit.turns.wire import Ephemeral

if TYPE_CHECKING:
    import contextvars

    from roomkit.turns.base import OwnerEvents
    from roomkit.turns.wire import FrameworkEventFrame, TurnRequest

logger = logging.getLogger("roomkit.turns")

TURN_FRAMEWORK_EVENTS = frozenset(
    {
        "tool_call",
        "before_tool_use",
        "user_input_required",
        "ai_response",
        "before_ai_generation",
        "hook_timeout",
    }
)
"""The framework events a turn's executor emits (RFC §8.2): the owner emits no
other a ``framework_event`` frame names."""


def names_its_room(
    room_id: str, channel_id: str, named_room: str | None, named: str | None
) -> bool:
    """Whether what a turn of *channel_id* in *room_id* published names that
    room and channel (or none)."""
    return named_room in (None, room_id) and named in (None, channel_id)


async def publish_frame(
    events: OwnerEvents,
    room_id: str,
    channel_id: str,
    frame: Ephemeral | FrameworkEventFrame,
    *,
    label: str,
    unscoped: bool,
) -> None:
    """Publish *frame*, published by a turn of *channel_id* in *room_id*
    (*label* names it in the log), as that channel in that room.

    An ephemeral is published only when the turn's answer reaches the whole
    room (RFC §8.4): its trigger is *unscoped*, and the channel's binding, as
    the room holds it at this publish, lets it (``events.reaches_room``). Its
    subscribers are not bound channels, so a turn whose answer the room
    narrows publishes none, from the moment the room narrows it."""
    if isinstance(frame, Ephemeral):
        if not (unscoped and await events.reaches_room(room_id, channel_id)):
            logger.debug("%s: its answer does not reach the room, ephemeral not published", label)
            return
        event = EphemeralEvent.from_dict(frame.event)
        if not names_its_room(room_id, channel_id, event.room_id, event.channel_id) or (
            frame.channel != f"room:{room_id}"
        ):
            logger.warning("%s published outside its room: dropped", label)
            return
        await events.publish(frame.channel, event)
        return
    if frame.event_type not in TURN_FRAMEWORK_EVENTS or not names_its_room(
        room_id, channel_id, frame.room_id, frame.channel_id
    ):
        logger.warning("%s emitted %s outside its room or kind: dropped", label, frame.event_type)
        return
    await events.emit(
        frame.event_type,
        room_id=room_id,
        channel_id=channel_id,
        event_id=frame.event_id,
        data=frame.data,
    )


class TurnPublisher:
    """What one turn published on its executor, published here in order by a
    task of its own: a handler that waits on the turn (an answer to an input
    request, a close) never stalls the frames it waits for."""

    def __init__(self, events: OwnerEvents | None, request: TurnRequest) -> None:
        self._events = events
        self._request = request
        self._unscoped = answers_reach(request.event) == Visibility.ALL
        self._queue: asyncio.Queue[Ephemeral | FrameworkEventFrame | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def publish_later(
        self, frame: Ephemeral | FrameworkEventFrame, waiting: contextvars.Context | None
    ) -> None:
        """Queue *frame*, the task started on the first in a copy of
        *waiting*, the context of the task waiting on the turn (RFC §10.6.5)."""
        if self._events is None:
            return
        if self._task is None:
            context = waiting.copy() if waiting is not None else None
            self._task = asyncio.get_running_loop().create_task(self._run(), context=context)
        self._queue.put_nowait(frame)

    def close(self) -> None:
        """What was queued is still published; nothing is added after."""
        self._queue.put_nowait(None)

    async def _run(self) -> None:
        while (frame := await self._queue.get()) is not None:
            await self._publish(frame)

    async def _publish(self, frame: Ephemeral | FrameworkEventFrame) -> None:
        """Publish on the owner what the turn published on its executor, as
        the turn's channel in the turn's room (RFC §10.6.2)."""
        assert self._events is not None
        request = self._request
        try:
            await publish_frame(
                self._events,
                request.room_id,
                request.channel_id,
                frame,
                label=f"Turn {request.turn_id}",
                unscoped=self._unscoped,
            )
        except Exception:
            # What the turn published never stops the turn.
            logger.exception("Turn %s: a %s could not be published", request.turn_id, frame.type)
