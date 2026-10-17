"""What the room owner publishes of what a turn's channel published on its
executor (RFC §10.6.2, §10.6.3): during the turn, from its ``ephemeral`` and
``framework_event`` frames; after it, from a ``room_publication``. Only what
names the turn's room and channel, and only the framework events a turn's
executor emits, are published; anything else is dropped."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.realtime.base import EphemeralEvent
from roomkit.turns.wire import Ephemeral

if TYPE_CHECKING:
    from roomkit.turns.base import OwnerEvents
    from roomkit.turns.wire import FrameworkEventFrame

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
) -> None:
    """Publish *frame*, published by a turn of *channel_id* in *room_id*
    (*label* names it in the log), as that channel in that room."""
    if isinstance(frame, Ephemeral):
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
