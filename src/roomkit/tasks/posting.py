"""A task's result posted to the room as it is (RFC §23.3 step 8, ``hand_back``).

``relay``, the default, tells the notified agent the result (see
:mod:`roomkit.tasks.handback`), and the agent says it in a turn of its own.
``post`` publishes it instead as a message of the notified agent, without a
turn of its model: at once, and as the worker wrote it. The message names
the task before the result, so the room tells what it answers, and goes
through the room's pipeline as the agent's response would. In a room a
discussion holds, it takes its place at the front of the speak queue and is
posted when its turn comes, never cutting a turn under way (RFC §19.7.5 rule
19); its names ask no one, its worker having written it outside the room.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit._text import quoted
from roomkit.models.enums import EventStatus, EventType, TaskStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTaskResult

logger = logging.getLogger("roomkit.tasks")

HAND_BACKS = ("relay", "post")
"""How a background task's result reaches the room."""

_TASK_NAME_CHARS = 200


def posted_text(result: DelegatedTaskResult, task: str) -> str | None:
    """What a posted result says: the task (*task*, quoted), then the
    worker's output; that the task failed or was cancelled, without its
    error (RFC §9.3); ``None`` for a completed task with nothing to say."""
    named = f" {quoted(task, _TASK_NAME_CHARS)}" if task.strip() else ""
    if result.status == TaskStatus.COMPLETED:
        output = (result.output or "").strip()
        return f"Result of the task{named}:\n\n{output}" if output else None
    if result.status == TaskStatus.CANCELLED:
        return f"The task{named} was cancelled."
    return f"The task{named} failed."


async def post_result(
    kit: RoomKit,
    room_id: str,
    agent_id: str,
    text: str,
    *,
    chain_depth: int,
    metadata: dict[str, Any],
) -> None:
    """Post *text* in *room_id* as *agent_id*'s message, at *chain_depth*,
    with *metadata*: through the discussion's speak queue when one holds
    the room, else at once. A closing framework or an agent no longer
    attached posts nothing, logged."""
    if kit._closed:
        logger.info("Result for %s in room %s not posted: closing", agent_id, room_id)
        return
    binding = await kit.store.get_binding(room_id, agent_id)
    if binding is None:
        logger.info("Result for %s not posted: not attached to room %s", agent_id, room_id)
        return
    event = RoomEvent(
        room_id=room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id=agent_id, channel_type=binding.channel_type),
        content=TextContent(body=text),
        chain_depth=chain_depth,
        status=EventStatus.DELIVERED,
        metadata=metadata,
        addressed_to=[],
    )
    if await kit._queue_discussion_post(room_id, event):
        return
    await send_posted(kit, event)


async def send_posted(kit: RoomKit, event: RoomEvent) -> None:
    """Commit a posted result, *event*, as its agent's message: through the
    room's pipeline (``BEFORE_BROADCAST``, the agent's right to write,
    storage, broadcast), addressed to no one. A closing framework posts
    nothing, logged: a post held as it closes is cut once started."""
    if kit._closed:
        logger.info("Result in room %s not posted: closing", event.room_id)
        return
    await kit.send_event(
        event.room_id,
        event.source.channel_id,
        event.content,
        chain_depth=event.chain_depth,
        metadata=event.metadata,
        addressed_to=[],
    )
