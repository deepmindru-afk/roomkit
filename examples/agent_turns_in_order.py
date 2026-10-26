"""Give one agent several tasks in a room: it does them one after the other.

A lead hands a worker three tasks in the same room, each message carrying a
``delivery_key`` of the lead's own (a task id). By default an agent answers a
room's messages side by side: the turn for task 2 would start while task 1
runs, and read task 1's order still unanswered. With
``serialize_room_turns=True`` the worker's turns in the room run one after the
other, in the order the tasks arrived. Each turn reads the room as it stood
when its task arrived, plus the worker's answers since: task 2 reads task 1
done, and never task 3, which waits its own turn.

While task 1 runs, the lead cancels task 3 by its id: its turn never starts,
and its message still reaches the room.

Run with: uv run python examples/agent_turns_in_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
from typing import Any

from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    InboundMessage,
    InboundResult,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIContext, AIResponse, MockAIProvider

logger = setup_logging("agent_turns_in_order")


def _text(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "text", "") for part in content)


class Worker(MockAIProvider):
    """Takes a while on every task, and says what each turn reads."""

    async def generate(self, context: AIContext) -> AIResponse:
        conversation = [_text(message.content) for message in context.messages]
        task = conversation[-1]
        logger.info("Turn for %r starts, reading %s", task, conversation)
        await asyncio.sleep(0.3)
        return AIResponse(content=f"Done: {task}")


async def hand(kit: RoomKit, body: str, task_id: str) -> InboundResult:
    message = InboundMessage(
        channel_id="lead",
        sender_id="lead",
        content=TextContent(body=body),
        delivery_key=task_id,
    )
    return await kit.process_inbound(message, room_id="tasks", defer_delivery=True)


async def main() -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("lead"))
    kit.register_channel(Agent("worker", provider=Worker(), serialize_room_turns=True))
    await kit.create_room(room_id="tasks")
    await kit.attach_channel("tasks", "lead")
    await kit.attach_channel("tasks", "worker", category=ChannelCategory.INTELLIGENCE)

    try:
        handed = [
            await hand(kit, "Create the release card.", "task-1"),
            await hand(kit, "Write the changelog.", "task-2"),
            await hand(kit, "Tag the release.", "task-3"),
        ]
        # Task 3 waits behind tasks 1 and 2: cancelled, its turn never starts.
        cancelled = await kit.cancel_delivery("task-3", reason="lead_cancelled")
        logger.info("cancel_delivery('task-3') cancelled %d delivery", cancelled)

        for task_id, result in zip(("task-1", "task-2", "task-3"), handed, strict=True):
            assert result.delivery is not None
            done = await result.delivery.wait()
            logger.info("%s: cancelled=%s", task_id, done.cancellation_reason)

        for event in await kit.store.list_events("tasks"):
            if isinstance(event.content, TextContent):
                logger.info("In the room: %s: %r", event.source.channel_id, event.content.body)
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
