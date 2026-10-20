"""Cancel one of two tasks running in the same room, by a key of your own.

A lead hands an agent two tasks in one room. Each message carries a
``delivery_key`` the lead chose (a task id), so the lead can cancel one task
later without holding anything but that id: ``kit.cancel_delivery(key)``. The
cancelled task's turn stops; the other task runs to its end; the cancelled
task's message still reaches the room (its AFTER_BROADCAST hook fires).

Run with: uv run python examples/cancel_delivery_by_key.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider

logger = setup_logging("cancel_delivery_by_key")


class SlowWorker(MockAIProvider):
    """Takes a while on every task, the way a real agent does."""

    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(0.5)
        return AIResponse(content="Task done.")


def task(body: str, task_id: str) -> InboundMessage:
    return InboundMessage(
        channel_id="lead",
        sender_id="lead",
        content=TextContent(body=body),
        delivery_key=task_id,
    )


async def main() -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("lead"))
    kit.register_channel(Agent("worker", provider=SlowWorker()))
    await kit.create_room(room_id="tasks")
    await kit.attach_channel("tasks", "lead")
    await kit.attach_channel("tasks", "worker", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def reached(event: RoomEvent, ctx: object) -> None:
        if isinstance(event.content, TextContent):
            logger.info("In the room: %r", event.content.body)

    try:
        first = asyncio.create_task(
            kit.process_inbound(task("Summarize the report.", "task-1"), room_id="tasks")
        )
        second = asyncio.create_task(
            kit.process_inbound(task("Translate the report.", "task-2"), room_id="tasks")
        )
        await asyncio.sleep(0.1)

        # Only the id is needed: an agent's tool can pass it along as text.
        cancelled = await kit.cancel_delivery("task-2", reason="lead_cancelled")
        logger.info("cancel_delivery('task-2') cancelled %d delivery", cancelled)

        done_first, done_second = await asyncio.gather(first, second)
        logger.info(
            "task-1: cancelled=%s, answered=%s",
            done_first.cancellation_reason,
            [e.content.body for e in done_first.response_events],
        )
        logger.info(
            "task-2: cancelled=%s, answered=%s",
            done_second.cancellation_reason,
            [e.content.body for e in done_second.response_events],
        )
        logger.info("cancel_delivery('task-2') once over: %d", await kit.cancel_delivery("task-2"))
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
