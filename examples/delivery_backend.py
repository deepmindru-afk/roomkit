"""Persistent delivery with InMemoryDeliveryBackend.

Demonstrates the delivery backend pattern: ``kit.deliver()`` enqueues an
item and returns ``queued`` at once; a background worker, started when the
kit is entered (``async with kit``), dequeues it, publishes the content into
the room and fires ``AFTER_DELIVER`` with the outcome. The script prints the
queue depth before and after, and waits for every ``AFTER_DELIVER``.

Without a delivery backend, ``kit.deliver()`` runs in-process and returns
the final outcome instead of ``queued``.

This example uses the in-memory backend and a mock AI (no external deps, no
key). For production, swap with ``RedisDeliveryBackend`` (see
``delivery_redis.py``).

Run with:
    uv run python examples/delivery_backend.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InMemoryDeliveryBackend,
    RoomKit,
    WaitForIdle,
    WebSocketChannel,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.mock import MockAIProvider

logger = setup_logging("delivery_backend")
logging.getLogger("roomkit.delivery").setLevel(logging.DEBUG)

NOTIFICATIONS = [
    "Background job 17 finished: the nightly report is ready.",
    "Background job 18 failed: the export server timed out.",
]


async def main() -> None:
    assistant = Agent(
        "agent-assistant",
        provider=MockAIProvider(
            responses=[
                "Good news: your nightly report is ready.",
                "Heads up: the export failed, I will retry it.",
            ]
        ),
        role="Assistant",
        system_prompt="Tell the user about background job results.",
    )

    # InMemoryDeliveryBackend: items are enqueued and processed by a
    # background worker task. Replace with RedisDeliveryBackend for
    # production persistence.
    backend = InMemoryDeliveryBackend()

    kit = RoomKit(delivery_strategy=WaitForIdle(buffer=1.0), delivery_backend=backend)

    delivered = asyncio.Event()
    outcomes: list[str] = []

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def on_delivered(event: RoomEvent, ctx: RoomContext) -> None:
        outcome = event.metadata.get("delivery_outcome", {})
        error = event.metadata.get("error")
        status = outcome.get("status", "unknown")
        logger.info(
            "AFTER_DELIVER item=%s status=%s%s",
            event.metadata.get("delivery_item_id"),
            status,
            f" error={error}" if error else "",
        )
        outcomes.append(status)
        if len(outcomes) == len(NOTIFICATIONS):
            delivered.set()

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)
    kit.register_channel(assistant)

    async def on_user_receives(_conn: str, event: RoomEvent) -> None:
        if isinstance(event.content, TextContent):
            logger.info("User sees [%s]: %s", event.source.channel_id, event.content.body)

    ws.register_connection("user-conn", on_user_receives, room_id="demo")

    async with kit:  # starts the delivery worker
        await kit.create_room(room_id="demo")
        await kit.attach_channel("demo", "ws-user")
        await kit.attach_channel("demo", "agent-assistant", category=ChannelCategory.INTELLIGENCE)

        logger.info("Queue depth before: %d", await backend.get_queue_depth())

        for text in NOTIFICATIONS:
            outcome = await kit.deliver("demo", text, channel_id="ws-user")
            logger.info("kit.deliver() -> %s (item %s)", outcome.status, outcome.delivery_item_id)

        logger.info("Queue depth after enqueue: %d", await backend.get_queue_depth())

        await asyncio.wait_for(delivered.wait(), timeout=15.0)

        logger.info("Queue depth after the worker ran: %d", await backend.get_queue_depth())
        dead = await backend.get_dead_letter_items()
        logger.info("Dead-lettered items: %d", len(dead))

    if any(status != "sent" for status in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
