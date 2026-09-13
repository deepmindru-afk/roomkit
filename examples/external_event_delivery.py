"""Target an agent with an external event, then safely replay its publication.

Run without network access or API keys:
    uv run python examples/external_event_delivery.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    AgentResponsePolicy,
    AIChannel,
    ChannelCategory,
    RoomKit,
    WebSocketChannel,
)
from roomkit.providers.ai.mock import MockAIProvider

logger = logging.getLogger(__name__)


async def main() -> None:
    provider_a = MockAIProvider(responses=["The background task finished successfully."])
    provider_b = MockAIProvider(responses=["No request expected."])
    async with RoomKit() as kit:
        kit.register_channel(WebSocketChannel("text"))
        kit.register_channel(AIChannel("agent-a", provider=provider_a))
        kit.register_channel(AIChannel("agent-b", provider=provider_b))
        await kit.create_room(
            room_id="conversation",
            agent_response_policy=AgentResponsePolicy.ADDRESSED_ONLY,
        )
        await kit.attach_channel("conversation", "text")
        for channel_id in ("agent-a", "agent-b"):
            await kit.attach_channel(
                "conversation", channel_id, category=ChannelCategory.INTELLIGENCE
            )

        results = await asyncio.gather(
            *[
                kit.deliver(
                    "conversation",
                    "Background task 42 has finished.",
                    channel_id="text",
                    addressed_to=["agent-a"],
                    idempotency_key="task.finished:42",
                    metadata={"external_event": "task.finished", "task_id": "42"},
                )
                for _ in range(2)
            ]
        )
        assert results[0].event_id == results[1].event_id
        assert sum(result.duplicate for result in results) == 1
        assert len(provider_a.calls) == 1
        assert len(provider_b.calls) == 0
        for result in results:
            logger.info(
                "status=%s event=%s duplicate=%s turn_complete=%s",
                result.status,
                result.event_id,
                result.duplicate,
                result.turn_complete,
            )
        logger.info(
            "Agent A: %d turn; agent B: %d turns", len(provider_a.calls), len(provider_b.calls)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
