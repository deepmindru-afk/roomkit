"""The host writes to a room's people in its own name: kit.post().

    shop backend ──post()──▶ Room ──┬──▶ SMSChannel "sms"  (the customer)
                                    └──▶ Agent "assistant"  (answers the customer)

A shop's backend texts a customer that their order shipped. The message is the
host's own words, not an agent's: it reaches the customer's phone and no agent
answers it. When the customer replies, the agent answers as usual, and it
reads the backend's message signed "Acme Support". Later the backend asks the
agent, in front of the customer, to confirm the address: it names the agent.

What post() saves you: the host channel (a transport of type HOST) is
registered and attached on first use, and the post solicits no agent unless it
names one (RFC §10.5).

No credentials: the SMS provider is RoomKit's mock, which prints what would
have been sent, and the agent is a mock AI.

Run with:
    uv run python examples/host_post.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    ChannelCategory,
    InboundMessage,
    Participant,
    RoomEvent,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.channels.agent import Agent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

CUSTOMER = "+15550100001"

logger = setup_logging("host_post")


class PrintingSMSProvider(MockSMSProvider):
    """The mock, made visible: what would leave shows up on the terminal."""

    async def send(self, event: RoomEvent, to: str, from_: str | None = None):
        logger.info("📱 → %s: %s", to, event.content.body)
        return await super().send(event, to, from_)


async def main() -> None:
    ai = MockAIProvider(
        responses=[
            "It left our warehouse this morning; you'll get it tomorrow before noon.",
            "Marie, can you confirm the delivery address is 12 rue des Lilas?",
        ]
    )
    kit = RoomKit()
    kit.register_channel(SMSChannel("sms", provider=PrintingSMSProvider()))
    kit.register_channel(Agent("assistant", provider=ai, role="Shop assistant"))

    await kit.create_room(room_id="order-1042")
    await kit.attach_channel("order-1042", "sms", metadata={"phone_number": CUSTOMER})
    await kit.attach_channel("order-1042", "assistant", category=ChannelCategory.INTELLIGENCE)
    for participant_id, channel_id, name in (
        (CUSTOMER, "sms", "Marie"),
        ("desk", "host", "Acme Support"),
    ):
        await kit.store.add_participant(
            Participant(
                id=participant_id, room_id="order-1042", channel_id=channel_id, display_name=name
            )
        )

    try:
        # 1. The backend's notice: texted to Marie, no agent answers it.
        await kit.post(
            "order-1042",
            "Your order #1042 shipped today.",
            participant_id="desk",
            metadata={"job": "shipping-notice"},
        )
        logger.info("agent turns after the notice: %d", len(ai.calls))

        # 2. Marie replies: the agent answers her as usual.
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id=CUSTOMER,
                content=TextContent(body="When will it arrive?"),
            ),
            room_id="order-1042",
        )
        notice = next(m.content for m in ai.calls[-1].messages if "shipped" in str(m.content))
        logger.info("the agent read the notice as: %s", notice)

        # 3. The backend asks the agent, in front of Marie, to act: it names it.
        # (Work for an agent alone, kept from Marie, is a kit.delegate().)
        await kit.post(
            "order-1042",
            "Assistant, please confirm the delivery address with Marie.",
            participant_id="desk",
            addressed_to=["assistant"],
        )
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
