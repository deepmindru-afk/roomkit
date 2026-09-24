"""Directing an agent without putting words in anyone's mouth (RFC §10.1.1).

An application sometimes needs an agent to speak because of something nobody
in the room said: a handoff asking the new agent to introduce itself, a
scheduled nudge. Sent as an ordinary inbound message, that direction is stored
as a participant's words and read by the model as a user turn. Sent as an
``INSTRUCTION``, it reaches only the agent it addresses, is never stored, and
the agent's reply records it.

The example sends one of each and prints what the room kept.

Run with:
    uv run python examples/instruction_event.py
"""

from __future__ import annotations

import asyncio

from roomkit import AIChannel, InboundMessage, RoomKit
from roomkit.channels import WebSocketChannel
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider

ROOM = "support"


async def main() -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(
        AIChannel("advisor", provider=MockAIProvider(["Hello, I'm the advisor. How can I help?"]))
    )
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "ws", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "advisor", category=ChannelCategory.INTELLIGENCE)

    # The application directs the advisor. Nobody in the room says this.
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws",
            sender_id="system",
            event_type=EventType.INSTRUCTION,
            content=TextContent(body="You just took over the call. Introduce yourself."),
            addressed_to=["advisor"],
        ),
        room_id=ROOM,
    )

    print("What the room holds:")
    for event in await kit.store.list_events(ROOM):
        if event.type == EventType.MESSAGE:
            body = event.content.body  # type: ignore[union-attr]
            print(f"  [{event.source.channel_id}] {body}")
            print(f"      instruction: {event.metadata.get('instruction')!r}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
