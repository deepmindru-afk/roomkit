"""Who takes part: an agent that joins a meeting knows who is there.

An agent that joins a meeting where Alice speaks while Bob listens would read
Alice's words as an anonymous user's until Bob says something, and would not
know Bob is there. Who takes part is read from the room's roster, silent
people included: when two people or more are on it, every turn is labelled
from the first ("Alice: ...") and the turn's notes name them, so the agent can
address Bob before he has said a word.

A ConferenceChannel keeps the roster itself. An integration that bridges a
meeting into one stream writes it with ``kit.add_member``, one member per
person present, as this example does; ``remove_member`` takes them off.

Run with (no API key needed — MockAIProvider records what it was asked):
    uv run python examples/ai_people_taking_part.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext
from roomkit.providers.ai.mock import MockAIProvider

ROOM = "weekly-sync"
CHANNEL = "meeting"


def show(title: str, context: AIContext) -> None:
    """Print the turn's input and the notes that ride it."""
    print(f"\n== {title}")
    print(f"  {context.messages[-1].content}")


async def say(kit: RoomKit, sender_id: str, body: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id=CHANNEL, sender_id=sender_id, content=TextContent(body=body))
    )


async def main() -> None:
    kit = RoomKit()
    provider = MockAIProvider(responses=["Alice and Bob are here.", "Just you now, Alice."])
    kit.register_channel(WebSocketChannel(CHANNEL))
    kit.register_channel(
        AIChannel("assistant", provider=provider, system_prompt="You take part in a meeting.")
    )
    await kit.create_room(room_id=ROOM)
    # Several people on this one channel: a group binding (RFC §10.4).
    await kit.attach_channel(ROOM, CHANNEL, group=True)
    await kit.attach_channel(ROOM, "assistant", category=ChannelCategory.INTELLIGENCE)

    # The meeting's people, as the integration knows them when the agent joins.
    await kit.add_member(ROOM, CHANNEL, "u-alice", display_name="Alice")
    await kit.add_member(ROOM, CHANNEL, "u-bob", display_name="Bob")

    # 1. Alice speaks first, Bob has said nothing: her turn is labelled and
    #    the notes name both of them.
    await say(kit, "u-alice", "Who is here?")
    show("two people on the roster, one has spoken", provider.calls[-1])

    # 2. Bob leaves: one person on the roster, no label and no note.
    await kit.remove_member(ROOM, "u-bob")
    await say(kit, "u-alice", "And now?")
    show("Bob left, one person on the roster", provider.calls[-1])

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
