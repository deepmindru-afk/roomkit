"""One agent shared by two rooms: what its turns read, and a Stop scoped to one room.

A host binds one Agent to many rooms. Three channel settings keep each turn
the host's:
- Agent(identity_in_prompt=False): the host renders the agent's identity in
  its own prompt, and RoomKit appends no identity block (role, scope and
  language stay readable on the agent)
- AIChannel(describe_empty_event=...): an upload stored without a caption
  extracts to nothing; the describer says what it is, so the turn reads it
  instead of omitting it
- steer(Cancel(...), room_id=...): a member's Stop cancels the turns of their
  room only, and says how many it reached; a turn in another room answers

Run with:
    uv run python examples/ai_shared_agent_rooms.py
"""

from __future__ import annotations

import asyncio

from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.models.steering import Cancel
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider

logger = setup_logging("ai_shared_agent_rooms")

HOST_PROMPT = "## Identity\nYou are Support, for the billing team. Answer in French.\n"


def describe_upload(event: RoomEvent) -> str | None:
    """An upload stored with an empty body, its files in the event's metadata."""
    files = event.metadata.get("attachments") or []
    if not files:
        return None
    names = ", ".join(str(item.get("name")) for item in files)
    return f"The member sent attachments without a caption: {names}."


class HeldModel(MockAIProvider):
    """Answers once released, so two rooms' turns run side by side."""

    def __init__(self) -> None:
        super().__init__(ai_responses=[AIResponse(content="Votre facture est réglée.")])
        self.release = asyncio.Event()
        self.waiting = 0

    async def generate(self, context: AIContext) -> AIResponse:
        self.waiting += 1
        await self.release.wait()
        return await super().generate(context)


async def ask(kit: RoomKit, room_id: str, body: str, **metadata: object) -> None:
    await kit.process_inbound(
        InboundMessage(
            channel_id=f"member-{room_id}",
            sender_id=f"user-{room_id}",
            content=TextContent(body=body),
            metadata=dict(metadata),
        ),
        room_id=room_id,
    )


async def main() -> None:
    model = HeldModel()
    agent = Agent(
        "support",
        provider=model,
        system_prompt=HOST_PROMPT,
        role="Billing support",
        language="French",
        identity_in_prompt=False,
        describe_empty_event=describe_upload,
    )
    kit = RoomKit()
    kit.register_channel(agent)
    for room_id in ("A", "B"):
        kit.register_channel(WebSocketChannel(f"member-{room_id}"))
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, f"member-{room_id}")
        await kit.attach_channel(room_id, "support", category=ChannelCategory.INTELLIGENCE)

    # Room A's member uploads a file without a caption; room B's asks a question.
    turn_a = asyncio.create_task(ask(kit, "A", "", attachments=[{"name": "invoice.pdf"}]))
    turn_b = asyncio.create_task(ask(kit, "B", "Ma facture est-elle réglée ?"))
    while model.waiting < 2:
        await asyncio.sleep(0.01)

    # Room A's member presses Stop: only room A's turn is cancelled.
    reached = agent.steer(Cancel(reason="stop pressed"), room_id="A")
    logger.info("Stop in room A reached %d turn(s)", reached)
    model.release.set()
    await asyncio.gather(turn_a, turn_b)

    for call in model.calls:
        logger.info("the model read: %r", call.messages[-1].content)
        identity = "Agent Identity" in (call.system_prompt or "")
        logger.info("RoomKit's identity block in its prompt: %s", identity)
    for room_id in ("A", "B"):
        said = [
            e.content.body
            for e in await kit.store.list_events(room_id)
            if e.source.channel_id == "support" and isinstance(e.content, TextContent)
        ]
        logger.info("room %s, the agent said: %s", room_id, said or "nothing (stopped)")
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
