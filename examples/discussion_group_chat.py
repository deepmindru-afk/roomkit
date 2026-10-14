"""A group chat of agents and a person: the Discussion strategy.

An on-call engineer and three agents hold one conversation about an incident.
Nobody fixes the order in advance: who speaks next follows from who was
addressed, one agent at a time (RFC §19.7.5).

1. "The checkout API returns 500s since 14:00. What happened?" names nobody,
   so it asks ``everyone``, in the host's order: the investigator opens and
   asks ``@dev`` to confirm the deploy; dev, already queued, answers both in
   one turn and asks ``@sre`` to roll back; sre asks ``@oncall`` for approval,
   and the discussion waits for that person.
2. "Yes, roll it back." names nobody either: it answers the agent that asked,
   sre alone.
3. "@dev please open a fix ticket" asks dev alone.

The agents' channel ids are ``agent:<uuid>``, as an application that keys
its agents by record id names them: such an id cannot be a name
(``@agent:2b7d…`` reads as ``@agent``), so each agent is addressed by its
handle, here its ``name`` (``@dev``, matched ignoring case). The notes, the
transcript and the logs below name agents by handle; the queue and
``addressed_to`` keep channel ids.

Each turn reads the room as it is when the turn starts, and every change of
the speak queue reaches ``ON_SPEAK_QUEUE``: a console follows it to show who
speaks, who is next and whether the room waits for a person. The models are
scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/discussion_group_chat.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    Discussion,
    EventType,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    SpeakQueueEvent,
    TextContent,
    WebSocketChannel,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider

logger = setup_logging("example.discussion_group_chat")

ROOM = "incident-room"


def scripted(*answers: str) -> MockAIProvider:
    """A model that answers its turns with *answers*, in order."""
    return MockAIProvider(ai_responses=[AIResponse(content=a) for a in answers])


async def until_quiet(kit: RoomKit) -> None:
    """Wait until no agent speaks and none can take a turn."""
    quiet = 0
    while quiet < 3:
        await asyncio.sleep(0.02)
        queue = kit.speak_queue(ROOM)
        assert queue is not None
        idle = not queue.queue or queue.waiting or queue.over
        quiet = quiet + 1 if queue.speaking is None and idle else 0


def incident_team() -> list[Agent]:
    """The investigator, dev and sre, keyed by record id as an application
    keys them, each with the scripted answers of the incident."""
    investigator = Agent(
        "agent:2b7d9c41-5e0a-4f3b-9d6e-1a8c7f20e5b3",
        provider=scripted(
            "Error logs show a null pointer in the payment client from 13:55. "
            "@dev can you confirm what was deployed then?"
        ),
        name="Investigator",
        role="Investigator",
        description="Reads logs and metrics first",
    )
    dev = Agent(
        "agent:8e41f0d2-6b3c-4a7e-b5d9-3c2f1e0a9d84",
        provider=scripted(
            "Confirmed: release 4.2 went out at 13:55 and touched the payment client. "
            "@sre can you roll it back?",
            "Ticket PAY-101 opened for the null check in the payment client.",
        ),
        name="Dev",
        role="Developer",
        description="Knows the code and the releases",
    )
    sre = Agent(
        "agent:c5a3e7b1-0f2d-4e6c-8a9b-7d1f3e5c2a60",
        provider=scripted(
            "Rolling back production needs approval. @oncall may I roll back release 4.2?",
            "Rolled back to 4.1. The error rate is back to 0.1%.",
        ),
        name="SRE",
        role="Site reliability engineer",
        description="Runs production changes",
    )
    return [investigator, dev, sre]


def named(handles: dict[str, str], channel_ids: Sequence[str]) -> str:
    """Agents as the room addresses them: ``@`` and each one's handle."""
    return ",".join(f"@{handles.get(c, c)}" for c in channel_ids) or "-"


async def main() -> None:
    team = incident_team()
    discussion = Discussion(
        agents=team,
        people=["oncall"],
        everyone=[agent.channel_id for agent in team],
    )
    # Each agent's handle, by channel id: Investigator, Dev, SRE.
    handles = discussion.handles()

    kit = RoomKit()

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: RoomContext) -> None:
        queue = event.queue
        speaking = [queue.speaking] if queue.speaking else []
        logger.info(
            "  [queue] %s %s | speaking=%s next=%s waiting=%s",
            event.change,
            named(handles, event.channel_ids),
            named(handles, speaking),
            named(handles, queue.queue),
            queue.waiting,
        )

    oncall = WebSocketChannel("oncall")

    async def on_receive(_connection: str, event: RoomEvent) -> None:
        if event.type == EventType.MESSAGE and event.source.channel_id != "oncall":
            author = named(handles, [event.source.channel_id])
            logger.info("%s: %s", author, event.content.body)  # type: ignore[union-attr]

    oncall.register_connection("laptop", on_receive, room_id=ROOM)
    kit.register_channel(oncall)

    await kit.create_room(room_id=ROOM, orchestration=discussion)
    await kit.attach_channel(ROOM, "oncall")

    for said in (
        "The checkout API returns 500s since 14:00. What happened?",
        "Yes, roll it back.",
        "@dev please open a fix ticket",
    ):
        logger.info("oncall: %s", said)
        await kit.process_inbound(
            InboundMessage(channel_id="oncall", sender_id="oncall", content=TextContent(body=said))
        )
        await until_quiet(kit)

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
