"""Talking and working: long work runs as a task, off the floor (RFC §19.7.5 rule 19).

An on-call engineer asks @comms to go through every support ticket, a long
job, then asks @sre a quick question one second later. The same two
messages run three times:

1. Without background work, @comms does the job in its turn: the floor is
   held for the whole of it, and the quick question waits behind it.
2. With ``kit.setup_background(comms, worker="comms-worker")``, @comms
   starts the job with ``work_in_background``, says it is on it and ends its
   turn. @sre answers at once; the result comes back at the front of the
   queue, and @comms says it in a turn of its own (``hand_back="relay"``).
3. The same with ``hand_back="post"``: the worker's findings are posted as
   @comms's message as they are, with no turn of @comms's model.

Each run prints when each agent spoke. The models are scripted and the long
job is a tool that takes three seconds: no key needed.

Run with:
    uv run python examples/discussion_background.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    Discussion,
    EventStatus,
    EventType,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks import BACKGROUND_TOOL

setup_logging("discussion_background")
logging.getLogger("roomkit").setLevel(logging.WARNING)

JOB_SECONDS = 3.0
FINDINGS = "12 tickets since 14:07, all Canadian addresses with no province: checkout fails."


class SupportTickets:
    """``list_support_tickets``: reading every ticket takes a while."""

    definition: dict[str, Any] = {
        "name": "list_support_tickets",
        "description": "Every support ticket of the day.",
        "parameters": {"type": "object", "properties": {}},
    }

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        await asyncio.sleep(JOB_SECONDS)
        return FINDINGS


def _call(name: str, **arguments: Any) -> AIResponse:
    return AIResponse(content="", tool_calls=[AIToolCall(id=name, name=name, arguments=arguments)])


def _comms(background: str | None) -> Agent:
    """@comms: does the job in its turn, or starts it as a task."""
    if background is None:
        answers = [_call("list_support_tickets"), AIResponse(content=f"Done: {FINDINGS}")]
        tools: list[Any] = [SupportTickets()]
    else:
        answers = [
            _call(BACKGROUND_TOOL, task="Go through every support ticket of the day."),
            AIResponse(content="On it, going through the tickets now."),
            AIResponse(content=f"The tickets are in: {FINDINGS}"),  # the relayed result
        ]
        tools = []
    return Agent(
        "comms",
        provider=MockAIProvider(ai_responses=answers),
        description="reads the support tickets",
        tools=tools,
    )


def _worker() -> Agent:
    """Does @comms's background work, with the slow tool."""
    answers = [_call("list_support_tickets"), AIResponse(content=FINDINGS)]
    return Agent(
        "comms-worker", provider=MockAIProvider(ai_responses=answers), tools=[SupportTickets()]
    )


async def run(background: str | None) -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("oncall"))
    comms = _comms(background)
    sre = Agent(
        "sre",
        provider=MockAIProvider(ai_responses=[AIResponse(content="Error rate: 6%, flat.")]),
        description="reads production metrics",
    )
    if background is not None:
        kit.register_channel(_worker())
        kit.setup_background(comms, worker="comms-worker", hand_back=background)
    await kit.create_room(room_id="incident", orchestration=Discussion([comms, sre]))
    await kit.attach_channel("incident", "oncall")

    start, started = time.monotonic(), datetime.now(UTC)

    async def say(text: str) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id="oncall", sender_id="oncall", content=TextContent(body=text))
        )

    await say("@comms please go through all the support tickets")
    await asyncio.sleep(1.0)
    await say("@sre quick one: the checkout error rate?")
    seen: dict[str, float] = {}
    while time.monotonic() - start < 20:
        await asyncio.sleep(0.05)
        events = await kit.store.list_events("incident", limit=100)
        for event in events:
            if (
                event.type == EventType.MESSAGE
                and event.status == EventStatus.DELIVERED
                and event.source.channel_id != "oncall"
                and event.id not in seen
            ):
                seen[event.id] = (event.created_at - started).total_seconds()
                text = event.content.body.replace("\n", " ")
                print(f"  {seen[event.id]:4.1f} s  @{event.source.channel_id:<6} {text[:90]}")
        if len(seen) >= (2 if background is None else 3):
            break
    await kit.close()


async def main() -> None:
    for title, background in (
        ("1. Without background work: the long job holds the floor", None),
        ("2. Background work, the result relayed by @comms", "relay"),
        ("3. Background work, the result posted as it is", "post"),
    ):
        print(f"\n{title}")
        await run(background)


if __name__ == "__main__":
    asyncio.run(main())
