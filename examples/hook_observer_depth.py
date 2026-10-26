"""Two hooks that ask an agent again after its answers: one ends at the depth
limit, the other goes on as long as it decides.

A moderation hook reminds the agent of a rule after each of its turns, from
the hook's own body, once a moderation check took its time. The agent answers
each reminder, which fires the hook again. What a hook's body sends is at
least as deep as the turn was when its trigger fired, once the turn is over
too, so each reminder is one deeper and the loop ends at ``max_chain_depth``.

A quality gate judges each answer and, while it fails, relaunches the agent
from a task it starts, five times at most. A slower audit hook observes the
same answers, so the agent's turn still runs when the gate relaunches it.
Work a hook's body starts begins a new chain once the body returned: every
relaunch is at depth 0 and the agent answers all five; the gate's own limit
ends it.

Run with: uv run python examples/hook_observer_depth.py
Uses a mock agent; no credentials or external calls required.
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
    HookExecution,
    HookTrigger,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import MockAIProvider

logger = setup_logging("hook_observer_depth")

MAX_CHAIN_DEPTH = 5
GATE_ATTEMPTS = 5


async def room() -> RoomKit:
    kit = RoomKit(max_chain_depth=MAX_CHAIN_DEPTH)
    kit.register_channel(WebSocketChannel("chat"))
    kit.register_channel(Agent("writer", provider=MockAIProvider(responses=["Here it is."])))
    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "chat")
    await kit.attach_channel("support", "writer", category=ChannelCategory.INTELLIGENCE)
    return kit


async def ask(kit: RoomKit, body: str) -> None:
    await kit.send_event("support", "chat", TextContent(body=body))


async def moderation_reminders() -> None:
    """A hook that tells the agent again from its body ends at the limit."""
    kit = await room()
    reminders: list[int] = []
    ended = asyncio.Event()

    @kit.on("chain_depth_exceeded")
    async def limited(event: object) -> None:
        ended.set()

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION, execution=HookExecution.ASYNC, name="moderation")
    async def moderation(event: Any, ctx: Any) -> None:
        if len(reminders) >= 20:  # a guard, should the limit not end it
            ended.set()
            return
        await asyncio.sleep(0.05)  # the moderation check: the turn is over by now
        reminder = await kit.post(
            "support", "Reminder: cite your sources.", addressed_to=["writer"]
        )
        reminders.append(reminder.chain_depth)
        logger.info("Moderation reminds the agent, %d deep", reminder.chain_depth)

    try:
        await ask(kit, "What does RoomKit do?")
        await asyncio.wait_for(ended.wait(), 5)
        logger.info("Reminders at depths %s", reminders)
        assert reminders == [1, 2, 3, 4], reminders
    finally:
        await kit.close()


async def quality_gate_relaunches() -> None:
    """A relaunch from a task the gate starts begins a new chain each time."""
    kit = await room()
    relaunches: list[int] = []
    tasks: set[asyncio.Task[None]] = set()
    answers = 0
    done = asyncio.Event()

    async def relaunch() -> None:
        event = await kit.post(
            "support", "Quality gate: add a source, then answer again.", addressed_to=["writer"]
        )
        relaunches.append(event.chain_depth)
        logger.info("The gate relaunches the agent, %d deep", event.chain_depth)

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="quality_gate")
    async def quality_gate(event: Any, ctx: Any) -> None:
        nonlocal answers
        answers += 1
        if answers > GATE_ATTEMPTS:
            done.set()
            return
        task = asyncio.create_task(relaunch())  # the judge's verdict: failed
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        await asyncio.sleep(0.05)  # an audit write, slower than the gate

    try:
        await ask(kit, "Write the incident summary.")
        await asyncio.wait_for(done.wait(), 5)
        await asyncio.gather(*tasks)
        logger.info("Relaunches at depths %s: the agent answered all of them", relaunches)
        assert relaunches == [0] * GATE_ATTEMPTS, relaunches
    finally:
        await kit.close()


async def main() -> None:
    await moderation_reminders()
    await quality_gate_relaunches()


if __name__ == "__main__":
    asyncio.run(main())
