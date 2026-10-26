"""An agent that takes a room's turns one at a time (``serialize_room_turns``,
RFC §6.4): its turns in a room run in the order they were asked, each once the
one before it is over, and each reads the room as it stood when it was asked
plus the agent's answers since, never a message that came after it.

The agent runs in the kit's process, or on an executor the kit knows only by
its declaration (RFC §10.6.2). Cancelling a waiting or a running turn is
covered with the other kinds of agent in ``test_delivery_cancel_turns.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from roomkit import Agent, ChannelCategory, ChannelDeclaration, Loop, RoomKit
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from tests.test_framework import SimpleChannel

WAIT = 2.0


def _text(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "text", "") for part in content)


class _Gated(MockAIProvider):
    """Answers each message once its gate opens; keeps what each turn read."""

    def __init__(self) -> None:
        super().__init__()
        self.read: dict[str, list[str]] = {}
        self.gates: dict[str, asyncio.Event] = {}

    async def generate(self, context: AIContext) -> AIResponse:
        conversation = [_text(message.content) for message in context.messages]
        asked = conversation[-1]
        self.read[asked] = conversation
        await self.gates.setdefault(asked, asyncio.Event()).wait()
        return AIResponse(content=f"answer to {asked}")

    def answer(self, asked: str) -> None:
        self.gates.setdefault(asked, asyncio.Event()).set()

    async def until_started(self, asked: str) -> None:
        async with asyncio.timeout(WAIT):
            while asked not in self.read:
                await asyncio.sleep(0.005)


@dataclass
class Room:
    kit: RoomKit
    agent: _Gated

    async def send(self, body: str) -> InboundResult:
        message = InboundMessage(
            channel_id="sms", sender_id="user", content=TextContent(body=body)
        )
        return await self.kit.process_inbound(message, room_id="r", defer_delivery=True)

    async def until_waiting(self, turns: int) -> None:
        """Until *turns* of the agent hold or wait a place in the room."""
        lines = self.kit._get_router()._room_turns._lines  # noqa: SLF001
        async with asyncio.timeout(WAIT):
            while (line := lines.get(("r", "agent"))) is None or line.users < turns:
                await asyncio.sleep(0.005)


async def _kit(where: str, agent: Agent) -> tuple[RoomKit, TurnExecutor | None]:
    """A kit with the agent in its own process, or on an executor that builds
    it while the kit holds its declaration alone."""
    if where == "live":
        kit = RoomKit()
        kit.register_channel(agent)
        return kit, None
    store, backend = InMemoryStore(), InMemoryTurnBackend()
    executor = TurnExecutor(lambda channel_id: agent, hosts={"agent"}.__contains__, store=store)
    await backend.serve(executor)
    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="agent", serializes_room_turns=True))
    return kit, executor


@pytest.fixture(params=["live", "declared"])
async def room(request: pytest.FixtureRequest) -> AsyncIterator[Room]:
    provider = _Gated()
    agent = Agent("agent", provider=provider, serialize_room_turns=True)
    kit, executor = await _kit(request.param, agent)
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    yield Room(kit, provider)
    for gate in provider.gates.values():
        gate.set()
    await kit.close()
    if executor is not None:
        await executor.close()


async def test_a_room_s_turns_run_one_after_the_other_each_reading_the_answers_before_it(
    room: Room,
) -> None:
    a = await room.send("task A")
    await room.agent.until_started("task A")
    b = await room.send("task B")
    c = await room.send("task C")
    await room.until_waiting(3)
    assert list(room.agent.read) == ["task A"]

    room.agent.answer("task A")
    await room.agent.until_started("task B")
    # B reads A answered, and not C, which came after it.
    assert room.agent.read["task B"] == ["task A", "answer to task A", "task B"]
    assert "task C" not in room.agent.read

    room.agent.answer("task B")
    await room.agent.until_started("task C")
    # In the room's order: the answers were written after C arrived.
    assert room.agent.read["task C"] == [
        "task A",
        "task B",
        "answer to task A",
        "answer to task B",
        "task C",
    ]
    room.agent.answer("task C")
    for result in (a, b, c):
        assert result.delivery is not None
        assert (await asyncio.wait_for(result.delivery.wait(), WAIT)).error is None


async def test_a_waiting_turn_whose_agent_leaves_the_room_never_starts(room: Room) -> None:
    a = await room.send("task A")
    await room.agent.until_started("task A")
    b = await room.send("task B")
    await room.until_waiting(2)

    await room.kit.detach_channel("r", "agent")
    room.agent.answer("task A")
    assert a.delivery is not None and b.delivery is not None
    await asyncio.wait_for(a.delivery.wait(), WAIT)
    await asyncio.wait_for(b.delivery.wait(), WAIT)

    assert list(room.agent.read) == ["task A"]


async def _bare_room(*agents: Agent, **room: object) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    for agent in agents:
        kit.register_channel(agent)
    await kit.create_room(room_id="r", **room)  # type: ignore[arg-type]
    await kit.attach_channel("r", "sms")
    return kit


async def _said(kit: RoomKit, channel_id: str) -> list[str]:
    return [
        event.content.body
        for event in await kit.store.list_events("r")
        if event.source.channel_id == channel_id and isinstance(event.content, TextContent)
    ]


async def test_an_agent_asked_again_in_the_same_delivery_is_not_held_by_its_first_turn() -> None:
    # One message asks both agents; the planner's answer asks the writer
    # again, a turn read in the same delivery once its first turn is over.
    writer = Agent(
        "writer",
        provider=MockAIProvider(responses=["drafted", "redrafted"]),
        serialize_room_turns=True,
    )
    planner = Agent("planner", provider=MockAIProvider(responses=["plan"]))
    kit = await _bare_room(writer, planner)
    await kit.attach_channel("r", "writer", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r", "planner", category=ChannelCategory.INTELLIGENCE)
    try:
        message = InboundMessage(
            channel_id="sms", sender_id="user", content=TextContent(body="go")
        )
        result = await asyncio.wait_for(kit.process_inbound(message, room_id="r"), WAIT)

        assert result.error is None
        assert (await _said(kit, "writer"))[:2] == ["drafted", "redrafted"]
    finally:
        await kit.close()


async def test_a_strategy_that_takes_the_agent_s_turns_orders_them_itself() -> None:
    producer = Agent(
        "prod", provider=MockAIProvider(responses=["draft"]), serialize_room_turns=True
    )
    reviewer = Agent("rev", provider=MockAIProvider(responses=["APPROVED"]))
    loop = Loop(agent=producer, reviewer=reviewer, max_iterations=1)
    kit = await _bare_room(producer, orchestration=loop)
    try:
        message = InboundMessage(
            channel_id="sms", sender_id="user", content=TextContent(body="go")
        )
        await asyncio.wait_for(kit.process_inbound(message, room_id="r"), WAIT)

        # The loop's answer, which no stream carries.
        assert await _said(kit, "prod") == ["draft"]
    finally:
        await kit.close()


async def test_an_agent_s_turns_run_side_by_side_by_default() -> None:
    provider = _Gated()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("agent", provider=provider))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    room = Room(kit, provider)
    try:
        await room.send("task A")
        await provider.until_started("task A")
        await room.send("task B")

        # B starts while A runs, A's order before it unanswered.
        await provider.until_started("task B")
        assert provider.read["task B"] == ["task A", "task B"]
    finally:
        for gate in provider.gates.values():
            gate.set()
        await kit.close()
