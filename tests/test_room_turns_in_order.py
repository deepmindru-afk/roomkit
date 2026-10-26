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

from roomkit import (
    Agent,
    ChannelCategory,
    ChannelDeclaration,
    HookResult,
    HookTrigger,
    Loop,
    RoomKit,
)
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend, TurnExecutor, current_turn_host
from tests.test_framework import AILikeChannel, SimpleChannel

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
        """Until *turns* of the agent have their place in the room."""
        lines = self.kit._get_router()._room_turns._lines  # noqa: SLF001
        async with asyncio.timeout(WAIT):
            while len(lines.get(("r", "agent"), ())) < turns:
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


async def test_a_message_removed_while_a_turn_waits_is_not_read(room: Room) -> None:
    await room.send("my pin is 4242")
    await room.agent.until_started("my pin is 4242")
    await room.send("task B")
    await room.until_waiting(2)
    pin = next(
        event
        for event in await room.kit.store.list_events("r")
        if getattr(event.content, "body", None) == "my pin is 4242"
    )

    await room.kit.delete_event("r", pin.id)
    room.agent.answer("my pin is 4242")
    await room.agent.until_started("task B")

    assert room.agent.read["task B"] == ["answer to my pin is 4242", "task B"]


class _AnswersA(AILikeChannel):
    """An agent that answers task A at once, its answer delivered with A."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if getattr(event.content, "body", None) != "task A":
            return ChannelOutput.empty()
        return await super().on_event(event, binding, context)


async def test_turns_start_in_the_order_they_were_asked_whichever_delivery_reads_first() -> None:
    # A's delivery reads another agent's answer to A first, B's has nothing
    # else to read: B's delivery reaches its turn first, which still waits.
    provider = _Gated()
    agent = Agent("agent", provider=provider, serialize_room_turns=True)
    kit = await _bare_room(agent, _AnswersA("noter", response="noted A"))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r", "noter", category=ChannelCategory.INTELLIGENCE)
    room = Room(kit, provider)
    try:
        await room.send("task A")
        await room.send("task B")
        await provider.until_started("task A")
        await room.until_waiting(2)
        assert "task B" not in provider.read

        provider.answer("task A")
        await provider.until_started("task B")
        assert provider.read["task B"][-3:] == ["task A", "answer to task A", "task B"]
    finally:
        for gate in provider.gates.values():
            gate.set()
        await kit.close()


class _StopsAtFirstWord(SimpleChannel):
    """A voice transport its listener cuts at the agent's first word."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for chunk in text_stream:
            if isinstance(chunk, str):
                break
        return ChannelOutput.empty()


async def test_a_waiting_turn_reads_what_a_cut_turn_said() -> None:
    provider = _Gated()
    agent = Agent("agent", provider=provider, serialize_room_turns=True)
    kit = RoomKit()
    kit.register_channel(_StopsAtFirstWord("sms"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    room = Room(kit, provider)
    try:
        await room.send("task A")
        await provider.until_started("task A")
        await room.send("task B")
        await room.until_waiting(2)

        provider.answer("task A")
        await provider.until_started("task B")
        assert provider.read["task B"] == ["task A", "answer to task A", "task B"]
    finally:
        for gate in provider.gates.values():
            gate.set()
        await kit.close()


async def test_a_muted_agent_still_reads_each_message_at_once() -> None:
    # A muted channel reacts to every event and its answer is not delivered
    # (RFC §7.2): it takes no place, and reads the message when it comes.
    read: list[str] = []
    agent = Agent("agent", provider=MockAIProvider(responses=["hi"]), serialize_room_turns=True)
    kit = await _bare_room(agent)
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    await kit.mute("r", "agent")

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def reads(event: Any, ctx: Any) -> HookResult:
        read.append(_text(event.ai_context.messages[-1].content))
        return HookResult.allow()

    try:
        message = InboundMessage(
            channel_id="sms", sender_id="user", content=TextContent(body="hi")
        )
        await asyncio.wait_for(kit.process_inbound(message, room_id="r"), WAIT)

        assert read == ["hi"]
        assert await _said(kit, "agent") == []
    finally:
        await kit.close()


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


_ASKS_ITS_OWN_ROOM = [
    AIResponse(
        content="Let me ask.",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="tc1", name="ask_room", arguments={})],
    ),
    AIResponse(content="outer answer", finish_reason="stop"),
    AIResponse(content="nested answer", finish_reason="stop"),
]


_THE_KIT_OBJECT = pytest.mark.in_process("the tool calls the kit object, not its turn's host")


@pytest.mark.parametrize(
    "door",
    [
        pytest.param("process_inbound", marks=_THE_KIT_OBJECT),
        pytest.param("deferred", marks=_THE_KIT_OBJECT),
        "turn host",
    ],
)
async def test_a_message_a_turn_sends_into_its_own_room_is_answered_after_it(door: str) -> None:
    # The tool's call returns without the answer: waiting for it would wait
    # on the turn that made the call. A tool on another executor reaches the
    # room through its turn's host.
    kit = await _bare_room()

    async def ask_room(name: str, arguments: dict[str, Any]) -> str:
        question = TextContent(body="nested question")
        if door == "turn host":
            host = current_turn_host()
            assert host is not None
            await host.send_event("r", "sms", question)
            return "sent"
        message = InboundMessage(channel_id="sms", sender_id="user", content=question)
        result = await kit.process_inbound(message, room_id="r", defer_delivery=door == "deferred")
        if result.delivery is not None:
            await result.delivery.wait()
        return "sent"

    agent = Agent(
        "agent",
        provider=MockAIProvider(ai_responses=list(_ASKS_ITS_OWN_ROOM)),
        tool_handler=ask_room,
        serialize_room_turns=True,
    )
    kit.register_channel(agent)
    tools = {"tools": [{"name": "ask_room", "description": "Ask the room"}]}
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE, metadata=tools)
    try:
        message = InboundMessage(
            channel_id="sms", sender_id="user", content=TextContent(body="go")
        )
        await asyncio.wait_for(kit.process_inbound(message, room_id="r"), WAIT)
        async with asyncio.timeout(WAIT):
            while len(await _said(kit, "agent")) < 3:
                await asyncio.sleep(0.005)

        assert await _said(kit, "agent") == ["Let me ask.", "outer answer", "nested answer"]
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
