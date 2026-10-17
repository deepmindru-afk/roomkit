"""The room owner controls a turn across the turn backend (RFC §10.6.8):
its cancellation says why, and steering reaches the turn's loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.models.steering import Cancel, InjectMessage, SteeringDirective
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor
from roomkit.turns._executor_turn import ExecutorTurn
from tests.test_framework import SimpleChannel

WAIT = AITool(name="wait_here", description="Wait.", parameters={"type": "object"})


class Paused:
    """A host tool that waits until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.started.set()
        await self.release.wait()
        return "waited"


async def _kit(tool: Paused, *answers: str) -> tuple[RoomKit, Agent]:
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="wait_here", arguments={})])
    provider = MockAIProvider(ai_responses=[call, *(AIResponse(content=a) for a in answers)])
    agent = Agent("agent", provider=provider, role="agent", tools=[WAIT], tool_handler=tool)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, agent


def _message(body: str = "go") -> InboundMessage:
    return InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body))


# -- Cancellation ---------------------------------------------------------------


async def test_a_cancelled_delivery_tells_the_executor_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasons: list[str] = []
    cut = ExecutorTurn._cut_turn

    def spy(self: ExecutorTurn, reason: str) -> None:
        reasons.append(reason)
        cut(self, reason)

    monkeypatch.setattr(ExecutorTurn, "_cut_turn", spy)
    tool = Paused()
    kit, _ = await _kit(tool, "never")
    first = await kit.process_inbound(_message(), room_id="r1", defer_delivery=True)
    await asyncio.wait_for(tool.started.wait(), 2)

    assert first.delivery is not None
    await first.delivery.cancel(reason="superseded")

    assert "superseded" in reasons
    await kit.close()


# -- Steering through the kit ---------------------------------------------------


async def test_a_directive_for_a_room_reaches_its_turns_loop() -> None:
    tool = Paused()
    kit, agent = await _kit(tool, "done")
    turn = asyncio.create_task(kit.process_inbound(_message(), room_id="r1"))
    await asyncio.wait_for(tool.started.wait(), 2)

    reached = await kit.steer(InjectMessage(content="Mind the budget."), "agent", room_id="r1")
    tool.release.set()
    await asyncio.wait_for(turn, 5)

    assert reached == 1
    # The next round reads the injected message.
    second = agent.provider.calls[1]
    assert any("Mind the budget." in str(m.content) for m in second.messages)
    await kit.close()


async def test_a_cancel_for_a_room_ends_its_turn_before_another_round() -> None:
    tool = Paused()
    kit, agent = await _kit(tool, "never asked")
    turn = asyncio.create_task(kit.process_inbound(_message(), room_id="r1"))
    await asyncio.wait_for(tool.started.wait(), 2)

    reached = await kit.steer(Cancel(reason="stop"), "agent", room_id="r1")
    tool.release.set()
    await asyncio.wait_for(turn, 5)

    assert reached == 1
    assert len(agent.provider.calls) == 1
    await kit.close()


async def test_a_directive_for_a_room_with_no_turn_reaches_nothing() -> None:
    kit, _ = await _kit(Paused())

    assert await kit.steer(Cancel(), "agent", room_id="r1") == 0
    with pytest.raises(ValueError):
        await kit.steer(Cancel(), "agent", room_id="r1", turn_id="t1")
    await kit.close()


# -- Steering one turn, on a channel with or without steering ------------------


class Held(Channel):
    """An intelligence channel whose stream waits until released; with
    ``steers``, a steerable one recording what reached it."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self) -> None:
        super().__init__("agent")
        self.release = asyncio.Event()
        self.steered: list[tuple[SteeringDirective, str | None]] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(responded=True, response_stream=self._stream())

    async def _stream(self) -> AsyncIterator[str]:
        yield "first"
        await self.release.wait()
        yield "second"

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class Steerable(Held):
    def steer(self, directive: SteeringDirective, *, turn_id: str | None = None) -> int:
        self.steered.append((directive, turn_id))
        return 1


def _client(channel: Channel) -> TurnClient:
    channels = {channel.channel_id: channel}
    return TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor(channels.get, hosts=channels.__contains__),
    )


async def _start(client: TurnClient) -> ChannelOutput:
    event = RoomEvent(
        room_id="room",
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
        content=TextContent(body="hello"),
    )
    binding = ChannelBinding(
        channel_id="agent",
        room_id="room",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    return await client.run(event, binding, RoomContext(room=Room(id="room")))


async def test_a_directive_for_a_turn_reaches_that_turns_loop_even_unread() -> None:
    channel = Steerable()
    client = _client(channel)
    output = await _start(client)
    turn_id = client._started[0].request.turn_id

    # Nobody reads the stream: the owner still hears the turn's answer.
    reached = await asyncio.wait_for(
        client.steer(InjectMessage(content="x"), "agent", turn_id=turn_id), 2
    )

    assert reached == 1
    assert channel.steered == [(InjectMessage(content="x"), turn_id)]
    channel.release.set()
    assert output.response_stream is not None
    assert [item async for item in output.response_stream] == ["first", "second"]
    await client.close()


async def test_a_cancel_for_a_channel_without_steering_cuts_its_turn() -> None:
    channel = Held()
    client = _client(channel)
    output = await _start(client)
    assert output.response_stream is not None
    stream = output.response_stream
    assert await anext(stream) == "first"

    reached = await asyncio.wait_for(client.steer(Cancel(), "agent", room_id="room"), 2)
    # The turn ends as a closed stream would: no error for its reader.
    rest = [item async for item in stream]

    assert reached == 1
    assert rest == []
    await client.close()


async def test_a_directive_other_than_cancel_reaches_no_channel_without_steering() -> None:
    channel = Held()
    client = _client(channel)
    output = await _start(client)

    assert await client.steer(InjectMessage(content="x"), "agent", room_id="room") == 0
    channel.release.set()
    assert output.response_stream is not None
    assert [item async for item in output.response_stream] == ["first", "second"]
    await client.close()
