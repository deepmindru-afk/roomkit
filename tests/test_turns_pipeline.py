"""Every intelligence turn of the pipeline runs through the turn backend
(RFC §10.6.1): the solicited turn, a reentry's, and an injected event's.

The channel below records the task each of its ``on_event`` calls ran on: a
turn runs on an executor's task (``roomkit-turn-…``), never on the lane's or
the caller's.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from tests.test_framework import SimpleChannel


class RecordingAgent(Channel):
    """An intelligence channel that streams one reply and records its turns."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str = "agent") -> None:
        super().__init__(channel_id)
        self.turn_tasks: list[str] = []
        self.streams_started = 0

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        task = asyncio.current_task()
        self.turn_tasks.append(task.get_name() if task is not None else "")
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        return ChannelOutput(responded=True, response_stream=self._reply())

    async def _reply(self) -> AsyncIterator[str]:
        self.streams_started += 1
        yield "an answer"

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def _room() -> tuple[RoomKit, RecordingAgent]:
    kit = RoomKit()
    agent = RecordingAgent()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(agent)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "sms")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, agent


async def test_a_solicited_turn_runs_on_an_executor_and_its_stream_reaches_the_room() -> None:
    kit, agent = await _room()
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")),
        room_id="room",
    )
    events = await kit.store.list_events("room")
    assert any(
        isinstance(e.content, TextContent) and e.content.body == "an answer" for e in events
    )
    assert agent.turn_tasks
    assert all(name.startswith("roomkit-turn-") for name in agent.turn_tasks)
    await kit.close()


async def test_an_injected_event_reaches_the_agent_as_a_turn_whose_stream_never_starts() -> None:
    kit, agent = await _room()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC, name="inject")
    async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "sms":
            return HookResult.allow()
        notice = RoomEvent(
            room_id="room",
            source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
            content=TextContent(body="a notice"),
        )
        return HookResult.block(
            reason="filtered", injected=[InjectedEvent(event=notice, target_channel_ids=["agent"])]
        )

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")),
        room_id="room",
    )
    assert agent.turn_tasks
    assert all(name.startswith("roomkit-turn-") for name in agent.turn_tasks)
    assert agent.streams_started == 0
    await kit.close()
