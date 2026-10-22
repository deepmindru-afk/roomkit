"""The pipeline gives every event it commits its id (RFC §8.1)."""

from __future__ import annotations

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent

_CHANNEL_CHOSEN_ID = "chosen-by-the-channel"


class _FixedIdTransport(Channel):
    """A transport that parses every message under the same id."""

    channel_type = ChannelType.SMS

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            id=_CHANNEL_CHOSEN_ID,
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _FixedIdAgent(Channel):
    """An agent that answers every event with a response under one id."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        answer = RoomEvent(
            id=_CHANNEL_CHOSEN_ID,
            room_id=event.room_id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=TextContent(body="answer"),
            chain_depth=event.chain_depth + 1,
        )
        return ChannelOutput(responded=True, response_events=[answer])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def _room(*channels: Channel) -> RoomKit:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    for channel in channels:
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    return kit


def _message(body: str) -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))


async def test_an_inbound_event_is_committed_under_an_id_of_the_pipeline() -> None:
    kit = await _room(_FixedIdTransport("sms"))
    try:
        first = await kit.process_inbound(_message("one"))
        second = await kit.process_inbound(_message("two"))
        assert first.event is not None and second.event is not None
        assert _CHANNEL_CHOSEN_ID not in {first.event.id, second.event.id}
        assert first.event.id != second.event.id
    finally:
        await kit.close()


async def test_a_response_is_committed_under_an_id_of_the_pipeline() -> None:
    kit = await _room(_FixedIdTransport("sms"), _FixedIdAgent("ai"))
    try:
        await kit.process_inbound(_message("one"))
        events = await kit.store.list_events("r1")
        answers = [e for e in events if e.source.channel_id == "ai"]
        assert [e.id == _CHANNEL_CHOSEN_ID for e in answers] == [False]
    finally:
        await kit.close()
