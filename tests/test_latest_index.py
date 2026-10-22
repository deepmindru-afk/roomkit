"""``Room.latest_index`` is null while the room has no event (RFC §5.1, §8.1)."""

from __future__ import annotations

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.models.store_filter import PersistencePolicy


class _Recorder(Channel):
    """A transport that parses messages of the type their metadata names and
    records what it is delivered."""

    channel_type = ChannelType.SMS

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[str] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=EventType(message.metadata.get("type", "message")),
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        assert isinstance(event.content, TextContent)
        self.delivered.append(event.content.body)
        return ChannelOutput.empty()


def _message(body: str, event_type: str = "message") -> InboundMessage:
    return InboundMessage(
        channel_id="a",
        sender_id="u",
        content=TextContent(body=body),
        metadata={"type": event_type},
    )


async def _room_with_unstored_system_events() -> tuple[RoomKit, _Recorder]:
    kit = RoomKit(
        persistence_policy=PersistencePolicy(persist_types={EventType.MESSAGE}),
        delivery_gap_timeout=0.5,
    )
    sender, reader = _Recorder("a"), _Recorder("b")
    kit.register_channel(sender)
    kit.register_channel(reader)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "a")
    await kit.attach_channel("r1", "b")
    return kit, reader


class TestAnEmptyRoom:
    def test_has_no_latest_index(self) -> None:
        assert Room(id="r1").latest_index is None

    async def test_delivers_its_first_stored_message_after_an_unstored_event(self) -> None:
        kit, reader = await _room_with_unstored_system_events()
        try:
            await kit.process_inbound(_message("sys", "system"))
            await kit.process_inbound(_message("hello"))
            assert reader.delivered == ["sys", "hello"]
        finally:
            await kit.close()

    async def test_takes_its_first_event_s_index_as_latest(self) -> None:
        kit, _ = await _room_with_unstored_system_events()
        try:
            await kit.process_inbound(_message("hello"))
            room = await kit.store.get_room("r1")
            assert room is not None
            assert (room.event_count, room.latest_index) == (1, 0)
        finally:
            await kit.close()


class TestARoomStoredBeforeLatestIndexWasNullable:
    def test_reads_an_empty_room_s_zero_as_no_index(self) -> None:
        stored = Room.model_validate({"id": "r1", "event_count": 0, "latest_index": 0})
        assert stored.latest_index is None
