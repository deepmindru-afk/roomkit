"""A sixth composite level is refused where it is built (RFC §5.3).

A model's constructor refuses one, but ``model_copy(update=...)`` skips the
validators, so content can be assembled past the limit and handed to the
kit: each boundary refuses it before anything of its event is written, and no
event that holds one is ever committed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from roomkit import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, HookTrigger
from roomkit.models.event import (
    CompositeContent,
    EditContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.store_filter import EventFilter
from tests.test_framework import SimpleChannel


def _six_levels() -> CompositeContent:
    """A composite six levels deep, assembled past its validator."""
    content = CompositeContent(parts=[TextContent(body="leaf")])
    for _ in range(5):
        content = content.model_copy(update={"parts": [content]})
    return content


def _holding[M: (RoomEvent, InboundMessage)](model: M) -> M:
    """*model* with six levels as its content, past its validator."""
    return model.model_copy(update={"content": _six_levels()})


def _event(room_id: str, channel_id: str = "sms1") -> RoomEvent:
    source = EventSource(channel_id=channel_id, channel_type=ChannelType.SMS)
    return RoomEvent(room_id=room_id, source=source, content=TextContent(body="hi"))


def _levels(content: object) -> int:
    if isinstance(content, EditContent):
        return _levels(content.new_content)
    if not isinstance(content, CompositeContent):
        return 0
    return 1 + max(_levels(part) for part in content.parts)


async def _messages_levels(kit: RoomKit) -> list[int]:
    """The composite levels of each message the room stored, BLOCKED included."""
    events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
    return [_levels(e.content) for e in events if e.type == EventType.MESSAGE]


class _Parser(SimpleChannel):
    """A transport whose parse builds six levels from a text message."""

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return _holding(await super().handle_inbound(message, context))


class _Answering(SimpleChannel):
    """A transport that answers every other channel's event with six levels."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        answer = _holding(_event(event.room_id, self.channel_id))
        return ChannelOutput(responded=True, response_events=[answer])


async def _room(sender: SimpleChannel | None = None) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(sender or SimpleChannel("sms1"))
    kit.register_channel(SimpleChannel("ws1"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ws1")
    return kit


def _message() -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi"))


async def test_an_inbound_message_holding_six_levels_is_refused_before_it_is_routed() -> None:
    """Routed, its message would open a room: no room binds its channel."""
    kit = await _room()
    kit.register_channel(SimpleChannel("sms2"))
    message = _message().model_copy(update={"channel_id": "sms2"})

    with pytest.raises(ValueError, match="nesting depth"):
        await kit.process_inbound(_holding(message))

    assert [room.id for room in await kit.store.list_rooms()] == ["r1"]
    await kit.close()


async def test_a_parsed_event_holding_six_levels_fails_its_message_with_nothing_stored() -> None:
    kit = await _room(_Parser("sms1"))

    with pytest.raises(ValueError, match="nesting depth"):
        await kit.process_inbound(_message(), room_id="r1")

    assert await _messages_levels(kit) == []
    await kit.close()


def _edit_of_six_levels() -> EditContent:
    edit = EditContent(target_event_id="e1", new_content=TextContent(body="x"))
    return edit.model_copy(update={"new_content": _six_levels()})


_DIRECT_CALLS: dict[str, Callable[[RoomKit], Awaitable[object]]] = {
    "send_event": lambda kit: kit.send_event("gone", "sms1", _six_levels()),
    "an edit's new content": lambda kit: kit.send_event(
        "gone", "sms1", _edit_of_six_levels(), event_type=EventType.EDIT
    ),
    "commit_event": lambda kit: kit.commit_event("gone", _holding(_event("gone"))),
    "update_event": lambda kit: kit.update_event("gone", "e1", content=_six_levels()),
}


@pytest.mark.parametrize("call", list(_DIRECT_CALLS.values()), ids=list(_DIRECT_CALLS))
async def test_a_direct_call_holding_six_levels_raises_before_anything_is_read(
    call: Callable[[RoomKit], Awaitable[object]],
) -> None:
    """The room it names does not exist: a refusal past the read would say so."""
    kit = await _room()

    with pytest.raises(ValueError, match="nesting depth"):
        await call(kit)

    await kit.close()


@pytest.mark.parametrize("held", ["modify", "injection"])
async def test_a_hook_result_holding_six_levels_is_unusable(held: str) -> None:
    kit = await _room()
    errors: list[FrameworkEvent] = []

    @kit.on("hook_error")
    async def failed(event: FrameworkEvent) -> None:
        errors.append(event)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="rewriter")
    async def rewriter(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "sms1":
            return HookResult.allow()
        deep = _holding(event)
        if held == "modify":
            return HookResult.modify(deep)
        return HookResult.allow([InjectedEvent(event=deep, target_channel_ids=["ws1"])])

    result = await kit.process_inbound(_message(), room_id="r1")

    assert result.event is not None and result.event.content == TextContent(body="hi")
    assert await _messages_levels(kit) == [0]
    assert [error.data["hook"] for error in errors] == ["rewriter"]
    await kit.close()


async def test_a_response_holding_six_levels_is_its_channel_s_failure() -> None:
    kit = await _room()
    kit.register_channel(_Answering("echo"))
    await kit.attach_channel("r1", "echo")

    result = await kit.process_inbound(_message(), room_id="r1")

    assert result.event is not None
    assert await _messages_levels(kit) == [0]
    assert result.delivery_results["echo"].status == "failed"
    await kit.close()
