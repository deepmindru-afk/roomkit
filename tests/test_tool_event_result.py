"""A TOOL_CALL_END event keeps a bounded share of a result's images.

The event is persisted, broadcast and handed to the event pipeline's hooks,
so a screenshot's base64 must not ride along without bound; the model's copy
of the result keeps every image (RMK-260).
"""

from __future__ import annotations

from typing import Any

from roomkit.channels._tool_event_result import TOOL_EVENT_IMAGE_MAX_CHARS, tool_event_result
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIImagePart,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCall,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_framework import SimpleChannel


def _image(kb: int) -> AIImagePart:
    header = "data:image/png;base64,"
    return AIImagePart(url=header + "A" * (kb * 1024 - len(header)), mime_type="image/png")


class TestToolEventResult:
    def test_images_are_kept_while_they_fit_and_noted_past_the_bound(self) -> None:
        first, second, third = _image(300), _image(300), _image(100)
        result = [AITextPart(text="page"), first, second, third]

        kept = tool_event_result(result)

        assert kept[:2] == [AITextPart(text="page"), first]
        assert kept[2] == AITextPart(text="[image image/png, 225 KB, not kept in the event]")
        # A later image that still fits is kept: the bound is on the total.
        assert kept[3] is third
        assert sum(len(p.url) for p in kept if isinstance(p, AIImagePart)) <= (
            TOOL_EVENT_IMAGE_MAX_CHARS
        )

    def test_a_result_under_the_bound_is_unchanged(self) -> None:
        result = [AITextPart(text="page"), _image(200)]
        assert tool_event_result(result) == result

    def test_a_string_result_is_unchanged(self) -> None:
        assert tool_event_result("x" * 2_000_000) == "x" * 2_000_000


def _three_screenshots() -> list[AITextPart | AIImagePart]:
    return [AITextPart(text="page"), _image(300), _image(300), _image(300)]


async def _screenshot_handler(name: str, args: dict[str, Any]) -> list[Any]:
    return _three_screenshots()


def _responses() -> list[AIResponse]:
    return [
        AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="tc1", name="shoot", arguments={})],
        ),
        AIResponse(content="done", finish_reason="stop"),
    ]


def _images(result: Any) -> int:
    return sum(isinstance(p, AIImagePart) for p in result)


async def test_a_streamed_turn_persists_the_bounded_event_and_the_model_keeps_all() -> None:
    provider = MockAIProvider(streaming=True, vision=True, ai_responses=_responses())
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tool_handler=_screenshot_handler,
            tools=[AITool(name="shoot", description="Screenshot")],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    ends = [e for e in await kit.store.list_events("r1") if e.type == EventType.TOOL_CALL_END]
    assert len(ends) == 1
    assert isinstance(ends[0].content, ToolCallContent)
    assert _images(ends[0].content.result) == 1

    tool_message = next(m for m in provider.calls[-1].messages if m.role == "tool")
    part = tool_message.content[0]
    assert isinstance(part, AIToolResultPart)
    assert _images(part.result) == 3


async def test_a_non_streamed_turn_returns_the_bounded_event() -> None:
    provider = MockAIProvider(streaming=False, vision=True, ai_responses=_responses())
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=_screenshot_handler,
        tools=[AITool(name="shoot", description="Screenshot")],
    )

    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        RoomContext(room=Room(id="r1")),
    )

    ends = [e for e in output.response_events if e.type == EventType.TOOL_CALL_END]
    assert isinstance(ends[0].content, ToolCallContent)
    assert _images(ends[0].content.result) == 1
