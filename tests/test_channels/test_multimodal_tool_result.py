"""A multimodal tool result survives the unified dispatcher intact.

``AIToolResultPart.result`` has accepted a content-part list (text + images,
e.g. a screenshot) since 0.50 — but the unified dispatcher's user-handler
branch coerced every result through ``str()``, flattening the list to its
Python repr (base64 included) before it could reach the provider. These
tests pin the whole path: dispatcher → tool loop → provider context.
"""

from __future__ import annotations

from roomkit.channels._tool_eviction import is_eviction_placeholder
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIImagePart,
    AIResponse,
    AITextPart,
    AIToolCall,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_SCREENSHOT_TOOL = {
    "name": "take_screenshot",
    "description": "Capture the screen.",
    "parameters": {"type": "object", "properties": {}},
}

_PARTS = [
    AITextPart(text="here"),
    AIImagePart(url="data:image/png;base64,AAAA", mime_type="image/png"),
]


async def _handler(name: str, arguments: dict) -> list[AITextPart | AIImagePart]:
    return list(_PARTS)


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_SCREENSHOT_TOOL]},
    )


def _responses() -> list[AIResponse]:
    return [
        AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="t1", name="take_screenshot", arguments={})],
        ),
        AIResponse(content="done", finish_reason="stop"),
    ]


async def test_a_part_list_result_reaches_the_provider_intact() -> None:
    provider = MockAIProvider(ai_responses=_responses(), vision=True)
    ch = AIChannel("ai1", provider=provider, tool_handler=_handler)

    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    final_context = provider.calls[-1]
    tool_messages = [m for m in final_context.messages if m.role == "tool"]
    assert tool_messages, "the tool round's results must be in the next context"
    result_part = tool_messages[-1].content[0]
    assert isinstance(result_part, AIToolResultPart)
    # The list of parts, not its repr: a str here means the dispatcher
    # flattened the screenshot into prose.
    assert result_part.result == _PARTS


async def test_streaming_loop_carries_the_part_list_too() -> None:
    provider = MockAIProvider(ai_responses=_responses(), streaming=True, vision=True)
    ch = AIChannel("ai1", provider=provider, tool_handler=_handler)

    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )
    assert output.response_stream is not None
    async for _ in output.response_stream:
        pass

    final_context = provider.calls[-1]
    tool_messages = [m for m in final_context.messages if m.role == "tool"]
    assert tool_messages
    result_part = tool_messages[-1].content[0]
    assert isinstance(result_part, AIToolResultPart)
    assert result_part.result == _PARTS


async def test_an_oversized_part_list_is_evicted_with_its_images_kept(streaming: bool) -> None:
    """The text of a part list is bounded like a string result; the images
    still reach the provider, and the full text reads back."""
    big = "node " * 30_000  # an accessibility tree, ~150 KB
    image = AIImagePart(url="data:image/png;base64,AAAA", mime_type="image/png")

    async def handler(name: str, arguments: dict) -> list[AITextPart | AIImagePart]:
        return [AITextPart(text=big), image]

    provider = MockAIProvider(ai_responses=_responses(), vision=True, streaming=streaming)
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)

    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    tool_messages = [m for m in provider.calls[-1].messages if m.role == "tool"]
    result_part = tool_messages[-1].content[0]
    assert isinstance(result_part, AIToolResultPart)
    text, kept = result_part.result
    assert isinstance(text, AITextPart) and is_eviction_placeholder(text.text)
    assert len(text.text) < 9_000
    assert kept == image
    assert big in ch._eviction._store.values()


async def test_a_text_only_model_gets_the_text_of_a_part_list(streaming: bool) -> None:
    """Like a message's images: an image a text-only model cannot take would
    fail the request, so it reads the parts' text and an [image] mark."""
    provider = MockAIProvider(ai_responses=_responses(), vision=False, streaming=streaming)
    ch = AIChannel("ai1", provider=provider, tool_handler=_handler)

    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    tool_messages = [m for m in provider.calls[-1].messages if m.role == "tool"]
    result_part = tool_messages[-1].content[0]
    assert isinstance(result_part, AIToolResultPart)
    assert result_part.result == "here\n[image]"


async def test_the_hook_sees_the_text_a_text_only_model_reads(streaming: bool) -> None:
    """ON_TOOL_CALL sees the shape the model reads: for a text-only model, the
    flattened text, so a redacting hook written for text covers it."""
    provider = MockAIProvider(ai_responses=_responses(), vision=False, streaming=streaming)
    ch = AIChannel("ai1", provider=provider, tool_handler=_handler)
    seen: list[object] = []

    async def observe(event: object) -> None:
        seen.append(event.result)  # type: ignore[attr-defined]
        return None

    ch._tool_call_hook = observe

    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    assert seen == ["here\n[image]"]
