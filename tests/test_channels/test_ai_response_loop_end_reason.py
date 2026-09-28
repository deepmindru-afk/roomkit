"""``ON_AI_RESPONSE`` says whether the turn finished or was cut off.

Both loops already knew — the streaming one names it on ``LoopEndMarker``, the
buffered one on ``ToolLoopResult.reason`` — but neither reason reached
``AIResponseEvent``. A hook saw how much work the turn did and never whether
the work was allowed to finish, so consumers re-derived it from
``tool_calls_count``. That reading was only ever accidental: the counter
reports the calls the turn *ran*, so a healthy multi-round answer and one
guillotined by the round cap both come back positive.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_ECHO_TOOL = {
    "name": "echo",
    "description": "Echo a value.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_ECHO_TOOL]},
    )


def _tool(i: int = 0) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"t{i}", name="echo", arguments={"value": str(i)})],
    )


def _channel(responses: list[AIResponse], *, streaming: bool, **kwargs: object) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=responses, streaming=streaming),
        tool_handler=AsyncMock(return_value="ok"),
        **kwargs,  # type: ignore[arg-type]
    )


def _observed(ch: AIChannel) -> list[AIResponseEvent]:
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    ch._after_response_hook = observe
    return seen


async def _turn(ch: AIChannel) -> AIResponseEvent:
    seen = _observed(ch)
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )
    assert len(seen) == 1
    return seen[0]


async def test_plain_answer_is_completed(streaming: bool) -> None:
    event = await _turn(_channel([AIResponse(content="hello")], streaming=streaming))

    assert event.loop_end_reason == "completed"
    assert event.tool_calls_count == 0


async def test_answer_after_tools_is_completed_despite_a_positive_count(streaming: bool) -> None:
    """The case that made counting tool calls the wrong signal: a healthy turn
    that used tools reports work done, not a cut-off."""
    ch = _channel([_tool(0), _tool(1), AIResponse(content="done")], streaming=streaming)

    event = await _turn(ch)

    assert event.loop_end_reason == "completed"
    assert event.tool_calls_count == 2
    assert event.round_count == 2


async def test_round_cap_is_named(streaming: bool) -> None:
    ch = _channel([_tool(0), _tool(1), _tool(2)], streaming=streaming, max_tool_rounds=1)

    event = await _turn(ch)

    assert event.loop_end_reason == "max_rounds"


async def test_ripcord_is_named(streaming: bool) -> None:
    ch = _channel(
        [*[_tool(0) for _ in range(6)], AIResponse(content="what I found")],
        streaming=streaming,
    )

    event = await _turn(ch)

    assert event.loop_end_reason == "force_stopped"
