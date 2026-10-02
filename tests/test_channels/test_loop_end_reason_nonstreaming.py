"""A provider read through its ``generate()`` gets the loop's exit named too,
and every round counted (RFC §6.4).

A force-stopped or round-capped turn must not read as a completed one, and a
multi-round turn's usage is every round's, not the last generation's. The
loop names both on its ``LoopEndMarker``, which the room writes on the turn's
last message as ``loop_end_reason`` and ``ai_usage``.
"""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.streaming import LoopEndMarker
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

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


def _tool(i: int = 0, usage: dict[str, int] | None = None) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        usage=usage or {},
        tool_calls=[AIToolCall(id=f"t{i}", name="echo", arguments={"value": str(i)})],
    )


async def _handler(name: str, arguments: dict) -> str:
    return "ok"


async def _turn_end(ch: AIChannel) -> tuple[LoopEndMarker, str]:
    """The turn's end marker and the text of its final round."""
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )
    assert output.response_stream is not None
    end: LoopEndMarker | None = None
    text: list[str] = []
    async for delta in output.response_stream:
        if isinstance(delta, LoopEndMarker):
            end = delta
        elif isinstance(delta, str):
            text.append(delta)
    assert end is not None, "the loop names how every turn ended"
    return end, "".join(text)


def _channel(responses: list[AIResponse], **kwargs: object) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=responses),
        tool_handler=_handler,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_a_plain_answer_is_marked_completed() -> None:
    end, text = await _turn_end(_channel([AIResponse(content="hello")]))

    assert end.reason == "completed"


async def test_an_answer_after_tools_is_still_completed() -> None:
    end, text = await _turn_end(_channel([_tool(), AIResponse(content="done")]))

    assert end.reason == "completed"


async def test_the_anti_loop_ripcord_is_named_not_disguised_as_an_answer() -> None:
    """Mirror of the streaming test: six identical calls trip the guard, the
    ripcord demands prose — text that is a summary of a cut turn, not an
    answer, and now says so."""
    ch = _channel([*[_tool(0) for _ in range(6)], AIResponse(content="here is what I found")])

    end, text = await _turn_end(ch)

    assert end.reason == "force_stopped"
    assert "here is what I found" in text


async def test_the_round_cap_is_named() -> None:
    """Exhausting the budget with calls still pending used to end the loop
    with no log and no name — the pending calls just vanished."""
    ch = _channel([_tool(0), _tool(1), _tool(2)], max_tool_rounds=1)

    end, text = await _turn_end(ch)

    assert end.reason == "max_rounds"


async def test_an_answer_landing_on_the_last_round_is_a_plain_completion() -> None:
    """Budget exhaustion is only a cut when the model still wanted tools."""
    ch = _channel([_tool(0), AIResponse(content="done", finish_reason="stop")], max_tool_rounds=1)

    end, text = await _turn_end(ch)

    assert end.reason == "completed"


async def test_usage_sums_every_round_not_just_the_last() -> None:
    ch = _channel(
        [
            _tool(0, usage={"input_tokens": 10, "output_tokens": 5}),
            _tool(1, usage={"input_tokens": 20, "output_tokens": 6}),
            AIResponse(
                content="done",
                finish_reason="stop",
                usage={"input_tokens": 40, "output_tokens": 7},
            ),
        ]
    )

    end, text = await _turn_end(ch)

    assert end.usage == {"input_tokens": 70, "output_tokens": 18}
