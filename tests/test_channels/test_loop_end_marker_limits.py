"""The end marker states the limits the turn ran under, on every exit (RFC §6.4).

A consumer names the limit a ``max_rounds``, ``timeout`` or ``budget_exceeded``
end hit without reading the channel: the round cap and the deadline are the
channel's, the budget the turn's own, resolved per turn (here by the binding,
on a channel that sets none). ``rounds`` is how many tool rounds ran, the
``round_count`` the turn's ``ON_AI_RESPONSE`` reports. Each exit runs on a
provider that streams and on one that only has ``generate()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.streaming import LoopEndMarker, LoopEndReason
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_interrupted_turn import _FailingAt

_LOOKUP = AITool(name="lookup", description="Look it up", parameters={})


def _call(value: str = "0", **usage: int) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"c{value}", name="lookup", arguments={"q": value})],
        usage=usage,
    )


def _said(text: str, finish_reason: str = "stop") -> AIResponse:
    return AIResponse(content=text, finish_reason=finish_reason)


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return "ok"


@dataclass
class _Exit:
    """A turn scripted to end on *reason* after *rounds* tool rounds."""

    reason: LoopEndReason
    rounds: int
    responses: list[AIResponse]
    channel: dict[str, Any] = field(default_factory=dict)
    # The generation the provider fails on, for an ``error`` end.
    fail_at: int | None = None


_EXITS = [
    _Exit("completed", 1, [_call(), _said("done")]),
    _Exit("max_rounds", 2, [_call("0"), _call("1"), _call("2")], {"max_tool_rounds": 2}),
    _Exit("timeout", 0, [_call()], {"tool_loop_timeout_seconds": 1e-9}),
    _Exit("budget_exceeded", 0, [_call(input_tokens=400, output_tokens=200)]),
    _Exit("truncated", 1, [_call(), _said("", "length")]),
    _Exit("empty_response", 1, [_call(), _said(""), _said("")]),
    _Exit(
        "unfinished",
        0,
        [_said("I will check the run.")],
        {"continuation": lambda text: "Run it now."},
    ),
    _Exit(
        "force_stopped",
        6,
        [*[_call()] * 6, _said("here is what I found")],
        {"max_tool_rounds": 10},
    ),
    _Exit("error", 1, [_call(), _said("never")], fail_at=2),
]


async def _turn_end(exit_: _Exit, streaming: bool) -> tuple[LoopEndMarker, AIResponseEvent]:
    """The turn's end marker, and the ON_AI_RESPONSE that reports the same end."""
    provider = (
        _FailingAt(exit_.fail_at, exit_.responses, streaming=streaming)
        if exit_.fail_at is not None
        else MockAIProvider(ai_responses=exit_.responses, streaming=streaming)
    )
    settings: dict[str, Any] = {"max_tool_rounds": 5, "tool_loop_timeout_seconds": 120.0}
    ch = AIChannel(
        "ai1",
        provider=provider,
        tools=[_LOOKUP],
        tool_handler=_served,
        **{**settings, **exit_.channel},
    )
    reported: list[AIResponseEvent] = []

    async def report(event: AIResponseEvent) -> None:
        reported.append(event)

    ch._after_response_hook = report
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        # The turn's own budget: the channel sets none.
        metadata={"turn_budget_tokens": 500},
    )
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )
    assert output.response_stream is not None
    deltas: list[Any] = []
    try:
        async for delta in output.response_stream:
            deltas.append(delta)
    except ProviderError:
        # An ``error`` end: the exception follows the marker (RFC §6.4).
        assert exit_.reason == "error"
    [marker] = [d for d in deltas if isinstance(d, LoopEndMarker)]
    [response] = reported
    return marker, response


async def _marker(exit_: _Exit, streaming: bool) -> LoopEndMarker:
    return (await _turn_end(exit_, streaming))[0]


@pytest.mark.parametrize("exit_", _EXITS, ids=[exit_.reason for exit_ in _EXITS])
async def test_every_exit_states_the_limits_the_turn_ran_under(
    exit_: _Exit, streaming: bool
) -> None:
    marker, response = await _turn_end(exit_, streaming)

    assert (marker.reason, marker.rounds) == (exit_.reason, exit_.rounds)
    assert (response.loop_end_reason, response.round_count) == (marker.reason, marker.rounds)
    assert marker.max_rounds == exit_.channel.get("max_tool_rounds", 5)
    assert marker.timeout_seconds == exit_.channel.get("tool_loop_timeout_seconds", 120.0)
    assert (marker.budget_tokens, marker.budget_usd) == (500, None)


async def test_a_cancelled_turn_states_them_too(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """The exit taken before the loop's first round, which builds no rules."""
    monkeypatch.setattr(
        AIChannel, "_drain_steering_queue", lambda self, context, loop_ctx: (context, True)
    )

    marker = await _marker(_Exit("cancelled", 0, [_said("never")]), streaming)

    assert marker.reason == "cancelled"
    assert (marker.max_rounds, marker.timeout_seconds, marker.budget_tokens) == (5, 120.0, 500)


async def test_a_turn_without_a_deadline_or_a_budget_states_none(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_said("hello")], streaming=streaming)
    ch = AIChannel("ai1", provider=provider, tools=[_LOOKUP], tool_loop_timeout_seconds=None)
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )
    assert output.response_stream is not None
    [marker] = [d async for d in output.response_stream if isinstance(d, LoopEndMarker)]

    assert marker.reason == "completed"
    assert (marker.timeout_seconds, marker.budget_tokens, marker.budget_usd) == (None, None, None)
    assert marker.max_rounds == 50
