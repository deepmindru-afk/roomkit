"""Invalid calls must reach the same attempt budget as successfully dispatched calls."""

from __future__ import annotations

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_tool_repeat_guard import (
    _ECHO_TOOL,
    _binding,
    _same_call_response,
    _tool_results,
)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("args", [{"value": []}, {}, {"wrong": "value"}])
async def test_invalid_calls_stop_before_budget_and_reset_next_turn(streaming, args) -> None:
    executions = []

    async def handler(name, arguments):
        executions.append(arguments)
        return "{}"

    provider = MockAIProvider(
        ai_responses=[
            *[_same_call_response(f"t{i}", args) for i in range(6)],
            AIResponse(content="The tool failed; correct the arguments before retrying."),
        ],
        streaming=streaming,
    )
    channel = AIChannel("ai1", provider=provider, tool_handler=handler, max_tool_rounds=20)
    for turn in range(2):
        output = await channel.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding([_ECHO_TOOL]),
            RoomContext(room=Room(id="r1")),
        )
        if output.response_stream is not None:
            async for _ in output.response_stream:
                pass
        calls = provider.calls[turn * 7 :]
        assert len(calls) == 7  # six rejected attempts + one final answer, not 20 rounds
        assert calls[0].tools
        assert "Invalid arguments" in _tool_results(calls[1])[0]["error"]
        assert "EXACT arguments" in _tool_results(calls[3])[-1]["error"]
        assert not calls[-1].tools
        assert any("never claim an action succeeded" in str(m.content) for m in calls[-1].messages)
    assert not executions
