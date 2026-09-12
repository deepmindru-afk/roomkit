"""Invalid calls must reach the same attempt budget as successfully dispatched calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_tool_arg_fold import BOARDS_TOOL, FOLDED_CALL, HOISTED_CALL
from tests.test_tool_repeat_guard import (
    _ECHO_TOOL,
    _binding,
    _same_call_response,
    _tool_results,
)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("args", [{"value": []}, {}, {"wrong": "value"}])
async def test_invalid_calls_stop_before_budget_and_reset_next_turn(
    streaming: bool, args: dict[str, Any]
) -> None:
    executions: list[dict[str, Any]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
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


@pytest.mark.parametrize("streaming", [False, True])
async def test_folded_equivalent_attempts_share_the_execution_budget(streaming: bool) -> None:
    executions: list[dict[str, Any]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        executions.append(arguments)
        return "{}"

    variants = [HOISTED_CALL, FOLDED_CALL, {**HOISTED_CALL, "params": {}}]
    provider = MockAIProvider(
        ai_responses=[
            *[
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id=str(i), name="boards", arguments=deepcopy(args))],
                )
                for i, args in enumerate(variants)
            ],
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    channel = AIChannel("ai1", provider=provider, tool_handler=handler)
    output = await channel.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding([BOARDS_TOOL]),
        RoomContext(room=Room(id="r1")),
    )
    if output.response_stream is not None:
        async for _ in output.response_stream:
            pass
    assert executions == [FOLDED_CALL, FOLDED_CALL]
    assert "EXACT arguments" in _tool_results(provider.calls[-1])[-1]["error"]
