"""A turn's tool declaration holds from round to round (RFC §6.4, RMK-317).

A provider caches a request as a prefix, tools first: a declaration that
gains, loses or reorders a tool is billed as if nothing were cached. So the
large-result re-read is declared from the first round, last, not from the
round a result is first stored; the anti-loop stop keeps the round's tools;
and the declaration of a turn is the one the next turn starts from.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_SEARCH = AITool(name="search", description="Search the archive", parameters={})
_NOTE = AITool(name="write_note", description="Write a note", parameters={})
_LARGE = "\n".join(f"line {i} " + "x" * 80 for i in range(400))


def _round(call_id: str, name: str, arguments: dict[str, Any] | None = None) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments or {})],
    )


_DONE = AIResponse(content="done", finish_reason="stop")


async def _large(name: str, arguments: dict[str, Any]) -> str:
    return _LARGE


def _names(context: AIContext) -> list[str]:
    return [tool.name for tool in context.tools or []]


async def _turn(ch: AIChannel, tools: list[AITool]) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [tool.model_dump() for tool in tools]},
    )
    return await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )


async def test_an_eviction_leaves_the_declaration_as_it_was(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _round("c0", "search"),
            _round("c1", "read_stored_result", {"result_id": "evicted_c0"}),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_large, evict_threshold_tokens=500)

    run = await _turn(ch, [_SEARCH, _NOTE])

    declared = [_names(call) for call in provider.calls]
    assert declared[0] == ["search", "write_note", "read_stored_result"]
    assert declared == [declared[0]] * 3
    assert run.calls[-1].name == "read_stored_result"
    assert "line 0 " in str(run.calls[-1].result)


async def test_a_turn_without_tools_declares_no_re_read(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
    ch = AIChannel("ai1", provider=provider)

    await _turn(ch, [])

    assert _names(provider.calls[0]) == []


async def test_the_next_turn_starts_from_the_declaration_this_one_ended_with(
    streaming: bool,
) -> None:
    provider = MockAIProvider(
        ai_responses=[_round("c0", "search"), _DONE, _round("c1", "search"), _DONE],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_large, evict_threshold_tokens=500)

    await _turn(ch, [_SEARCH, _NOTE])
    await _turn(ch, [_SEARCH, _NOTE])

    declared = [_names(call) for call in provider.calls]
    assert declared == [declared[0]] * 4


async def test_the_forced_last_round_keeps_its_tools_and_runs_none(streaming: bool) -> None:
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "same"

    repeats = [_round(f"c{i}", "search", {"q": "x"}) for i in range(6)]
    provider = MockAIProvider(
        ai_responses=[*repeats, _round("after", "write_note"), _DONE], streaming=streaming
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)

    run = await _turn(ch, [_SEARCH, _NOTE])

    declared = [_names(call) for call in provider.calls]
    assert len(declared) == 7
    assert declared == [declared[0]] * 7
    assert "write_note" not in ran
    assert run.reason == "force_stopped"
