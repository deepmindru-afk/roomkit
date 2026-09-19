"""``ON_AI_RESPONSE`` reports the tools the provider received, revealed ones included.

``BEFORE_AI_GENERATION`` fires once, with the toolset the turn starts from.
Under Tool Search that is the pinned floor plus ``find_tools`` / ``list_tools``;
a tool ``find_tools`` reveals enters the declaration on the next round only,
and no event carried it: a host recording "what the model was offered" from
that hook never saw a revealed tool. ``AIResponseEvent.declared_tools`` is the
union, over the turn's generation rounds, of what each provider call declared,
each entry naming why Tool Search let it through.
"""

from __future__ import annotations

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIResponseEvent, DeclaredTool
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

_SMS_TOOL = {
    "name": "send_sms",
    "description": "Send an SMS text message to a phone number.",
    "parameters": {
        "type": "object",
        "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to", "body"],
    },
}

# No word in common with ``send_sms`` (name or description), so a query for
# one never scores the other.
_MAIL_TOOL = {
    "name": "mail_deliver",
    "description": "Deliver an electronic mail to an address.",
    "parameters": {
        "type": "object",
        "properties": {"address": {"type": "string"}},
        "required": ["address"],
    },
}


def _catalogue(n: int) -> list[dict]:
    """``n`` noise tools, lexically unrelated to the two above."""
    return [
        {"name": f"widget_{i}", "description": f"Operate widget number {i}."} for i in range(n)
    ]


def _binding(tools: list[dict]) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": tools},
    )


async def _noop_handler(name: str, arguments: dict) -> str:
    return "ok"


def _find(query: str, call_id: str) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name="find_tools", arguments={"query": query})],
    )


def _observe(ch: AIChannel) -> list[AIResponseEvent]:
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    ch._after_response_hook = observe
    return seen


async def _turn(ch: AIChannel, binding: ChannelBinding) -> None:
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1", room_id="r1"),
        binding,
        RoomContext(room=Room(id="r1")),
    )
    # A streaming provider runs the loop as the stream is drained.
    if output.response_stream is not None:
        async for _ in output.response_stream:
            pass


def _by_name(event: AIResponseEvent) -> dict[str, DeclaredTool]:
    return {tool.name: tool for tool in event.declared_tools}


def _round_names(context: AIContext) -> set[str]:
    return {tool.name for tool in context.tools or []}


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_revealed_tool_is_reported_with_its_schema(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[_find("send sms", "t1"), AIResponse(content="done", finish_reason="stop")],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_search=True,
        tool_search_pinned=["widget_1"],
        tool_handler=_noop_handler,
    )
    seen = _observe(ch)

    await _turn(ch, _binding([*_catalogue(5), _SMS_TOOL]))

    # The first round, all BEFORE_AI_GENERATION ever sees: no send_sms.
    assert "send_sms" not in _round_names(provider.calls[0])
    assert "send_sms" in _round_names(provider.calls[1])

    assert len(seen) == 1
    declared = _by_name(seen[0])
    assert declared["send_sms"].origin == "revealed"
    assert declared["send_sms"].parameters == _SMS_TOOL["parameters"]
    assert declared["send_sms"].description == _SMS_TOOL["description"]
    assert declared["widget_1"].origin == "pinned"
    assert declared["find_tools"].origin == "always"
    assert declared["list_tools"].origin == "always"
    # The catalogue behind find_tools was never declared, so it is not here.
    assert not any(name.startswith("widget_") for name in declared if name != "widget_1")


@pytest.mark.parametrize("streaming", [False, True])
async def test_without_tool_search_the_whole_declaration_is_reported(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[AIResponse(content="hi", finish_reason="stop")], streaming=streaming
    )
    ch = AIChannel("ai1", provider=provider, tool_search=False, tool_handler=_noop_handler)
    seen = _observe(ch)

    await _turn(ch, _binding([_SMS_TOOL, _MAIL_TOOL]))

    assert len(seen) == 1
    declared = _by_name(seen[0])
    assert set(declared) == {"send_sms", "mail_deliver"} == _round_names(provider.calls[0])
    assert {tool.origin for tool in declared.values()} == {"always"}
    assert declared["mail_deliver"].parameters == _MAIL_TOOL["parameters"]


async def test_the_union_keeps_a_tool_the_reveal_window_dropped() -> None:
    """``revealed_tools`` is swapped by every ``find_tools``; the event unions the rounds."""
    provider = MockAIProvider(
        ai_responses=[
            _find("sms phone", "t1"),
            _find("electronic mail", "t2"),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    ch = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=_noop_handler)
    seen = _observe(ch)

    await _turn(ch, _binding([*_catalogue(5), _SMS_TOOL, _MAIL_TOOL]))

    # Round 1 declared send_sms; the second find_tools slid the window, so
    # round 2 declared mail_deliver and not send_sms.
    assert "send_sms" in _round_names(provider.calls[1])
    assert "mail_deliver" not in _round_names(provider.calls[1])
    assert "mail_deliver" in _round_names(provider.calls[2])
    assert "send_sms" not in _round_names(provider.calls[2])

    declared = _by_name(seen[0])
    assert declared["send_sms"].origin == "revealed"
    assert declared["mail_deliver"].origin == "revealed"


async def test_a_tool_found_in_an_earlier_turn_is_sticky() -> None:
    provider = MockAIProvider(
        ai_responses=[
            _find("send sms", "t1"),
            AIResponse(content="done", finish_reason="stop"),
            AIResponse(content="hi again", finish_reason="stop"),
        ]
    )
    ch = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=_noop_handler)
    seen = _observe(ch)
    binding = _binding([*_catalogue(5), _SMS_TOOL])

    await _turn(ch, binding)
    await _turn(ch, binding)

    assert len(seen) == 2
    assert _by_name(seen[0])["send_sms"].origin == "revealed"
    # Turn 2 re-exposes it from the room's tool usage before any find_tools.
    assert "send_sms" in _round_names(provider.calls[2])
    assert _by_name(seen[1])["send_sms"].origin == "sticky"


async def test_a_text_only_stream_reports_its_declaration_too() -> None:
    """The streaming path without a tool loop fills the field like the others."""
    provider = MockAIProvider(
        ai_responses=[AIResponse(content="hello", finish_reason="stop")], streaming=True
    )
    ch = AIChannel("ai1", provider=provider)
    seen = _observe(ch)

    await _turn(ch, _binding([]))

    assert len(seen) == 1
    assert seen[0].declared_tools == []
