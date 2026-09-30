"""The system prompt stays the same from turn to turn (RFC §6.4, RMK-318).

What the room's working memories say changes from one turn to the next (the
tools already used and what they returned, the plan): it rides the turn's
input, after the participant's words, in the same message, never the system
prompt, which a provider caches ahead of the whole history.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels._turn_notes import with_turn_notes
from roomkit.channels.ai import AIChannel
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_LOOKUP = AITool(name="lookup", description="Look up an order", parameters={})


def _round(call_id: str, name: str, arguments: dict[str, Any] | None = None) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments or {})],
    )


_DONE = AIResponse(content="done", finish_reason="stop")


async def _lookup(name: str, arguments: dict[str, Any]) -> str:
    return '{"order": "A-1042", "status": "SHIPPED-7731"}'


async def _turn(ch: AIChannel, body: str) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_LOOKUP.model_dump()]},
    )
    return await respond(
        ch,
        make_event(room_id="r1", body=body, channel_id="sms1"),
        binding,
        RoomContext(room=Room(id="r1")),
    )


def _input(context: AIContext) -> str:
    return str(context.messages[-1].content)


async def test_the_system_prompt_holds_while_the_digest_rides_the_input(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[_round("c0", "lookup"), _DONE, _DONE], streaming=streaming
    )
    ch = AIChannel("ai1", provider=provider, system_prompt="Be brief.", tool_handler=_lookup)

    await _turn(ch, "where is A-1042?")
    await _turn(ch, "and its status?")

    first, second = provider.calls[0], provider.calls[2]
    assert first.system_prompt == second.system_prompt == "Be brief."
    notes = _input(second)
    assert notes.startswith("and its status?")
    assert "Tools you've already used here" in notes
    assert "SHIPPED-7731" in notes
    assert "<tool_result>" in notes  # data set apart, not instructions


async def test_the_plan_rides_the_input(streaming: bool) -> None:
    plan = {"tasks": [{"title": "Check stock", "status": "pending"}]}
    provider = MockAIProvider(
        ai_responses=[_round("c0", "plan_tasks", plan), _DONE, _DONE], streaming=streaming
    )
    ch = AIChannel("ai1", provider=provider, enable_planning=True, tool_handler=_lookup)

    await _turn(ch, "plan it")
    await _turn(ch, "go on")

    assert "Check stock" not in (provider.calls[2].system_prompt or "")
    notes = _input(provider.calls[2])
    assert notes.startswith("go on")
    assert "## Current Task Plan" in notes and "Check stock" in notes


async def test_the_generation_hook_sees_the_notes_in_the_input(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[_round("c0", "lookup"), _DONE, _DONE], streaming=streaming
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_lookup)
    seen: list[AIContext] = []

    async def hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
        seen.append(gen_event.ai_context)
        return SyncPipelineResult(allowed=True)

    ch._before_generation_hook = hook

    await _turn(ch, "where is A-1042?")
    await _turn(ch, "and its status?")

    assert "SHIPPED-7731" in _input(seen[-1])
    assert "SHIPPED-7731" not in (seen[-1].system_prompt or "")


def test_notes_follow_a_multimodal_input_as_its_last_part() -> None:
    image = AIImagePart(url="https://example.com/a.png")
    messages = [AIMessage(role="user", content=[AITextPart(text="see this"), image])]

    [message] = with_turn_notes(messages, "NOTES")

    assert message.content == [AITextPart(text="see this"), image, AITextPart(text="NOTES")]


def test_notes_stand_alone_when_the_conversation_does_not_end_on_the_participant() -> None:
    messages = [
        AIMessage(role="user", content="hi"),
        AIMessage(role="assistant", content="hello"),
    ]

    result = with_turn_notes(messages, "NOTES")

    assert result[:2] == messages
    assert result[2] == AIMessage(role="user", content="NOTES")
    assert with_turn_notes(messages, None) == messages
    assert with_turn_notes([AIMessage(role="user", content="")], "NOTES")[0].content == "NOTES"
