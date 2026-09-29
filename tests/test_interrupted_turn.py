"""A turn cut short delivers nothing twice and nothing from an earlier turn (RFC §6.4; RMK-156).

The room already holds each round's text as its own message. A turn the
provider interrupts after a round ends on the marker alone; a turn cancelled
between rounds adds no terminal text. The history the model was given is
context, never this turn's output.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent
from roomkit.models.steering import Cancel
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

T = AITool(name="t", description="a tool", parameters={"type": "object", "properties": {}})
MARKER = "[Response interrupted]"


class _FailingAt(MockAIProvider):
    """Scripted answers, and a provider error on the given generation."""

    def __init__(self, fail_at: int, answers: list[AIResponse], *, streaming: bool) -> None:
        super().__init__(ai_responses=answers, streaming=streaming)
        self._fail_at = fail_at
        self._generations = 0

    async def generate(self, context: AIContext) -> AIResponse:
        # The mock's stream draws its answers from generate() too: one count
        # per generation, whichever loop runs.
        self._generations += 1
        if self._generations == self._fail_at:
            raise ProviderError("upstream 500 req_abc123", retryable=False, status_code=500)
        return await super().generate(context)


async def _room(provider: MockAIProvider, **channel: Any) -> tuple[RoomKit, AIChannel, list[Any]]:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    ai = AIChannel(
        "ai1", provider=provider, tools=[T], tool_handler=handler, tool_search=False, **channel
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    responses: list[Any] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="spy")
    async def spy(event: Any, ctx: Any) -> None:
        responses.append(event)

    return kit, ai, responses


async def _say(kit: RoomKit, *bodies: str) -> None:
    for body in bodies:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body=body))
        )
        await asyncio.sleep(0.05)


async def _ai_messages(kit: RoomKit) -> list[str]:
    events = await kit.store.list_events("r1")
    return [
        e.content.body
        for e in events
        if e.type == EventType.MESSAGE
        and e.source.channel_id == "ai1"
        and isinstance(e.content, TextContent)
    ]


def _looking(content: str = "Looking.") -> AIResponse:
    return AIResponse(
        content=content,
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c1", name="t", arguments={})],
    )


async def test_an_interrupted_turn_replays_nothing(streaming: bool) -> None:
    answers = [AIResponse(content="Earlier answer."), _looking(), AIResponse(content="never")]
    kit, _, _ = await _room(_FailingAt(3, answers, streaming=streaming))

    await _say(kit, "first", "second")

    messages = await _ai_messages(kit)
    assert messages[0] == "Earlier answer."
    assert all("Earlier answer." not in m for m in messages[1:])
    assert sum(m.count("Looking.") for m in messages) == 1
    await kit.close()


@pytest.mark.parametrize("streaming", [False], ids=["non-streaming"])
async def test_an_interrupted_turn_ends_on_the_marker_alone(streaming: bool) -> None:
    answers = [AIResponse(content="Earlier answer."), _looking(), AIResponse(content="never")]
    kit, _, responses = await _room(_FailingAt(3, answers, streaming=streaming))

    await _say(kit, "first", "second")

    assert await _ai_messages(kit) == ["Earlier answer.", "Looking.", MARKER]
    # The hook's transcript is the turn's segments, once each.
    assert responses[-1].response_content == f"Looking.\n\n{MARKER}"
    assert "req_abc123" not in responses[-1].response_content
    await kit.close()


@pytest.mark.parametrize("streaming", [False], ids=["non-streaming"])
async def test_an_interrupted_round_without_text_keeps_its_calls(streaming: bool) -> None:
    kit, _, _ = await _room(
        _FailingAt(2, [_looking(""), AIResponse(content="never")], streaming=streaming)
    )

    await _say(kit, "go")

    events = await kit.store.list_events("r1")
    kinds = [e.type for e in events if e.source.channel_id == "ai1"]
    assert EventType.TOOL_CALL_START in kinds
    assert EventType.TOOL_CALL_END in kinds
    assert await _ai_messages(kit) == [MARKER]
    await kit.close()


@pytest.mark.parametrize("streaming", [False], ids=["non-streaming"])
async def test_a_turn_cancelled_between_rounds_adds_no_terminal_text(streaming: bool) -> None:
    holder: dict[str, AIChannel] = {}
    provider = MockAIProvider(
        ai_responses=[_looking(), AIResponse(content="never")], streaming=streaming
    )
    kit, ai, responses = await _room(provider)
    holder["ai"] = ai

    async def cancelling(name: str, arguments: dict[str, Any]) -> str:
        holder["ai"].steer(Cancel())
        return "ok"

    ai._user_tool_handler = cancelling

    await _say(kit, "go")

    assert await _ai_messages(kit) == ["Looking."]
    assert responses[-1].response_content == "Looking."
    await kit.close()
