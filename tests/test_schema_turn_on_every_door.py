"""A turn constrained to a response schema that its round cap cuts fails
``truncated`` on every door that reads its loop (RMK-479, RFC A.9).

As an agent's room turn and as a realtime reasoning backend, the turn's
``llm.generate`` span ends in error: the rule lives in the loop, not in the
envelope one door reads it through.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.test_framework import SimpleChannel

_LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _looping() -> MockAIProvider:
    """A tool round on every generation, under a response schema."""
    call = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
    )
    return MockAIProvider(
        ai_responses=[call], streaming=True, response_schema=True, response_schema_with_tools=True
    )


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


def _generate_spans(telemetry: MockTelemetryProvider) -> list[str]:
    return [s.status for s in telemetry.spans if s.kind == SpanKind.LLM_GENERATE]


async def test_a_cut_schema_room_turn_fails_its_span() -> None:
    telemetry = MockTelemetryProvider()
    kit = RoomKit(telemetry=telemetry)
    kit.register_channel(SimpleChannel("sms"))
    agent = Agent(
        "a",
        provider=_looping(),
        tools=[_LOOKUP],
        tool_handler=_found,
        tool_search=False,
        max_tool_rounds=1,
        response_schema=_SCHEMA,
    )
    kit.register_channel(agent)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "a", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )
    await kit.close()

    assert _generate_spans(telemetry) == ["error"]


async def test_a_cut_schema_backend_turn_fails_its_span() -> None:
    telemetry = MockTelemetryProvider()
    realtime = MockRealtimeProvider(full_duplex=True)
    backend = AgentReasoningBackend(
        Agent("reasoner", provider=_looping(), max_tool_rounds=1, response_schema=_SCHEMA)
    )
    channel = RealtimeVoiceChannel(
        "rt",
        provider=realtime,
        transport=MockRealtimeTransport(),
        tools=[{"name": "lookup", "description": "look up", "parameters": {"type": "object"}}],
        tool_handler=_found,
        reasoning_backend=backend,
        reasoning_timeout_s=5.0,
    )
    kit = RoomKit(telemetry=telemetry)
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    session = await channel.start_session("r", "u", "ws")

    await realtime.simulate_transcription(session, "find it", "user", False)
    await realtime.simulate_delegation(session, "d1", "integrator")
    for _ in range(100):
        if realtime.delegation_outputs:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert _generate_spans(telemetry) == ["error"]
    assert [text for _, _, text, _ in realtime.delegation_outputs] == [
        "The delegated work could not be completed."
    ]
