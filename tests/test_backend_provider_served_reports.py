"""A call a reasoning backend's own provider served is reported, as on an
AIChannel (RMK-480, RFC §9.3, §12.4.1).

The provider ran the call itself (``AIToolCall.served``): the voice channel's
ON_TOOL_CALL hooks hear it once through ``ReasoningRequest.report_call``,
served or failed, with what the provider returned.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.tool_doors import run_door


async def _unused(name: str, arguments: dict[str, Any]) -> str:
    raise AssertionError("the provider served the call")


@pytest.mark.parametrize("door", ["text-stream", "text-nostream", "rt-agent-backend"])
@pytest.mark.parametrize(
    ("served", "expected"),
    [
        (ServedCall(result="3 hits"), (False, "3 hits")),
        (ServedCall(result="quota", is_error=True), (True, "quota")),
    ],
    ids=["served", "failed"],
)
async def test_a_call_the_provider_served_is_reported_once(
    door: str, served: ServedCall, expected: tuple[bool, str]
) -> None:
    call = AIToolCall(id="c1", name="web_search", arguments={"q": "x"}, served=served)
    seen = await run_door(door, _unused, call=call)

    assert [(e.name, e.is_error, e.result) for e in seen.reports] == [("web_search", *expected)]


async def test_a_provider_served_call_cut_while_judged_is_reported_once_served() -> None:
    """The delegation ends while the voice channel's ON_TOOL_CALL chain
    judges the call: it is still reported once, served, never a second time
    as failed by the agent's turn end (RFC §9.3)."""
    call = AIToolCall(
        id="p1", name="web_search", arguments={"q": "x"}, served=ServedCall(result="3 hits")
    )
    ai = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    provider = MockRealtimeProvider(full_duplex=True)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=_unused,
        tools=[{"name": "lookup", "description": "x", "parameters": {"type": "object"}}],
        reasoning_backend=AgentReasoningBackend(Agent("reasoner", provider=ai)),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")
    judging = asyncio.Event()
    reports: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="slow")
    async def slow(event: Any, ctx: Any) -> HookResult:
        judging.set()
        await asyncio.sleep(0.3)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        reports.append(event)

    await provider.simulate_delegation(session, "d1", "integrator")
    await asyncio.wait_for(judging.wait(), 3)
    await channel.end_session(session)
    await asyncio.sleep(0.5)
    await kit.close()

    assert [(e.name, e.is_error, e.result) for e in reports] == [("web_search", False, "3 hits")]
