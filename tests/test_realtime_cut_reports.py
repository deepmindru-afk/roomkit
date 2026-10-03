"""A realtime call cut while it was reported, or issued once its session
ended, is reported once (RMK-431, RFC §9.3, §12.4).

A Tool Search call whose result went out is reported after delivery: an
ending that cuts that report leaves it owed, with what the model read. A
reasoning backend's call issued after its session ended runs no gate and is
reported once, cancelled.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    ToolCallEvent,
    ToolCallResult,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest

MANY = [
    {
        "name": f"tool_{i}",
        "description": f"weather forecast {i}",
        "parameters": {"type": "object", "properties": {}},
    }
    for i in range(30)
]
HANGUP_THEN_LOOKUP = [
    {"name": "hangup", "parameters": {"type": "object"}},
    {"name": "lookup", "parameters": {"type": "object"}},
]


async def _until(condition: Any) -> None:
    for _ in range(300):
        if condition():
            return
        await asyncio.sleep(0.01)


async def _session(channel: RealtimeVoiceChannel) -> tuple[RoomKit, Any]:
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", channel.channel_id)
    return kit, await channel.start_session("r1", "u1", "ws")


def _observe(kit: RoomKit) -> list[ToolCallEvent]:
    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


async def test_a_search_call_whose_report_the_ending_cut_keeps_what_the_model_read() -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=MANY,
        tool_search=True,
    )
    kit, session = await _session(channel)
    held = asyncio.Event()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="slow")
    async def slow(event: ToolCallEvent, ctx: Any) -> HookResult:
        held.set()
        await asyncio.sleep(30)
        return HookResult.allow()

    seen = _observe(kit)
    await provider.simulate_tool_call(session, "c1", "find_tools", {"query": "weather forecast"})
    await asyncio.wait_for(held.wait(), 5)
    await channel.end_session(session)
    await _until(lambda: bool(seen))
    await asyncio.sleep(0.05)

    [sent] = provider.tool_results
    [report] = seen
    assert (report.name, report.is_error, report.cancelled) == ("find_tools", False, False)
    assert report.result == sent[2]
    await kit.close()


class _HangsUpThenLooksUp(ReasoningBackend):
    def __init__(self) -> None:
        self.results: list[tuple[str, ToolCallResult]] = []

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        assert request.execute_tool_call is not None
        for name in ("hangup", "lookup"):
            self.results.append((name, await request.execute_tool_call(name, {})))
        yield ReasoningOutput("done", is_final=True)


async def test_a_backend_call_after_its_session_ended_is_reported_once_cancelled() -> None:
    provider = MockRealtimeProvider(full_duplex=True)
    holder: dict[str, Any] = {}
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        if name == "hangup":
            await holder["channel"].end_session(holder["session"])
            return "bye"
        return "found"

    backend = _HangsUpThenLooksUp()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=HANGUP_THEN_LOOKUP,
        reasoning_backend=backend,
    )
    holder["channel"] = channel
    kit, session = await _session(channel)
    holder["session"] = session
    gated: list[str] = []

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="gate")
    async def gate(event: ToolCallEvent, ctx: Any) -> HookResult:
        gated.append(event.name)
        return HookResult.allow()

    seen = _observe(kit)
    await provider.simulate_delegation(session, "d1", "integrator")
    await _until(lambda: len(backend.results) == 2)
    await asyncio.sleep(0.1)

    assert ran == ["hangup"]
    # Whoever issued it is gone: no gate runs for it.
    assert gated == ["hangup"]
    [(_, read)] = [r for r in backend.results if r[0] == "lookup"]
    assert read.is_error
    assert json.loads(read.text)["error"] == "Tool call cancelled"
    assert [(e.name, e.is_error, e.cancelled) for e in seen] == [
        ("hangup", False, False),
        ("lookup", True, True),
    ]
    await kit.close()
