"""Each realtime tool call is delivered once and reported once (RFC §12.4, RMK-306).

The calls in flight on a session are on one book: a second call under an id
still running is refused without a result of its own, the input stays muted
while any call holds it, the session's end reports what it interrupted, and a
call whose result went out gets no second result nor a cancellation report.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit, ToolCallEvent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.context import RoomContext
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import until

TOOLS = [
    {"name": "slow", "description": "Slow tool", "parameters": {"type": "object"}},
    {"name": "lookup", "description": "Look up", "parameters": {"type": "object"}},
]
MANY = [
    {
        "name": f"tool_{i}",
        "description": f"weather forecast number {i}",
        "parameters": {"type": "object", "properties": {}},
    }
    for i in range(30)
]


class _Gated:
    """A handler whose ``slow`` calls wait for a release."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = 0

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.started += 1
        if name == "slow":
            await self.release.wait()
        return f"{name} done"


async def _channel(
    handler: Any, **kwargs: Any
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, Any, list[ToolCallEvent]]:
    provider = MockRealtimeProvider()
    kwargs.setdefault("tools", TOOLS)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=kwargs.pop("transport", MockRealtimeTransport()),
        tool_handler=handler,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
    async def observe(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    session = await channel.start_session("r1", "u1", "ws")
    return kit, channel, provider, session, observed


async def test_a_second_call_under_an_id_in_flight_is_refused_and_sends_nothing() -> None:
    handler = _Gated()
    kit, channel, provider, session, observed = await _channel(handler)

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await provider.simulate_tool_call(session, "c1", "slow", {})
    await until(lambda: bool(observed))

    assert handler.started == 1
    assert provider.tool_results == []
    assert [(e.tool_call_id, e.is_error) for e in observed] == [("c1", True)]
    assert "has not had its result yet" in json.loads(observed[0].result)["error"]

    handler.release.set()
    await until(lambda: bool(provider.tool_results))
    assert [r[2] for r in provider.tool_results] == ["slow done"]
    await kit.close()


async def test_the_input_stays_muted_until_the_last_call_ends() -> None:
    handler, transport = _Gated(), MockRealtimeTransport()
    kit, channel, provider, session, _ = await _channel(
        handler, transport=transport, mute_on_tool_call=True
    )

    await provider.simulate_tool_call(session, "c-slow", "slow", {})
    await provider.simulate_tool_call(session, "c-fast", "lookup", {})
    await until(lambda: len(provider.tool_results) == 1)
    assert _mutes(transport) == [True]  # the fast call ended; the slow one holds the input

    handler.release.set()
    await until(lambda: len(_mutes(transport)) == 2)
    assert _mutes(transport) == [True, False]
    await kit.close()


def _mutes(transport: MockRealtimeTransport) -> list[bool]:
    return [c.args["muted"] for c in transport.calls if c.method == "set_input_muted"]


async def test_the_sessions_end_reports_the_calls_it_interrupted() -> None:
    handler = _Gated()
    kit, channel, provider, session, observed = await _channel(handler)

    await provider.simulate_tool_call(session, "c1", "slow", {})
    await provider.simulate_transcription(session, "call:slow{}", "assistant")
    await until(lambda: handler.started == 2)
    await channel.end_session(session)

    assert sorted((e.name, e.cancelled) for e in observed) == [("slow", True), ("slow", True)]
    assert {e.tool_call_id for e in observed} >= {"c1"}
    await kit.close()


async def _search_channel(provider_reconfigure: Any) -> tuple[RoomKit, Any, Any, Any]:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    kit, channel, provider, session, observed = await _channel(
        handler, tools=MANY, tool_search=True
    )
    provider.reconfigure = provider_reconfigure  # type: ignore[method-assign]
    return kit, provider, session, observed


async def test_a_failed_reconfiguration_sends_no_second_result() -> None:
    """The observers judged the served search before its result went out
    (RMK-447): a reconfiguration that fails after is logged, not reported."""
    failed = asyncio.Event()

    async def fails(*args: Any, **kwargs: Any) -> None:
        failed.set()
        raise RuntimeError("socket closed during reconfigure")

    kit, provider, session, observed = await _search_channel(fails)

    await provider.simulate_tool_call(session, "c-find", "find_tools", {"query": "weather"})
    await asyncio.wait_for(failed.wait(), 5)
    await asyncio.sleep(0.02)

    assert len(provider.tool_results) == 1
    assert "matches" in json.loads(provider.tool_results[0][2])
    assert [(e.tool_call_id, e.is_error) for e in observed] == [("c-find", False)]
    await kit.close()


async def test_a_cancellation_after_the_result_went_out_is_not_reported() -> None:
    gate, steps = asyncio.Event(), []

    async def slow(*args: Any, **kwargs: Any) -> None:
        steps.append("started")
        await gate.wait()
        steps.append("finished")

    kit, provider, session, observed = await _search_channel(slow)

    await provider.simulate_tool_call(session, "c-find", "find_tools", {"query": "weather"})
    await until(lambda: bool(steps))
    await provider.simulate_tool_call_cancellation(session, ["c-find"])
    gate.set()
    await until(lambda: len(steps) == 2)
    await asyncio.sleep(0.02)

    assert steps == ["started", "finished"]
    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c-find", False)]
    await kit.close()


async def test_a_call_whose_handler_ends_its_session_is_reported_once() -> None:
    """A hang-up tool: the session's end does not take its own call for one it
    interrupted, so the call reports its own outcome, once."""
    box: dict[str, Any] = {}

    async def hang_up(name: str, arguments: dict[str, Any]) -> str:
        await box["channel"].end_session(box["session"])
        return "bye"

    kit, channel, provider, session, observed = await _channel(hang_up)
    framework_events: list[Any] = []

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        framework_events.append(event.data)

    box.update(channel=channel, session=session)
    await provider.simulate_tool_call(session, "c-bye", "lookup", {})
    await until(lambda: bool(observed))
    await asyncio.sleep(0.05)

    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c-bye", False)]
    assert len(framework_events) == 1
    await kit.close()


async def test_a_cancellation_while_the_observers_run_adds_no_second_report() -> None:
    """The judgement claims the call's report before its observers run, so a
    cancellation landing meanwhile finds the outcome reported: no second
    report, and nothing sent, the provider having freed the id (RFC §12.4)."""

    async def found(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    kit, channel, provider, session, observed = await _channel(found)
    in_observer, release = asyncio.Event(), asyncio.Event()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="slow-audit")
    async def slow_audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        in_observer.set()
        await release.wait()

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(in_observer.is_set)
    await provider.simulate_tool_call_cancellation(session, ["c1"])
    release.set()
    await asyncio.sleep(0.1)

    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", False)]
    assert provider.tool_results == []
    await kit.close()


async def test_a_refusal_is_on_the_wire_before_it_is_reported() -> None:
    kit, channel, provider, session, _ = await _channel(_Gated())
    on_wire: list[int] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="wire")
    async def wire(event: ToolCallEvent, ctx: RoomContext) -> None:
        on_wire.append(len(provider.tool_results))

    await provider.simulate_tool_call(session, "c1", "undeclared_tool", {})
    await until(lambda: bool(on_wire))

    assert on_wire == [1]
    await kit.close()
