"""A call an ending interrupts is reported once, cancelled, on the conference
as on the realtime channel, and when the session's start fails
(RMK-477, RFC §9.3, §12.4).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit

TOOLS = [{"name": "lookup", "description": "look up", "parameters": {"type": "object"}}]


def _hold_reports(kit: RoomKit) -> asyncio.Event:
    """Hold every ON_TOOL_CALL report on its hook context until set."""
    gate = asyncio.Event()
    original = kit._hook_context

    async def slow(room_id: str, trigger: Any, **kwargs: Any) -> Any:
        if trigger == HookTrigger.ON_TOOL_CALL:
            await gate.wait()
        return await original(room_id, trigger, **kwargs)

    kit._hook_context = slow  # type: ignore[method-assign]
    return gate


def _audit(kit: RoomKit) -> list[tuple[str, bool, bool]]:
    seen: list[tuple[str, bool, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append((event.tool_call_id, bool(event.cancelled), bool(event.refused)))

    return seen


async def _never(*_: Any) -> str:
    await asyncio.Event().wait()
    return "late"


async def _session_door() -> tuple[RoomKit, MockRealtimeProvider, Any, Any]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=_never,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    return kit, provider, session, lambda: channel.end_session(session)


async def _conference_door() -> tuple[RoomKit, MockRealtimeProvider, Any, Any]:
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(provider=provider, tools=TOOLS, tool_handler=_never)
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    return kit, provider, session, lambda: kit.detach_channel(ROOM, "conf")


DOORS = {"session": _session_door, "conference": _conference_door}


@pytest.mark.parametrize("door", list(DOORS))
@pytest.mark.parametrize("held", [False, True], ids=["same-step", "report-waiting"])
async def test_a_refused_duplicate_an_ending_cuts_is_reported(door: str, held: bool) -> None:
    kit, provider, session, end = await DOORS[door]()
    seen = _audit(kit)
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.sleep(0.01)  # the first call runs its handler
    gate = _hold_reports(kit) if held else None
    await provider.simulate_tool_call(session, "c1", "lookup", {})  # its duplicate
    if gate is not None:
        await asyncio.sleep(0.01)  # the duplicate's report waits on its context
        asyncio.get_running_loop().call_later(0.1, gate.set)
    await end()
    await asyncio.sleep(0.3)
    await kit.close()

    # The first call cancelled, and the duplicate reported once, whatever cut it.
    assert len(seen) == 2
    assert ("c1", True, False) in seen


@pytest.mark.parametrize("door", list(DOORS))
async def test_an_abandonment_report_an_ending_cuts_is_kept(door: str) -> None:
    kit, provider, session, end = await DOORS[door]()
    seen = _audit(kit)
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.sleep(0.01)
    gate = _hold_reports(kit)
    await provider.simulate_tool_call_cancellation(session, ["c1"])
    await asyncio.sleep(0.01)  # the handler is cancelled, its report waits
    asyncio.get_running_loop().call_later(0.1, gate.set)
    await end()
    await asyncio.sleep(0.3)
    await kit.close()

    assert seen == [("c1", True, False)]


class _CallingOnConnect(MockRealtimeProvider):
    async def connect(self, session: Any, **kwargs: Any) -> None:
        await super().connect(session, **kwargs)
        await self.simulate_tool_call(session, "c1", "lookup", {})


async def test_a_call_issued_while_a_failed_start_was_pending_is_reported() -> None:
    provider = _CallingOnConnect()
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "found"

    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    seen = _audit(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")

    async def never_answered() -> Any:
        await asyncio.sleep(0.01)
        raise RuntimeError("the leg was never answered")

    with pytest.raises(RuntimeError):
        await channel.start_session("r1", "u", never_answered())
    await asyncio.sleep(0.2)
    await kit.close()

    assert seen == [("c1", True, False)]
    assert ran == [] and provider.tool_results == []
