"""A realtime call an ending cuts before its task ran, or that arrives while
the session is torn down, is reported once, cancelled, on every door
(RFC §9.3, §12.4).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit

TOOLS = [
    {
        "name": "lookup",
        "description": "look up",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]


def _audit(kit: RoomKit) -> list[tuple[str, bool]]:
    seen: list[tuple[str, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append((event.name, event.cancelled))

    return seen


@pytest.mark.parametrize("door", ["provider", "recovered"])
async def test_a_call_the_session_s_end_cuts_before_its_task_ran_is_reported(door: str) -> None:
    provider = MockRealtimeProvider()
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
    session = await channel.start_session("r1", "u", "ws")
    if door == "provider":
        await provider.simulate_tool_call(session, "c1", "lookup", {"q": "x"})
    else:
        recovered, _ = channel._try_recover_tool_call_from_text(session, "call:lookup{q:x}")
        assert recovered
    # The session ends in the step the call arrived in, before its task ran.
    await channel.end_session(session)
    await asyncio.sleep(0.1)
    await kit.close()

    assert seen == [("lookup", True)] and ran == []


class _LateCallProvider(MockRealtimeProvider):
    """Hands on a call its receive loop had in flight when disconnect runs."""

    async def disconnect(self, session: Any) -> None:
        await self.simulate_tool_call(session, "late", "lookup", {})
        await asyncio.sleep(0.01)
        await super().disconnect(session)


@pytest.mark.parametrize("end", ["detach", "unplug"])
async def test_a_conference_reports_a_call_that_arrives_while_it_ends(end: str) -> None:
    provider = _LateCallProvider()
    config = ConferenceRealtimeConfig(provider=provider, tools=TOOLS, tool_handler=lambda *a: "ok")
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    seen = _audit(kit)
    await channel._realtime.ensure_session(ROOM)
    if end == "detach":
        await kit.detach_channel(ROOM, "conf")
    else:
        await channel.unplug_realtime()
    await asyncio.sleep(0.1)
    await kit.close()

    assert seen == [("lookup", True)]
