"""A call an ending interrupts is reported once, cancelled, on the conference
as on the realtime channel, and when the session's start fails
(RMK-477, RFC §9.3, §12.4).

Every ending of each door: a session's end and the channel's close; a
conference's detach, the realtime's unplug, and a detach after the bot was
lost. Every way a start fails: its handshake rolled back, its session ended
once filed, the start cancelled before or after the provider's call, the
channel closed while the leg rings, a connection that was no awaitable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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


Ending = Callable[[], Awaitable[Any]]


async def _session_door() -> tuple[RoomKit, MockRealtimeProvider, Any, dict[str, Ending]]:
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
    endings: dict[str, Ending] = {
        "end_session": lambda: channel.end_session(session),
        "close": channel.close,
    }
    return kit, provider, session, endings


async def _conference_door() -> tuple[RoomKit, MockRealtimeProvider, Any, dict[str, Ending]]:
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(provider=provider, tools=TOOLS, tool_handler=_never)
    kit, channel, backend, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)

    async def bot_lost_then_detach() -> None:
        await backend.simulate_bot_disconnected(backend.bots[-1])
        await asyncio.sleep(0.01)  # the lost bot's disconnect waits on the reports
        await kit.detach_channel(ROOM, "conf")

    endings: dict[str, Ending] = {
        "detach": lambda: kit.detach_channel(ROOM, "conf"),
        "unplug": channel.unplug_realtime,
        "bot-lost-detach": bot_lost_then_detach,
    }
    return kit, provider, session, endings


DOORS = {"session": _session_door, "conference": _conference_door}
ENDINGS = [
    ("session", "end_session"),
    ("session", "close"),
    ("conference", "detach"),
    ("conference", "unplug"),
    ("conference", "bot-lost-detach"),
]
EVERY_ENDING = pytest.mark.parametrize(("door", "ending"), ENDINGS)


@EVERY_ENDING
@pytest.mark.parametrize("held", [False, True], ids=["same-step", "report-waiting"])
async def test_a_refused_duplicate_an_ending_cuts_is_reported(
    door: str, ending: str, held: bool
) -> None:
    kit, provider, session, endings = await DOORS[door]()
    seen = _audit(kit)
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.sleep(0.01)  # the first call runs its handler
    gate = _hold_reports(kit) if held else None
    await provider.simulate_tool_call(session, "c1", "lookup", {})  # its duplicate
    if gate is not None:
        await asyncio.sleep(0.01)  # the duplicate's report waits on its context
        asyncio.get_running_loop().call_later(0.1, gate.set)
    await endings[ending]()
    await asyncio.sleep(0.3)
    await kit.close()

    # The first call cancelled, and the duplicate reported once, whatever cut it.
    assert len(seen) == 2
    assert ("c1", True, False) in seen


@EVERY_ENDING
async def test_an_abandonment_report_an_ending_cuts_is_kept(door: str, ending: str) -> None:
    kit, provider, session, endings = await DOORS[door]()
    seen = _audit(kit)
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.sleep(0.01)
    gate = _hold_reports(kit)
    await provider.simulate_tool_call_cancellation(session, ["c1"])
    await asyncio.sleep(0.01)  # the handler is cancelled, its report waits
    asyncio.get_running_loop().call_later(0.1, gate.set)
    await endings[ending]()
    await asyncio.sleep(0.3)
    await kit.close()

    assert seen == [("c1", True, False)]


class _CallingOnConnect(MockRealtimeProvider):
    """Calls ``lookup`` as soon as it is connected."""

    async def connect(self, session: Any, **kwargs: Any) -> None:
        await super().connect(session, **kwargs)
        await self.simulate_tool_call(session, "c1", "lookup", {})


class _ClientGoneTransport(MockRealtimeTransport):
    """Loses the client as the session's start is announced to it."""

    async def send_message(self, session: Any, message: dict[str, Any]) -> None:
        if message.get("type") == "session_started":
            raise ConnectionError("client went away")
        await super().send_message(session, message)


class _DroppedWhileConnecting(_CallingOnConnect):
    """Calls, then loses the client before its handshake ends."""

    transport: MockRealtimeTransport

    async def connect(self, session: Any, **kwargs: Any) -> None:
        await super().connect(session, **kwargs)
        await self.transport.simulate_client_disconnect(session)
        await asyncio.sleep(0.05)


async def _never_answered() -> Any:
    await asyncio.sleep(0.01)
    raise RuntimeError("the leg was never answered")


async def _answered() -> str:
    await asyncio.sleep(0.01)
    return "ws"


async def _failing_start(way: str, channel: RealtimeVoiceChannel, kit: RoomKit) -> None:
    """Fail the session's start in *way*, its failure swallowed."""
    provider = channel._provider
    assert isinstance(provider, MockRealtimeProvider)
    loop = asyncio.get_running_loop()
    if way in ("rolled-back", "session-ended", "not-deferred"):
        connection: Any = {"rolled-back": _never_answered, "session-ended": _answered}.get(
            way, lambda: "ws"
        )()
        await asyncio.gather(channel.start_session("r1", "u", connection), return_exceptions=True)
        return
    ringing = loop.create_future()  # the leg never answers
    start = asyncio.create_task(channel.start_session("r1", "u", ringing))
    await asyncio.sleep(0.05)  # the provider is connected, the leg rings
    if way == "closed-ringing":
        gate = _hold_reports(kit)
        loop.call_later(0.1, gate.set)
        await channel.close()
    else:
        session = next(iter(provider._sessions.values()))
        if way == "call-then-cancel":
            await provider.simulate_tool_call(session, "c1", "lookup", {})
        start.cancel()
        if way == "cancel-then-call":
            await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.gather(start, return_exceptions=True)


_FAILED_STARTS = {
    "rolled-back": (_CallingOnConnect, MockRealtimeTransport),
    "session-ended": (_CallingOnConnect, _ClientGoneTransport),
    "not-deferred": (_DroppedWhileConnecting, MockRealtimeTransport),
    "closed-ringing": (_CallingOnConnect, MockRealtimeTransport),
    "call-then-cancel": (MockRealtimeProvider, MockRealtimeTransport),
    "cancel-then-call": (MockRealtimeProvider, MockRealtimeTransport),
}


@pytest.mark.parametrize("way", list(_FAILED_STARTS))
async def test_a_call_issued_while_a_failed_start_was_pending_is_reported(way: str) -> None:
    provider_kind, transport_kind = _FAILED_STARTS[way]
    provider = provider_kind()
    transport = transport_kind()
    if isinstance(provider, _DroppedWhileConnecting):
        provider.transport = transport
    finished: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        await asyncio.sleep(0.5)
        finished.append(name)
        return "found"

    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=transport, tools=TOOLS, tool_handler=handler
    )
    kit = RoomKit()
    kit.register_channel(channel)
    seen = _audit(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")

    await _failing_start(way, channel, kit)
    await asyncio.sleep(0.6)
    await kit.close()

    assert seen == [("c1", True, False)]
    assert finished == [] and provider.tool_results == []
