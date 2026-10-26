"""A realtime call an ending cuts before its task ran, or that arrives while
the session is torn down, is reported once, cancelled, on every door
(RFC §9.3, §12.4).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSessionState
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


async def _reported(seen: list[tuple[str, bool]]) -> None:
    """Until the cut call's report came; the kit's close lets a second one in."""
    async with asyncio.timeout(2):
        while not seen:
            await asyncio.sleep(0.005)


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
    await _reported(seen)
    await kit.close()

    assert seen == [("lookup", True)] and ran == []


async def test_a_call_after_its_session_ended_is_reported_and_leaves_no_state() -> None:
    """A provider call that arrives once its session ended is reported,
    cancelled, and keeps nothing of the session's: no depth of the model's
    answers is recorded for a session gone (RMK-983)."""
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), tools=TOOLS
    )
    kit = RoomKit()
    kit.register_channel(channel)
    seen = _audit(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    await channel.end_session(session)

    await provider.simulate_tool_call(session, "late", "lookup", {})
    await _reported(seen)
    await kit.close()

    assert seen == [("lookup", True)]
    assert session.id not in channel._answer_depth  # noqa: SLF001


class _OutlivedProvider(MockRealtimeProvider):
    """A provider whose session something outlives: on *fail*, it transcribes
    and the pipeline hears the user while ``connect()`` runs, then refuses
    the start; an injection waits on *gate* before it is sent."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = False
        self.channel: Any = None
        self.session: Any = None
        self.gate: asyncio.Event | None = None

    async def connect(self, session: Any, **kwargs: Any) -> None:
        await super().connect(session, **kwargs)
        self.session = session
        if self.fail:
            await self.simulate_transcription(session, "hello", "assistant", True)
            self.channel._on_pipeline_speech_start(session)  # noqa: SLF001
            raise ConnectionError("handshake refused")

    async def inject_text(self, session: Any, text: str, **kwargs: Any) -> Any:
        if self.gate is not None:
            await self.gate.wait()
        return await super().inject_text(session, text, **kwargs)


async def _transcribed_after_the_end(kit: RoomKit, channel: Any, provider: Any) -> None:
    session = await channel.start_session("r1", "u", "ws")
    await channel.end_session(session)
    await provider.simulate_transcription(session, "late words", "assistant", True)


async def _transcribed_while_its_task_waited(kit: RoomKit, channel: Any, provider: Any) -> None:
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_transcription(session, "late words", "assistant", True)
    await channel.end_session(session)


async def _transcribed_with_a_hook_at_the_end(kit: RoomKit, channel: Any, provider: Any) -> None:
    hooked, release = asyncio.Event(), asyncio.Event()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, name="slow")
    async def slow(event: Any, ctx: Any) -> HookResult:
        hooked.set()
        await release.wait()
        return HookResult.allow()

    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_transcription(session, "words", "assistant", True)
    await asyncio.wait_for(hooked.wait(), 2)
    await channel.end_session(session)
    release.set()


async def _injected_in_flight_at_the_end(kit: RoomKit, channel: Any, provider: Any) -> None:
    session = await channel.start_session("r1", "u", "ws")
    provider.gate = asyncio.Event()
    injecting = asyncio.create_task(channel.inject_text(session, "hello", chain_depth=1))
    await asyncio.sleep(0)
    await channel.end_session(session)
    provider.gate.set()
    await injecting


async def _started_and_failed(kit: RoomKit, channel: Any, provider: Any) -> None:
    provider.fail = True
    with pytest.raises(ConnectionError):
        await channel.start_session("r1", "u", "ws")


_OUTLIVED = {
    "a transcription after the end": _transcribed_after_the_end,
    "a transcription whose task waited for the end": _transcribed_while_its_task_waited,
    "a transcription whose hook ran at the end": _transcribed_with_a_hook_at_the_end,
    "an injection in flight at the end": _injected_in_flight_at_the_end,
    "a start that failed": _started_and_failed,
}


def _holding(channel: Any, session_id: str) -> list[str]:
    """The channel's records that still name *session_id*."""
    return sorted(
        name
        for name, value in vars(channel).items()
        if isinstance(value, dict | set | frozenset) and session_id in value
    )


@pytest.mark.parametrize("when", list(_OUTLIVED))
async def test_what_its_session_outlives_leaves_no_state(when: str) -> None:
    """What reaches a realtime session as it ends, or while a start that then
    fails runs, keeps nothing of the session's: no record of the channel
    names it once it is gone (RMK-983)."""
    provider = _OutlivedProvider()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
    provider.channel = channel
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")

    await _OUTLIVED[when](kit, channel, provider)
    for _ in range(20):  # a transcription's task, had it been started
        await asyncio.sleep(0)
    await kit.close()

    assert _holding(channel, provider.session.id) == []


async def test_the_last_words_before_the_connection_drops_are_recorded() -> None:
    """A provider that loses its connection marks the session ended before
    it reports the loss, the last transcription already handed on: its room
    still holds the session, and records the words (RMK-983)."""
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")

    await provider.simulate_transcription(session, "last words", "assistant", True)
    session.state = VoiceSessionState.ENDED
    await provider.simulate_error(session, "connection_closed", "socket closed")
    async with asyncio.timeout(2):
        while not any(
            getattr(e.content, "body", None) == "last words"
            for e in await kit.store.list_events("r1")
        ):
            await asyncio.sleep(0.005)
    await kit.close()


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
    await _reported(seen)
    await kit.close()

    assert seen == [("lookup", True)]
