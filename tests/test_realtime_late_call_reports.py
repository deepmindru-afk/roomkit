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


class _TranscribingProvider(MockRealtimeProvider):
    """Transcribes what the model said while ``connect()`` runs, then, on
    *fail*, refuses the start."""

    def __init__(self, *, fail: bool) -> None:
        super().__init__()
        self._fail = fail

    async def connect(self, session: Any, **kwargs: Any) -> None:
        await super().connect(session, **kwargs)
        if self._fail:
            await self.simulate_transcription(session, "hello", "assistant", True)
            raise ConnectionError("handshake refused")


@pytest.mark.parametrize(
    "when",
    ["after its session ended", "while its task waited for the end", "during a start that failed"],
)
async def test_a_transcription_its_session_outlives_leaves_no_state(when: str) -> None:
    """A transcription that arrives once its session ended, or while a start
    that then fails runs, is not the room's to record, and keeps nothing of
    the session's: no depth of the model's answers, no ordering lock, no last
    text (RMK-983)."""
    provider = _TranscribingProvider(fail=when == "during a start that failed")
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    if when == "after its session ended":
        session = await channel.start_session("r1", "u", "ws")
        await channel.end_session(session)
        await provider.simulate_transcription(session, "late words", "assistant", True)
    elif when == "while its task waited for the end":
        session = await channel.start_session("r1", "u", "ws")
        # Arrives while the session is live; its task starts once it ended.
        await provider.simulate_transcription(session, "late words", "assistant", True)
        await channel.end_session(session)
    else:
        with pytest.raises(ConnectionError):
            await channel.start_session("r1", "u", "ws")
    for _ in range(20):  # the transcription's task, had it been started
        await asyncio.sleep(0)
    await kit.close()

    state = (
        channel._answer_depth,  # noqa: SLF001
        channel._transcription_order_locks,  # noqa: SLF001
        channel._last_assistant_text,  # noqa: SLF001
    )
    assert state == ({}, {}, {})


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
