"""A door that brings a session into a room reads the room with the caller's scope (RFC §17.2).

``kit.join()`` read the room unscoped and a realtime channel's
``start_session()`` did not read it at all: a host acting for one organization
that knew a room id of another could join a session to it, its audio entering
that room and its track declared to that room's recordings (RMK-475).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest

from roomkit import RoomKit, VideoChannel, VoiceChannel
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.framework_event import FrameworkEvent
from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import (
    MockRealtimeAudioVideoProvider,
    MockRealtimeProvider,
    MockRealtimeTransport,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

Door = Callable[[str | None], Awaitable[object]]


@dataclass
class Bench:
    """Tenant A's room ``r1``, recording, with one channel attached."""

    kit: RoomKit
    recorder: MockMediaRecorder
    door: Door = field(init=False)
    started: list[FrameworkEvent] = field(default_factory=list)

    def sessions_started(self) -> set[str]:
        """The sessions announced in ``r1`` (a door may announce one twice:
        as voice and as video, or from the bind and from the join)."""
        return {e.data["session_id"] for e in self.started if e.room_id == "r1"}


async def _bench(channel: object) -> Bench:
    kit = RoomKit()
    recorder = MockMediaRecorder()
    bench = Bench(kit, recorder)

    @kit.on("voice_session_started")
    async def on_voice(event: FrameworkEvent) -> None:
        bench.started.append(event)

    @kit.on("video_session_started")
    async def on_video(event: FrameworkEvent) -> None:
        bench.started.append(event)

    kit.register_channel(channel)  # ty: ignore[invalid-argument-type]
    await kit.create_room(
        room_id="r1",
        organization_id="tenant-a",
        recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())],
    )
    await kit.attach_channel("r1", channel.channel_id)  # ty: ignore[unresolved-attribute]
    return bench


def _voice_channel() -> VoiceChannel:
    return VoiceChannel(
        "voice", stt=MockSTTProvider(), tts=MockTTSProvider(), backend=MockVoiceBackend()
    )


def _realtime(cls: type[RealtimeVoiceChannel], provider: object) -> RealtimeVoiceChannel:
    return cls(
        "rt",
        provider=provider,  # ty: ignore[invalid-argument-type]
        transport=MockRealtimeTransport(),
        recording=ChannelRecordingConfig(audio=True),
    )


async def _join_voice_pull() -> Bench:
    bench = await _bench(_voice_channel())
    bench.door = lambda org: bench.kit.join("r1", "voice", organization_id=org)
    return bench


async def _join_voice_push() -> Bench:
    """The SIP shape: the backend accepted the call, the host binds it."""
    bench = await _bench(_voice_channel())

    def door(org: str | None) -> Awaitable[object]:
        session = VoiceSession(
            id=f"sip-{len(bench.started)}-{org}",
            room_id="r1",
            participant_id="caller",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )
        return bench.kit.join("r1", "voice", session=session, organization_id=org)

    bench.door = door
    return bench


async def _join_video() -> Bench:
    bench = await _bench(VideoChannel("video", backend=MockVideoBackend()))
    bench.door = lambda org: bench.kit.join("r1", "video", organization_id=org)
    return bench


async def _join_realtime() -> Bench:
    bench = await _bench(_realtime(RealtimeVoiceChannel, MockRealtimeProvider()))
    bench.door = lambda org: bench.kit.join("r1", "rt", connection="ws", organization_id=org)
    return bench


async def _start_realtime() -> Bench:
    channel = _realtime(RealtimeVoiceChannel, MockRealtimeProvider())
    bench = await _bench(channel)
    bench.door = lambda org: channel.start_session("r1", "user", "ws", organization_id=org)
    return bench


async def _start_realtime_av() -> Bench:
    channel = _realtime(RealtimeAudioVideoChannel, MockRealtimeAudioVideoProvider())
    bench = await _bench(channel)
    bench.door = lambda org: channel.start_session("r1", "user", "ws", organization_id=org)
    return bench


DOORS = pytest.mark.parametrize(
    "make",
    [
        _join_voice_pull,
        _join_voice_push,
        _join_video,
        _join_realtime,
        _start_realtime,
        _start_realtime_av,
    ],
    ids=[
        "join-voice-pull",
        "join-voice-push",
        "join-video",
        "join-realtime",
        "start-realtime",
        "start-realtime-av",
    ],
)


async def _settle() -> None:
    """Let the session-started events a bind schedules run."""
    await asyncio.sleep(0.05)


@DOORS
async def test_another_organizations_room_is_not_found_and_nothing_joins_it(
    make: Callable[[], Awaitable[Bench]],
) -> None:
    bench = await make()

    with pytest.raises(RoomNotFoundError):
        await bench.door("tenant-b")
    await _settle()

    assert bench.sessions_started() == set()
    assert bench.recorder.tracks == []
    await bench.kit.close()


@DOORS
@pytest.mark.parametrize("org", ["tenant-a", None], ids=["same-organization", "unscoped"])
async def test_the_rooms_organization_and_an_unscoped_call_join_as_before(
    make: Callable[[], Awaitable[Bench]], org: str | None
) -> None:
    bench = await make()

    await bench.door(org)
    await _settle()

    assert len(bench.sessions_started()) == 1
    assert len(bench.recorder.tracks) == 1
    await bench.kit.close()


async def test_a_scoped_start_on_a_channel_no_framework_registered_is_not_found() -> None:
    """With no framework there is no room to read, so the scope cannot hold."""
    channel = _realtime(RealtimeVoiceChannel, MockRealtimeProvider())

    with pytest.raises(RoomNotFoundError):
        await channel.start_session("r1", "user", "ws", organization_id="tenant-a")

    assert channel.get_room_sessions("r1") == []
    await channel.close()
