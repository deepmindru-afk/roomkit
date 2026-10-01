"""A realtime voice session's recording is announced like a voice session's (RFC §17.6).

``RealtimeVoiceChannel`` records through the same audio pipeline as
``VoiceChannel`` but never subscribed to its recording callbacks: a recorder
configured on a speech-to-speech session captured the call without
``ON_RECORDING_STARTED`` ever firing, so a host had no hook to notify the
participants, and the stop went unreported too (RMK-355).
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSession
from roomkit.voice.events import RecordingStartedEvent, RecordingStoppedEvent
from roomkit.voice.pipeline import AudioPipelineConfig, MockAudioRecorder, RecordingConfig
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


def _recorded_channel(recorder: MockAudioRecorder) -> RealtimeVoiceChannel:
    return RealtimeVoiceChannel(
        "rt-1",
        provider=MockRealtimeProvider(),
        transport=MockRealtimeTransport(),
        input_sample_rate=24000,
        output_sample_rate=24000,
        pipeline=AudioPipelineConfig(recorder=recorder, recording_config=RecordingConfig()),
    )


async def _start(kit: RoomKit, channel: RealtimeVoiceChannel) -> VoiceSession:
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt-1")
    return await channel.start_session("r1", "user-1", "fake-ws")


class TestRealtimeRecordingHooks:
    async def test_starting_a_session_announces_its_recording(self) -> None:
        kit = RoomKit()
        recorder = MockAudioRecorder()
        started: list[tuple[str, RecordingStartedEvent]] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def on_started(event: RecordingStartedEvent, ctx: object) -> None:
            started.append((ctx.room.id, event))  # type: ignore[attr-defined]

        session = await _start(kit, _recorded_channel(recorder))
        await asyncio.sleep(0.05)

        assert len(recorder.started) == 1
        assert len(started) == 1
        room_id, event = started[0]
        assert room_id == "r1"
        assert event.room_id == "r1"
        assert event.session.id == session.id
        assert recorder.started[0][0] == session.id
        await kit.close()

    async def test_ending_a_session_reports_the_recording(self) -> None:
        kit = RoomKit()
        recorder = MockAudioRecorder()
        stopped: list[tuple[str, RecordingStoppedEvent]] = []

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def on_stopped(event: RecordingStoppedEvent, ctx: object) -> None:
            stopped.append((ctx.room.id, event))  # type: ignore[attr-defined]

        channel = _recorded_channel(recorder)
        session = await _start(kit, channel)
        await channel.end_session(session)
        await asyncio.sleep(0.05)

        assert len(recorder.stopped) == 1
        assert len(stopped) == 1
        room_id, event = stopped[0]
        assert room_id == "r1"
        assert event.id == recorder.stopped[0].id
        await kit.close()

    async def test_no_recorder_announces_nothing(self) -> None:
        kit = RoomKit()
        fired: list[object] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def on_started(event: RecordingStartedEvent, ctx: object) -> None:
            fired.append(event)

        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            pipeline=AudioPipelineConfig(),
        )
        session = await _start(kit, channel)
        await channel.end_session(session)
        await asyncio.sleep(0.05)

        assert fired == []
        await kit.close()
