"""ON_RECORDING_STOPPED fires when a voice session leaves (RMK-353).

The pipeline stops a session's recording when the session ends, from inside
``VoiceChannel.unbind_session``. The channel used to drop the session's
binding first, so the recording-stopped callback found no room to report to
and the hook never fired: every recording opened on a voice channel ended in
silence, while ON_RECORDING_STARTED had announced it.
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.events import RecordingStoppedEvent
from roomkit.voice.pipeline import AudioPipelineConfig, MockAudioRecorder, RecordingConfig


async def _join_recorded_session(kit: RoomKit, recorder: MockAudioRecorder) -> tuple[str, object]:
    channel = VoiceChannel(
        "voice-1",
        backend=MockVoiceBackend(),
        pipeline=AudioPipelineConfig(recorder=recorder, recording_config=RecordingConfig()),
    )
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.join(room.id, "voice-1", participant_id="caller")
    return room.id, session


class TestRecordingStoppedOnLeave:
    async def test_leaving_fires_the_hook_with_the_recording(self) -> None:
        kit = RoomKit()
        recorder = MockAudioRecorder()
        stopped: list[tuple[str, RecordingStoppedEvent]] = []

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def on_stopped(event: RecordingStoppedEvent, ctx: object) -> None:
            stopped.append((ctx.room.id, event))  # type: ignore[attr-defined]

        room_id, session = await _join_recorded_session(kit, recorder)
        assert len(recorder.started) == 1

        await kit.leave(session)  # type: ignore[arg-type]
        await asyncio.sleep(0.05)

        assert len(recorder.stopped) == 1
        assert len(stopped) == 1
        hook_room, event = stopped[0]
        assert hook_room == room_id
        assert event.id == recorder.stopped[0].id
        await kit.close()

    async def test_the_hook_fires_once_when_leave_is_repeated(self) -> None:
        kit = RoomKit()
        recorder = MockAudioRecorder()
        stopped: list[RecordingStoppedEvent] = []

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def on_stopped(event: RecordingStoppedEvent, ctx: object) -> None:
            stopped.append(event)

        _, session = await _join_recorded_session(kit, recorder)
        await kit.leave(session)  # type: ignore[arg-type]
        await kit.leave(session)  # type: ignore[arg-type]
        await asyncio.sleep(0.05)

        assert len(stopped) == 1
        await kit.close()
