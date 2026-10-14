"""A room's recording holds its own sessions' media, never another room's (RMK-758).

The recording taps sit on the channel's whole pipeline, which every session of
every room on that channel goes through: each tap takes the frames of the
session it was wired for, and only those.
"""

from __future__ import annotations

import asyncio

from roomkit import RoomKit, VoiceChannel
from roomkit.recorder.base import MediaRecordingConfig, RoomRecorderBinding
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.tts.mock import MockTTSProvider


async def _two_recorded_rooms() -> tuple[
    RoomKit, VoiceChannel, MockVoiceBackend, dict[str, MockMediaRecorder], dict[str, object]
]:
    kit = RoomKit()
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice", tts=MockTTSProvider(), backend=backend, pipeline=AudioPipelineConfig()
    )
    kit.register_channel(channel)
    recorders: dict[str, MockMediaRecorder] = {}
    sessions: dict[str, object] = {}
    for room_id in ("room-a", "room-b"):
        recorders[room_id] = MockMediaRecorder()
        binding = RoomRecorderBinding(recorder=recorders[room_id], config=MediaRecordingConfig())
        await kit.create_room(room_id=room_id, recorders=[binding])
        await kit.attach_channel(room_id, "voice")
        sessions[room_id] = await kit.join(room_id, "voice", participant_id=f"p-{room_id}")
    return kit, channel, backend, recorders, sessions


def _recorded_tracks(recorder: MockMediaRecorder) -> set[str]:
    return {chunk.track_id for chunk in recorder.chunks}


def _frame(value: int) -> AudioFrame:
    return AudioFrame(data=value.to_bytes(2, "little", signed=True) * 160, sample_rate=16000)


async def test_a_sessions_audio_is_recorded_in_its_own_room_only() -> None:
    kit, _, backend, recorders, sessions = await _two_recorded_rooms()

    await backend.simulate_audio_received(sessions["room-b"], _frame(1000))
    await asyncio.sleep(0.05)

    assert recorders["room-a"].chunks == []
    assert len(_recorded_tracks(recorders["room-b"])) == 1
    await kit.close()


async def test_a_session_writes_only_its_own_track() -> None:
    kit, _, backend, recorders, sessions = await _two_recorded_rooms()

    await backend.simulate_audio_received(sessions["room-a"], _frame(1000))
    await backend.simulate_audio_received(sessions["room-b"], _frame(2000))
    await asyncio.sleep(0.05)

    for room_id, recorder in recorders.items():
        assert {chunk.data for chunk in recorder.chunks} == {
            _frame(1000 if room_id == "room-a" else 2000).data
        }
    await kit.close()


async def test_speech_to_one_room_is_not_mixed_into_another_rooms_recording() -> None:
    kit, channel, backend, recorders, sessions = await _two_recorded_rooms()

    await channel.say(sessions["room-b"], "Hello room B")  # type: ignore[arg-type]
    await backend.simulate_audio_received(sessions["room-a"], _frame(0))
    await asyncio.sleep(0.05)

    assert {chunk.data for chunk in recorders["room-a"].chunks} == {_frame(0).data}
    await kit.close()
