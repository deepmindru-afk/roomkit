"""A session unbound mid-utterance leaves no speech state behind (RMK-466).

The VAD handlers keep per-session state between a SPEECH_START and its
SPEECH_END. A caller who hangs up while speaking never sends the SPEECH_END
that would have cleared it, so ``unbind_session`` drops it itself.
"""

from __future__ import annotations

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

_FRAME = b"\x01\x00" * 320


async def test_unbind_mid_utterance_forgets_the_speech_state() -> None:
    backend = MockVoiceBackend()
    kit = RoomKit(voice=backend)
    channel = VoiceChannel(
        "voice-1",
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(
            vad=MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        ),
    )
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.join(room.id, "voice-1", participant_id="user-1")

    await backend.simulate_audio_received(session, AudioFrame(data=_FRAME))
    assert session.id in channel._speech_started_at
    # What a SPEECH_START during playback adds under the DISABLED strategy:
    # the segment is suppressed and queued for after playback (RFC §12.6).
    with channel._state_lock:
        channel._suppressed_sessions.add(session.id)
        channel._queueing_sessions.add(session.id)
        channel._queued_speech[session.id] = [(_FRAME, None)]

    channel.unbind_session(session)

    assert session.id not in channel._speech_started_at
    assert session.id not in channel._suppressed_sessions
    assert session.id not in channel._queueing_sessions
    assert session.id not in channel._queued_speech
    await kit.close()
