"""A session unbound from a VoiceChannel leaves no per-session state behind (RMK-466).

Two ways back in, both closed:

- the VAD handlers keep state between a SPEECH_START and its SPEECH_END, and a
  caller who hangs up while speaking never sends the SPEECH_END that would
  have cleared it;
- the doors that do not go through the pipeline (the frame-rate limiter, the
  output level hooks, an out-of-band DTMF) still see audio and events for the
  session after ``unbind_session``, until the backend disconnects.
"""

from __future__ import annotations

from typing import Any

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.dtmf.base import DTMFEvent, DTMFRedaction
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSContextLevel
from roomkit.voice.tts.mock import MockTTSProvider

_FRAME = b"\x01\x00" * 320


async def _bound(**channel_kwargs: Any) -> tuple[RoomKit, VoiceChannel, Any, VoiceSession]:
    """A VoiceChannel with one bound session, its pipeline's VAD hearing speech start."""
    backend = MockVoiceBackend()
    kit = RoomKit(voice=backend)
    pipeline = AudioPipelineConfig(
        vad=MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)]),
        dtmf_redaction=DTMFRedaction(),
    )
    channel = VoiceChannel(
        "voice-1",
        stt=MockSTTProvider(),
        tts=MockTTSProvider(context_level=TTSContextLevel.TEXT),
        backend=backend,
        pipeline=pipeline,
        **channel_kwargs,
    )
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.join(room.id, "voice-1", participant_id="user-1")
    return kit, channel, backend, session


async def test_unbind_mid_utterance_forgets_the_speech_state() -> None:
    kit, channel, backend, session = await _bound()
    await backend.simulate_audio_received(session, AudioFrame(data=_FRAME))
    assert session.id in channel._speech_started_at
    # What a SPEECH_START during playback adds: an energy barge-in count, and
    # under the DISABLED strategy a suppressed segment queued for after
    # playback (RFC §12.6).
    with channel._state_lock:
        channel._barge_in_energy_count[session.id] = 2
        channel._suppressed_sessions.add(session.id)
        channel._queueing_sessions.add(session.id)
        channel._queued_speech[session.id] = [(_FRAME, None)]

    channel.unbind_session(session)

    assert session.id not in channel._speech_started_at
    assert session.id not in channel._barge_in_energy_count
    assert session.id not in channel._suppressed_sessions
    assert session.id not in channel._queueing_sessions
    assert session.id not in channel._queued_speech
    await kit.close()


async def test_audio_and_events_after_unbind_rebuild_nothing() -> None:
    kit, channel, backend, session = await _bound(max_audio_frames_per_second=50)
    channel.unbind_session(session)

    # The backend is still connected: it delivers a frame, reports playback,
    # and a TTS chunk still in flight reaches the output level hook.
    await backend.simulate_audio_received(session, AudioFrame(data=_FRAME))
    channel._on_audio_played_for_level(session, AudioFrame(data=_FRAME))
    channel._fire_output_level(session, _FRAME)
    channel._on_pipeline_dtmf(session, DTMFEvent(digit="5", duration_ms=80.0))

    assert session.id not in channel._frame_counts
    assert session.id not in channel._last_output_level_at
    assert channel._tts_context is not None
    assert not channel._tts_context.take_dtmf(session.id)
    await kit.close()
