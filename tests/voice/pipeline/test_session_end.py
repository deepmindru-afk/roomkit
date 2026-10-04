"""Audio work of a session whose end has begun leaves nothing behind (RMK-466).

``on_session_ended`` releases a stream's state in the engine and the stages,
but work for the session can still arrive after it: a frame in flight on a DSP
worker, the callbacks it sent to the loop, a frame a backend keeps delivering,
a last TTS chunk, a playback reference. None of it may reach the channel's
handlers or rebuild the state the end released. ``on_session_ending`` marks the
start of an end that still has awaits ahead of it.
"""

from __future__ import annotations

import asyncio
import threading

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline._ended_streams import EndedStreams
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.recorder.base import RecordingConfig
from roomkit.voice.pipeline.recorder.mock import MockAudioRecorder
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id=f"p-{sid}", channel_id="c1")


def _frame() -> AudioFrame:
    return AudioFrame(data=b"\x01\x00" * 160, sample_rate=16000, channels=1, sample_width=2)


def _speech_vad() -> MockVADProvider:
    return MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)] * 8)


def _heard(pipeline: AudioPipeline) -> list[str]:
    """What the channel's handlers would hear, by callback kind."""
    heard: list[str] = []
    pipeline.on_vad_event(lambda s, e: heard.append("vad"))
    pipeline.on_speech_frame(lambda s, f: heard.append("speech_frame"))
    pipeline.on_processed_frame(lambda s, f: heard.append("processed"))
    return heard


class _EndsMidChain(MockDenoiserProvider):
    """A stage during which the session's end lands, as it does on a DSP worker."""

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: AudioPipeline | None = None
        self.session: VoiceSession | None = None

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        if self.pipeline is not None and self.session is not None:
            self.pipeline.on_session_ended(self.session)
            self.pipeline = None
        return super().process(frame, stream)


class TestInbound:
    def test_a_frame_after_the_end_reaches_no_stage_and_no_listener(self) -> None:
        vad = _speech_vad()
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))
        heard = _heard(pipeline)
        session = _session()
        pipeline.on_session_active(session)
        pipeline.on_session_ended(session)

        pipeline.process_inbound(session, _frame())

        assert heard == []
        assert vad.frames == []
        assert pipeline._stage_streams == set()

    def test_a_frame_in_flight_across_the_end_leaves_no_state(self) -> None:
        vad = _speech_vad()
        denoiser = _EndsMidChain()
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, denoiser=denoiser))
        heard = _heard(pipeline)
        session = _session()
        pipeline.on_session_active(session)
        denoiser.pipeline, denoiser.session = pipeline, session

        # The end lands in the denoiser; the VAD after it still opens a
        # segment for the stream, which the frame's exit releases again.
        pipeline.process_inbound(session, _frame())

        assert len(vad.frames) == 1
        assert heard == []
        assert pipeline._stage_streams == set()
        assert pipeline._in_speech_sessions == set()
        assert session.id not in vad._indexes

    async def test_callbacks_a_worker_sent_home_before_the_end_are_dropped(self) -> None:
        pipeline = AudioPipeline(AudioPipelineConfig(vad=_speech_vad()))  # home: this loop
        heard = _heard(pipeline)
        session = _session()
        pipeline.on_session_active(session)

        # The worker's callbacks are queued on the loop, which does not run
        # them before the end lands: this coroutine has not yielded.
        worker = threading.Thread(target=pipeline.process_inbound, args=(session, _frame()))
        worker.start()
        worker.join()
        pipeline.on_session_ended(session)
        await asyncio.sleep(0.01)

        assert heard == []

    def test_a_returning_session_is_processed_again(self) -> None:
        vad = _speech_vad()
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))
        heard = _heard(pipeline)
        session = _session()
        pipeline.on_session_active(session)
        pipeline.on_session_ended(session)
        pipeline.on_session_active(session)

        pipeline.process_inbound(session, _frame())

        assert heard == ["vad", "speech_frame", "processed"]


class TestEnding:
    def test_ending_drops_inbound_and_keeps_the_rest_until_ended(self) -> None:
        recorder = MockAudioRecorder()
        stopped: list[str] = []
        pipeline = AudioPipeline(
            AudioPipelineConfig(
                vad=_speech_vad(), recorder=recorder, recording_config=RecordingConfig()
            )
        )
        pipeline.on_recording_stopped(lambda s, r: stopped.append(s.id))
        heard = _heard(pipeline)
        session = _session()
        pipeline.on_session_active(session)

        pipeline.on_session_ending(session)
        pipeline.process_inbound(session, _frame())
        pipeline.process_outbound(session, _frame())

        # The teardown's outbound tail is still recorded and processed; the
        # inbound audio of a session that is ending is not.
        assert heard == []
        assert recorder.inbound_frames == []
        assert len(recorder.outbound_frames) == 1
        assert session.id in pipeline._stage_streams

        pipeline.on_session_ended(session)

        assert stopped == [session.id]
        assert pipeline._stage_streams == set()


class TestOutbound:
    def test_a_frame_played_after_the_end_is_processed_and_leaves_nothing(self) -> None:
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))
        session = _session()
        pipeline.on_session_active(session)
        pipeline.on_session_ended(session)
        released = len(aec.reset_streams)

        frame = pipeline.process_outbound(session, _frame())

        assert frame.data == _frame().data
        assert aec.reference_streams == []  # no capture is left to cancel
        assert aec.reset_streams[released:] == [session.id]
        assert pipeline._stage_streams == set()

    def test_an_ended_streams_aec_reference_and_activity_are_ignored(self) -> None:
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))
        session = _session()
        pipeline.on_session_active(session)
        pipeline.on_session_ending(session)

        pipeline.feed_aec_reference(_frame(), session.id)
        pipeline.set_aec_active(session.id, True)

        assert aec.reference_streams == []
        assert aec.active_changes == []
        assert pipeline._aec_active_sources == {}


def test_the_oldest_ended_streams_are_forgotten_beyond_the_bound() -> None:
    ended = EndedStreams(kept=2)
    for stream in ("s0", "s1", "s2"):
        ended.mark(stream, released=True)

    assert "s0" not in ended
    assert "s1" in ended and "s2" in ended
    ended.mark("s1", released=False)  # marked again: the newest, not yet released
    ended.mark("s3", released=True)
    assert "s2" not in ended
    assert not ended.released("s1")
