"""``pipeline_speakers``: the diarization stage names a transcript's speaker (RFC §12.2.3).

When the STT labels nobody, a VoiceChannel with ``pipeline_speakers=True``
gives each transcript the speaker the pipeline's DiarizationProvider heard the
longest over it. Opt-in: without it nothing changes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels._voice_speakers import PipelineSpeakerTally
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.enums import EventType
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, SpeakerSegment, TranscriptionResult
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.diarization.base import DiarizationProvider, DiarizationResult
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.mock import MockSTTProvider


def _heard(speaker: str | None) -> DiarizationResult | None:
    if speaker is None:
        return None
    return DiarizationResult(speaker_id=speaker, confidence=0.9, is_new_speaker=False)


def _utterance() -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        None,
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"fake-audio"),
    ]


class _Room:
    def __init__(self, channel: VoiceChannel, backend: MockVoiceBackend, stt: Any) -> None:
        self.channel, self.backend = channel, backend
        self.kit = RoomKit(stt=stt, voice=backend)
        self.kit.register_channel(channel)
        self.rename: dict[str, str] = {}
        self.transcriptions: list[Any] = []

        @self.kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event: Any, ctx: Any) -> HookResult:
            self.transcriptions.append(event)
            if event.speaker in self.rename:
                return HookResult.modify(
                    dataclasses.replace(event, sender_name=self.rename[event.speaker])
                )
            return HookResult.allow()

    async def start(self) -> None:
        room = await self.kit.create_room()
        self.room_id = room.id
        await self.kit.attach_channel(room.id, "voice-1")
        self.session = await self.kit.join(room.id, "voice-1", participant_id="owner")

    async def frames(self, count: int) -> None:
        for _ in range(count):
            await self.backend.simulate_audio_received(
                self.session, AudioFrame(data=b"\x01\x00" * 1600)
            )

    async def messages(self, count: int) -> list[Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 3
        while True:
            events = await self.kit.store.list_events(self.room_id, offset=0, limit=50)
            found = [
                e
                for e in events
                if e.type == EventType.MESSAGE and e.source.channel_id == "voice-1"
            ]
            if len(found) >= count or loop.time() > deadline:
                return found
            await asyncio.sleep(0.02)


def _vad_room(speakers: list[str | None], utterances: int, **channel: Any) -> _Room:
    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=[f"Phrase {i + 1}." for i in range(utterances)])
    pipeline = AudioPipelineConfig(
        vad=MockVADProvider(events=_utterance() * utterances),
        diarization=MockDiarizationProvider(results=[_heard(s) for s in speakers]),
    )
    voice = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline, **channel)
    return _Room(voice, backend, stt)


class TestTally:
    def _frame(self, speaker: str | None, seconds: float = 0.1) -> AudioFrame:
        frame = AudioFrame(data=b"\x00\x00" * int(16000 * seconds))
        if speaker is not None:
            frame.metadata["diarization"] = {"speaker_id": speaker, "confidence": 0.9}
        return frame

    def test_the_speaker_heard_longest_wins_then_the_slate_is_clean(self) -> None:
        tally = PipelineSpeakerTally()
        for frame in (self._frame("speaker_0"), self._frame("speaker_1", 0.3), self._frame(None)):
            tally.add("s", frame)
        taken = tally.take("s")
        assert taken is not None
        assert (taken.label, taken.epoch, taken.sender_name, taken.source) == (
            "1",
            0,
            "Speaker 1",
            "pipeline",
        )
        assert tally.take("s") is None

    def test_frames_the_stage_did_not_identify_count_for_nobody(self) -> None:
        tally = PipelineSpeakerTally()
        tally.add("s", self._frame(None))
        assert tally.take("s") is None

    def test_a_voice_matched_to_nobody_is_an_unknown_speaker(self) -> None:
        # sherpa-onnx says "unknown" below its match threshold: someone spoke,
        # and it was not the stream owner as far as the stage can tell.
        tally = PipelineSpeakerTally()
        tally.add("s", self._frame("speaker_0", 0.1))
        tally.add("s", self._frame("unknown", 0.3))
        taken = tally.take("s")
        assert taken is not None
        assert (taken.label, taken.sender_name) == (None, "Unknown speaker")

    def test_a_claim_is_answered_by_the_closing_frame_it_counts(self) -> None:
        tally = PipelineSpeakerTally()
        tally.add("s", self._frame(None))
        claim = tally.claim("s")
        assert not claim.done()
        closing = self._frame("speaker_1")
        closing.metadata["vad_speech_end"] = True
        tally.add("s", closing)
        answer = claim.result(timeout=0)
        assert answer is not None and answer.label == "1"
        assert tally.take("s") is None  # the count closed with the utterance

    def test_reset_answers_an_open_claim_with_nobody(self) -> None:
        tally = PipelineSpeakerTally()
        tally.add("s", self._frame("speaker_0"))
        claim = tally.claim("s")
        tally.reset("s")
        assert claim.result(timeout=0) is None


class TestVADMode:
    async def test_each_utterance_takes_its_speaker(self) -> None:
        room = _vad_room(["speaker_0"] * 3 + ["speaker_1"] * 3, 2, pipeline_speakers=True)
        await room.start()

        await room.frames(6)
        messages = await room.messages(2)
        await room.channel.close()

        assert [(m.metadata.get("sender_name"), m.content.body) for m in messages] == [
            ("Speaker 0", "Phrase 1."),
            ("Speaker 1", "Phrase 2."),
        ]
        assert [m.metadata.get("speaker_epoch") for m in messages] == [0, 0]
        assert {m.source.participant_id for m in messages} == {"owner"}

    async def test_the_dominant_speaker_names_a_mixed_utterance(self) -> None:
        # Every frame of the utterance counts, the SPEECH_END one included:
        # speaker_0 three times against speaker_1 twice.
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["Tu viens? Oui."])
        events = [
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            None,
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"fake-audio"),
        ]
        heard = ("speaker_1", "speaker_0", "speaker_0", "speaker_0", "speaker_1")
        pipeline = AudioPipelineConfig(
            vad=MockVADProvider(events=events),
            diarization=MockDiarizationProvider(results=[_heard(s) for s in heard]),
        )
        voice = VoiceChannel(
            "voice-1", stt=stt, backend=backend, pipeline=pipeline, pipeline_speakers=True
        )
        room = _Room(voice, backend, stt)
        await room.start()
        await room.frames(5)
        [message] = await room.messages(1)
        await room.channel.close()
        assert message.metadata["speaker_label"] == "0"

    async def test_a_hook_names_the_voice(self) -> None:
        room = _vad_room(["speaker_0"] * 3, 1, pipeline_speakers=True)
        room.rename = {"0": "Sylvain"}
        await room.start()
        await room.frames(3)
        [message] = await room.messages(1)
        await room.channel.close()
        assert message.metadata["sender_name"] == "Sylvain"
        assert room.transcriptions[0].speaker == "0"

    async def test_without_the_option_nothing_changes(self) -> None:
        room = _vad_room(["speaker_0"] * 3, 1)
        await room.start()
        await room.frames(3)
        [message] = await room.messages(1)
        await room.channel.close()
        assert "sender_name" not in message.metadata
        assert "speaker_label" not in message.metadata


class TestSpeechHeldDuringPlayback:
    async def test_an_utterance_replayed_after_playback_keeps_its_speaker(self) -> None:
        # DISABLED queues speech heard during playback and replays it once the
        # bot is done: the stage counted it when it was said, not at replay.
        room = _vad_room(
            ["speaker_1"] * 3,
            1,
            pipeline_speakers=True,
            interruption=InterruptionConfig(strategy=InterruptionStrategy.DISABLED),
        )
        await room.start()
        room.channel._playing_sessions[room.session.id] = TTSPlaybackState(
            session_id=room.session.id, text="a long answer"
        )
        await room.frames(3)
        assert len(room.channel._queued_speech[room.session.id]) == 1

        room.channel._playing_sessions.pop(room.session.id)
        await room.channel._flush_queued_speech(room.session.id)
        [message] = await room.messages(1)
        await room.channel.close()
        assert message.metadata["speaker_label"] == "1"


class _IdentifiesAtSpeechEnd(DiarizationProvider):
    """Like sherpa-onnx's stage: one identification, on the utterance's last frame.

    The pipeline fires SPEECH_END callbacks before its diarization stage sees
    that frame, so this is the case a count taken at SPEECH_END would miss.
    """

    def __init__(self, speakers: list[str], *, delay_s: float = 0.0) -> None:
        self._speakers = list(speakers)
        self._delay_s = delay_s

    @property
    def name(self) -> str:
        return "IdentifiesAtSpeechEnd"

    def process(self, frame: AudioFrame, stream: str) -> DiarizationResult | None:
        if not frame.metadata.get("vad_speech_end") or not self._speakers:
            return None
        time.sleep(self._delay_s)  # an embedding takes a while, off the loop
        return _heard(self._speakers.pop(0))


class TestAStageThatIdentifiesAtSpeechEnd:
    def _room(self, *, threads: int | None, delay_s: float) -> _Room:
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["Bonjour.", "Salut."])
        pipeline = AudioPipelineConfig(
            vad=MockVADProvider(events=_utterance() * 2),
            diarization=_IdentifiesAtSpeechEnd(["speaker_0", "speaker_1"], delay_s=delay_s),
            inbound_dsp_threads=threads,
        )
        voice = VoiceChannel(
            "voice-1", stt=stt, backend=backend, pipeline=pipeline, pipeline_speakers=True
        )
        return _Room(voice, backend, stt)

    @pytest.mark.parametrize(
        ("threads", "delay_s"), [(None, 0.0), (1, 0.2)], ids=["inline", "dsp-thread"]
    )
    async def test_the_closing_frames_speaker_names_the_utterance(
        self, threads: int | None, delay_s: float
    ) -> None:
        # On a DSP thread the speech-end task reaches the speaker while the
        # stage is still identifying: it waits for the answer.
        room = self._room(threads=threads, delay_s=delay_s)
        await room.start()
        await room.frames(6)
        messages = await room.messages(2)
        await room.channel.close()
        assert [m.metadata.get("speaker_label") for m in messages] == ["0", "1"]


class _OneFinalPerChunk(STTProvider):
    """A continuous STT, diarizing or not, answering one scripted final per chunk."""

    def __init__(self, finals: list[TranscriptionResult], *, diarizing: bool = False) -> None:
        self._finals = list(finals)
        self._diarizing = diarizing

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_diarization(self) -> bool:
        return self._diarizing

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for chunk in audio_stream:
            if any(chunk.data) and self._finals:
                yield self._finals.pop(0)


def _continuous_room(stt: STTProvider, speakers: list[str | None]) -> _Room:
    backend = MockVoiceBackend()
    pipeline = AudioPipelineConfig(
        diarization=MockDiarizationProvider(results=[_heard(s) for s in speakers])
    )
    voice = VoiceChannel(
        "voice-1", stt=stt, backend=backend, pipeline=pipeline, pipeline_speakers=True
    )
    return _Room(voice, backend, stt)


class TestContinuousMode:
    async def test_a_plain_stt_takes_the_stages_speaker(self) -> None:
        stt = _OneFinalPerChunk([TranscriptionResult(text="Bonjour.")])
        room = _continuous_room(stt, ["speaker_0"])
        await room.start()
        await room.frames(1)
        [message] = await room.messages(1)
        await room.channel.close()
        assert message.metadata["sender_name"] == "Speaker 0"

    async def test_a_diarizing_stts_label_wins(self) -> None:
        stt = _OneFinalPerChunk(
            [TranscriptionResult(text="Bonjour.", segments=[SpeakerSegment("A", "Bonjour.")])],
            diarizing=True,
        )
        room = _continuous_room(stt, ["speaker_0"])
        await room.start()
        await room.frames(1)
        [message] = await room.messages(1)
        await room.channel.close()
        assert message.metadata["speaker_label"] == "A"


class TestConfiguration:
    def test_batch_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="batch_mode"):
            VoiceChannel(
                "voice-1",
                stt=MockSTTProvider(),
                backend=MockVoiceBackend(),
                pipeline=AudioPipelineConfig(diarization=MockDiarizationProvider()),
                batch_mode=True,
                pipeline_speakers=True,
            )

    def test_the_option_needs_a_diarization_stage(self) -> None:
        with pytest.raises(ValueError, match="DiarizationProvider"):
            VoiceChannel(
                "voice-1",
                stt=MockSTTProvider(),
                backend=MockVoiceBackend(),
                pipeline=AudioPipelineConfig(vad=MockVADProvider(events=[])),
                pipeline_speakers=True,
            )
