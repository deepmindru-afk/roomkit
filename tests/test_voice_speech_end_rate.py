"""Speech audio is labelled with the pipeline's rate, not the transport's.

A backend states its transport rate in ``session.metadata["input_sample_rate"]``
(FastRTC 48 kHz, SIP the codec rate). When the pipeline resamples inbound audio
to its internal format, every speech segment and every pre-roll it hands out is
at the internal rate. Labelling them with the transport rate announced 16 kHz
audio as 48 kHz to the batch STT, the turn detector and the TTS context.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import RoomKit, TTSContext, TTSContextConfig, TTSContextLevel, VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.config import AudioFormat, AudioPipelineContract
from roomkit.voice.pipeline.turn.mock import MockTurnDetector
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider

_SEGMENT = b"\x01\x00" * 1600  # 100 ms at 16 kHz


class _ContextTTS(TTSProvider):
    """A TTS that keeps audio context and records what it was given."""

    def __init__(self) -> None:
        self.contexts: list[TTSContext | None] = []

    @property
    def context_level(self) -> TTSContextLevel:
        return TTSContextLevel.AUDIO

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.contexts.append(context)
        yield AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)


def _contract(internal_rate: int) -> AudioPipelineContract:
    return AudioPipelineContract(
        transport_inbound_format=AudioFormat(sample_rate=48000),
        transport_outbound_format=AudioFormat(sample_rate=48000),
        internal_format=AudioFormat(sample_rate=internal_rate),
    )


async def _room(
    stt: MockSTTProvider, pipeline: AudioPipelineConfig, tts: TTSProvider | None = None
) -> tuple[RoomKit, VoiceChannel, VoiceSession]:
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
        tts_context=TTSContextConfig(include_audio=True) if tts is not None else None,
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    session.metadata["input_sample_rate"] = 48000
    return kit, channel, session


async def test_batch_stt_turn_detector_and_tts_context_get_the_internal_rate() -> None:
    stt = MockSTTProvider(transcripts=["hello there"])
    detector = MockTurnDetector()
    tts = _ContextTTS()
    kit, channel, session = await _room(
        stt,
        AudioPipelineConfig(contract=_contract(16000), turn_detector=detector),
        tts,
    )

    await channel._process_speech_end(session, _SEGMENT, "r1", None)  # noqa: SLF001
    await channel.say(session, "noted")

    [frame] = stt.calls
    assert frame.sample_rate == 16000
    [turn_ctx] = detector.evaluations
    assert turn_ctx.audio_sample_rate == 16000
    [context] = tts.contexts
    assert context is not None
    [user_turn] = [t for t in context.turns if t.role == "user"]
    assert user_turn.audio is not None
    assert user_turn.audio.sample_rate == 16000
    await kit.close()


async def test_stream_pre_roll_gets_the_internal_rate() -> None:
    stt = MockSTTProvider(transcripts=["hello"], streaming=True)
    kit, channel, session = await _room(stt, AudioPipelineConfig(contract=_contract(16000)))

    channel._start_stt_stream(session, "r1", pre_roll=_SEGMENT)  # noqa: SLF001
    state = channel._stt_streams[session.id]  # noqa: SLF001
    state.queue.put_nowait(None)
    assert state.task is not None
    await asyncio.wait_for(state.task, timeout=1.0)

    [pre_roll] = stt.calls
    assert pre_roll.sample_rate == 16000
    await kit.close()


async def test_without_resampling_the_transport_rate_stands() -> None:
    stt = MockSTTProvider(transcripts=["hello"])
    kit, channel, session = await _room(stt, AudioPipelineConfig())

    await channel._process_speech_end(session, _SEGMENT, "r1", None)  # noqa: SLF001

    [frame] = stt.calls
    assert frame.sample_rate == 48000
    await kit.close()
