"""CPU RoomKit spends per inbound 20 ms voice frame, the transport excluded.

Frames enter where a voice backend hands them over (its audio callback), for
N sessions in one room each, so the cost covers the pipeline, the VAD, the
STT hand-off and the inbound path of each turn, but no RTP and no model.
"""

from __future__ import annotations

import asyncio
import math
import resource
import struct
from dataclasses import dataclass

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

SAMPLE_RATE = 8000  # a G.711 call
FRAMES_PER_SECOND = 50  # 20 ms frames
_SAMPLES = SAMPLE_RATE // FRAMES_PER_SECOND
TONE = struct.pack(
    f"<{_SAMPLES}h",
    *(int(8000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)) for i in range(_SAMPLES)),
)
SILENCE = bytes(2 * _SAMPLES)


@dataclass(frozen=True)
class FrameCost:
    """One measurement: the CPU the frames took and what reached the STT."""

    calls: int
    frames: int
    cpu_seconds: float
    stt_requests: int  # batch transcriptions plus streams opened

    @property
    def us_per_frame(self) -> float:
        return 1e6 * self.cpu_seconds / self.frames

    @property
    def calls_per_core(self) -> float:
        """Calls one core carries for this layer alone, RTP and models excluded."""
        return 1e6 / (FRAMES_PER_SECOND * self.us_per_frame)


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


async def _join_calls(kit: RoomKit, calls: int) -> list[VoiceSession]:
    sessions = []
    for i in range(calls):
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice")
        sessions.append(await kit.join(room.id, "voice", participant_id=f"caller-{i}"))
    return sessions


async def _feed(backend: MockVoiceBackend, sessions: list[VoiceSession], frames: int) -> None:
    """One frame per session per tick: 1 s of tone, then 1 s of silence."""
    for tick in range(frames):
        data = TONE if (tick // FRAMES_PER_SECOND) % 2 == 0 else SILENCE
        for session in sessions:
            frame = AudioFrame(data=data, sample_rate=SAMPLE_RATE)
            await backend.simulate_audio_received(session, frame)
        await asyncio.sleep(0)
    for _ in range(200):  # the turns the last frames closed run to their end
        await asyncio.sleep(0)


async def measure(
    calls: int, seconds: int, *, vad: bool = True, streaming_stt: bool = False
) -> FrameCost:
    """CPU for *seconds* of audio on *calls* sessions; with the energy VAD,
    each second of tone is one turn per call."""
    backend = MockVoiceBackend()
    stt = MockSTTProvider(streaming=streaming_stt)
    channel = VoiceChannel(
        "voice",
        stt=stt,
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=EnergyVADProvider() if vad else None),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    try:
        sessions = await _join_calls(kit, calls)
        frames = seconds * FRAMES_PER_SECOND
        start = _cpu_seconds()
        await _feed(backend, sessions, frames)
        cpu = _cpu_seconds() - start
    finally:
        await kit.close()
    return FrameCost(calls, frames * calls, cpu, len(stt.languages_requested))
