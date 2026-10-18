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
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.postgres import PostgresStore
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
_QUIET_WINDOW = 0.1  # seconds
_QUIET_CPU = 0.001  # CPU seconds spent within the window that still count as quiet
_DRAIN_LIMIT = 30.0  # seconds


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
    await _drain()


async def _drain() -> None:
    """Return once the process has stopped spending CPU, bounded.

    The turns the last frames closed run to their end, and on a store with
    round trips a backlog of them can still be queued for its connections:
    a kit closed before they commit drops them, and their cost goes
    unmeasured. Waiting idle costs no CPU, so the figure is the same either way.
    """
    deadline = asyncio.get_running_loop().time() + _DRAIN_LIMIT
    while asyncio.get_running_loop().time() < deadline:
        before = _cpu_seconds()
        await asyncio.sleep(_QUIET_WINDOW)
        if _cpu_seconds() - before < _QUIET_CPU:
            return


async def _store(dsn: str | None) -> ConversationStore:
    """PostgresStore on *dsn*, its schema ensured; InMemoryStore without one."""
    if dsn is None:
        return InMemoryStore()
    store = PostgresStore(dsn=dsn)
    await store.init()
    return store


async def measure(
    calls: int,
    seconds: int,
    *,
    vad: bool = True,
    streaming_stt: bool = False,
    dsn: str | None = None,
) -> FrameCost:
    """CPU for *seconds* of audio on *calls* sessions; with the energy VAD,
    each second of tone is one turn per call. *dsn* puts the rooms in that
    Postgres instead of memory: the CPU is then this process's share of the
    store reads (driver and decoding), the database server's is not counted."""
    backend = MockVoiceBackend()
    stt = MockSTTProvider(streaming=streaming_stt)
    channel = VoiceChannel(
        "voice",
        stt=stt,
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=EnergyVADProvider() if vad else None),
    )
    kit = RoomKit(store=await _store(dsn))
    kit.register_channel(channel)
    sessions: list[VoiceSession] = []
    try:
        sessions = await _join_calls(kit, calls)
        frames = seconds * FRAMES_PER_SECOND
        start = _cpu_seconds()
        await _feed(backend, sessions, frames)
        cpu = _cpu_seconds() - start
    finally:
        if dsn is not None:  # a database outlives the run: leave it as found
            for session in sessions:
                await kit.store.delete_room(session.room_id)
        await kit.close()
    return FrameCost(calls, frames * calls, cpu, len(stt.languages_requested))
