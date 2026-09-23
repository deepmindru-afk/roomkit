"""Barge-in hooks and the interruption policy measure the audio the user heard.

``TTSPlaybackState.position_ms`` counts from the state's creation, synthesis
latency included, while the timeline and the TTS context record ``played_ms``,
the audio that actually went out. The hooks reported the first, so one cut had
two positions; and ``allow_during_first_ms`` measured latency too, leaving a
slow-to-start response interruptible before it made a sound.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.tts.base import TTSProvider

_RATE = 16000
_ONE_SECOND = b"\x00\x00" * _RATE


class _SlowStartTTS(TTSProvider):
    """Synthesis that takes ``latency_s`` before its first second of audio."""

    def __init__(self, latency_s: float) -> None:
        self._latency_s = latency_s

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        await asyncio.sleep(self._latency_s)
        yield AudioChunk(data=_ONE_SECOND, sample_rate=_RATE)
        await asyncio.sleep(1)


async def _room(
    tts: TTSProvider | None, interruption: InterruptionConfig | None = None
) -> tuple[RoomKit, VoiceChannel, VoiceSession]:
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel("voice-1", tts=tts, backend=backend, interruption=interruption)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    channel.bind_session(
        session,
        "r1",
        ChannelBinding(room_id="r1", channel_id="voice-1", channel_type=ChannelType.VOICE),
    )
    return kit, channel, session


async def test_cancel_hook_reports_the_position_the_timeline_records() -> None:
    kit, channel, session = await _room(_SlowStartTTS(latency_s=0.3))
    cancelled: list[Any] = []

    @kit.hook(HookTrigger.ON_TTS_CANCELLED, HookExecution.ASYNC)
    async def on_cancelled(event: Any, context: Any) -> None:
        cancelled.append(event)

    speaking = asyncio.create_task(channel.say(session, "a slow answer"))
    await asyncio.sleep(0.6)
    await channel.interrupt(session, reason="barge_in")
    await speaking
    await asyncio.sleep(0.05)

    events = await kit.store.list_events("r1")
    [stored] = [e for e in events if e.metadata.get("interrupted")]
    [event] = cancelled
    assert event.audio_position_ms == stored.metadata["played_ms"]
    # About 300 ms were heard; the 300 ms of synthesis latency are not counted.
    assert event.audio_position_ms < 450
    await kit.close()


async def test_a_measured_stream_is_at_zero_before_its_first_chunk() -> None:
    playback = TTSPlaybackState(
        session_id="s", text="slow", started_at=datetime.now(UTC) - timedelta(seconds=1)
    )
    assert playback.played_ms >= 1000  # nothing measured: elapsed time stands

    playback.start_measuring()

    assert playback.played_ms == 0


async def test_barge_in_threshold_ignores_synthesis_latency() -> None:
    """``allow_during_first_ms`` counts played audio: speech during the latency
    of a response that has not made a sound yet does not cut it."""
    kit, channel, session = await _room(
        None,
        InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE, allow_during_first_ms=500),
    )
    barge_ins: list[Any] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: Any, context: Any) -> None:
        barge_ins.append(event)

    playback = TTSPlaybackState(
        session_id=session.id,
        text="a response still synthesizing",
        started_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    playback.start_measuring()
    channel._playing_sessions[session.id] = playback  # noqa: SLF001

    channel._on_pipeline_vad_event(session, VADEvent(type=VADEventType.SPEECH_START))  # noqa: SLF001
    await asyncio.sleep(0.05)

    assert barge_ins == []
    assert session.id in channel._playing_sessions  # noqa: SLF001
    await kit.close()
