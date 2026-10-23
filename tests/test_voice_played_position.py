"""Barge-in hooks and the interruption policy measure the audio the user heard.

``TTSPlaybackState.position_ms`` counts from the state's creation, synthesis
latency included; ``played_ms`` counts the audio that actually went out. The
hooks, the timeline, the TTS context and ``allow_during_first_ms`` all read
``played_ms``, so one cut has one position and a response that has not made a
sound yet is not past its grace period.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

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
    """Synthesis that takes ``latency_s`` before its first second of audio,
    then keeps its stream open until *release* is set (a second at most)."""

    def __init__(self, latency_s: float) -> None:
        self._latency_s = latency_s
        self.release = asyncio.Event()

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        await asyncio.sleep(self._latency_s)
        yield AudioChunk(data=_ONE_SECOND, sample_rate=_RATE)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.release.wait(), 1)


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


@pytest.mark.wallclock
async def test_cancel_hook_reports_the_position_the_timeline_records() -> None:
    tts = _SlowStartTTS(latency_s=0.2)
    kit, channel, session = await _room(tts)
    cancelled: list[Any] = []

    @kit.hook(HookTrigger.ON_TTS_CANCELLED, HookExecution.ASYNC)
    async def on_cancelled(event: Any, context: Any) -> None:
        cancelled.append(event)

    speaking = asyncio.create_task(channel.say(session, "a slow answer"))
    async with asyncio.timeout(5):  # its first audio, past the synthesis latency
        while (playing := channel._playing_sessions.get(session.id)) is None or (  # noqa: SLF001
            playing.first_audio_at is None
        ):
            await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)
    await channel.interrupt(session, reason="barge_in")
    tts.release.set()
    await speaking
    async with asyncio.timeout(5):
        while not cancelled:
            await asyncio.sleep(0.005)

    events = await kit.store.list_events("r1")
    [stored] = [e for e in events if e.metadata.get("interrupted")]
    [event] = cancelled
    assert event.audio_position_ms == stored.metadata["played_ms"]
    # About 50 ms were heard; the 200 ms of synthesis latency are not counted.
    assert event.audio_position_ms < 200
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
