"""A voice event reads the room's context for its hooks only when a hook is
registered on one of its triggers, and once for the whole event (RMK-797).

Every door that announces voice hooks is a case: an utterance in VAD,
continuous and batch mode, a turn completed after its wait, a streaming
partial, a barge-in, a say(), and a realtime session's transcription and
speech boundaries. The inbound pipeline's own
reads (RFC §10.1 steps 3 and 6) are not counted: they happen with or without
a hook, and this is about the reads the voice side adds.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RealtimeVoiceChannel, RoomKit
from roomkit.channels.voice import TTSPlaybackState, VoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.event import TextContent
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.turn.base import TurnDecision
from roomkit.voice.pipeline.turn.mock import MockTurnDetector
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

_INBOUND = frozenset({"_prepare_event_inner", "_run_precommit"})
_SYNC = frozenset({HookTrigger.ON_TRANSCRIPTION, HookTrigger.BEFORE_TTS})
_UTTERANCE = [
    VADEvent(type=VADEventType.SPEECH_START),
    None,
    VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"fake-audio"),
]

Act = Callable[[], Awaitable[None]]


class _PartialSTT(STTProvider):
    """Streams one partial, then its final."""

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, **kwargs: Any) -> TranscriptionResult:
        return TranscriptionResult(text="hello there")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], **kwargs: Any
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            pass
        yield TranscriptionResult(text="hello", is_final=False)
        yield TranscriptionResult(text="hello there", is_final=True)


class _ContinuousSTT(STTProvider):
    """Answers one final after the first chunk, as a server-endpointing STT does."""

    def __init__(self) -> None:
        self._finals = ["hello there"]

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, **kwargs: Any) -> TranscriptionResult:
        return TranscriptionResult(text="")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], **kwargs: Any
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        if self._finals:
            yield TranscriptionResult(text=self._finals.pop(0), is_final=True)


async def _voice_room(
    channel: VoiceChannel, backend: MockVoiceBackend
) -> tuple[RoomKit, Any, str]:
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, channel.channel_id)
    session = await kit.connect_voice(room.id, "user-1", channel.channel_id)
    return kit, session, room.id


async def _speak(backend: MockVoiceBackend, session: Any) -> None:
    for byte in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        await backend.simulate_audio_received(session, AudioFrame(data=byte))
    await asyncio.sleep(0.15)


async def _vad_utterance() -> tuple[RoomKit, str, Act]:
    backend = MockVoiceBackend()
    pipeline = AudioPipelineConfig(
        vad=MockVADProvider(events=list(_UTTERANCE)), turn_detector=MockTurnDetector()
    )
    stt = MockSTTProvider(transcripts=["hello there"])
    channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline)
    kit, session, room_id = await _voice_room(channel, backend)
    return kit, room_id, lambda: _speak(backend, session)


async def _turn_after_wait() -> tuple[RoomKit, str, Act]:
    """The turn detector holds the utterance; the turn completes after the wait."""
    backend = MockVoiceBackend()
    held = TurnDecision(is_complete=False, confidence=0.3, suggested_wait_ms=50)
    pipeline = AudioPipelineConfig(
        vad=MockVADProvider(events=list(_UTTERANCE)), turn_detector=MockTurnDetector([held])
    )
    stt = MockSTTProvider(transcripts=["hello there"])
    channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline)
    kit, session, room_id = await _voice_room(channel, backend)

    async def act() -> None:
        await _speak(backend, session)
        await asyncio.sleep(0.3)

    return kit, room_id, act


async def _barge_in() -> tuple[RoomKit, str, Act]:
    """The user speaks over a playback: ON_BARGE_IN, then the cut's ON_TTS_CANCELLED."""
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice-1", tts=MockTTSProvider(), backend=backend)
    kit, session, room_id = await _voice_room(channel, backend)
    playback = TTSPlaybackState(session_id=session.id, text="let me explain")
    channel._playing_sessions[session.id] = playback  # noqa: SLF001
    return kit, room_id, lambda: channel._handle_barge_in(session, playback, room_id)  # noqa: SLF001


async def _partial() -> tuple[RoomKit, str, Act]:
    backend = MockVoiceBackend()
    pipeline = AudioPipelineConfig(vad=MockVADProvider(events=list(_UTTERANCE)))
    channel = VoiceChannel("voice-1", stt=_PartialSTT(), backend=backend, pipeline=pipeline)
    kit, session, room_id = await _voice_room(channel, backend)
    return kit, room_id, lambda: _speak(backend, session)


async def _continuous_final() -> tuple[RoomKit, str, Act]:
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice-1", stt=_ContinuousSTT(), backend=backend, pipeline=AudioPipelineConfig()
    )
    kit, session, room_id = await _voice_room(channel, backend)

    async def act() -> None:
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00" * 1600))
        await asyncio.sleep(0.3)

    return kit, room_id, act


async def _batch_flush() -> tuple[RoomKit, str, Act]:
    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["hello there"])
    channel = VoiceChannel(
        "voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig(), batch_mode=True
    )
    kit, session, room_id = await _voice_room(channel, backend)

    async def act() -> None:
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 100))
        await channel.flush_stt(session, route=True)

    return kit, room_id, act


async def _say() -> tuple[RoomKit, str, Act]:
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice-1", tts=MockTTSProvider(), backend=backend)
    kit, session, room_id = await _voice_room(channel, backend)
    return kit, room_id, lambda: channel.say(session, "hello there")


async def _realtime_room() -> tuple[RoomKit, MockRealtimeProvider, Any, str]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt-voice-1", provider=provider, transport=MockRealtimeTransport()
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt-voice-1")
    session = await channel.start_session(room.id, "user-1", "fake-ws")
    return kit, provider, session, room.id


async def _realtime_final() -> tuple[RoomKit, str, Act]:
    kit, provider, session, room_id = await _realtime_room()

    async def act() -> None:
        await provider.simulate_transcription(session, "hello there", "user", True)
        await asyncio.sleep(0.1)

    return kit, room_id, act


async def _realtime_speech() -> tuple[RoomKit, str, Act]:
    kit, provider, session, room_id = await _realtime_room()

    async def act() -> None:
        await provider.simulate_speech_start(session)
        await provider.simulate_speech_end(session)
        await asyncio.sleep(0.1)

    return kit, room_id, act


@dataclass(frozen=True)
class Door:
    setup: Callable[[], Awaitable[tuple[RoomKit, str, Act]]]
    triggers: tuple[HookTrigger, ...]
    events: int  # voice events the door announces: each reads at most once
    routes: bool  # whether the transcript reaches the room's timeline


DOORS = {
    "vad_utterance": Door(
        _vad_utterance,
        (
            HookTrigger.ON_SPEECH_START,
            HookTrigger.ON_SPEECH_END,
            HookTrigger.ON_TRANSCRIPTION,
            HookTrigger.ON_TURN_COMPLETE,
        ),
        events=2,  # the speech start, then the utterance
        routes=True,
    ),
    # The utterance's read crosses into the wait's task: ON_TURN_COMPLETE shares it.
    "turn_after_wait": Door(
        _turn_after_wait,
        (HookTrigger.ON_TURN_INCOMPLETE, HookTrigger.ON_TURN_COMPLETE),
        events=1,
        routes=True,
    ),
    "partial": Door(_partial, (HookTrigger.ON_PARTIAL_TRANSCRIPTION,), events=1, routes=True),
    "barge_in": Door(
        _barge_in,
        (HookTrigger.ON_BARGE_IN, HookTrigger.ON_TTS_CANCELLED),
        events=2,
        routes=False,
    ),
    "continuous_final": Door(
        _continuous_final,
        (HookTrigger.ON_SPEECH_END, HookTrigger.ON_TRANSCRIPTION),
        events=1,
        routes=True,
    ),
    "batch_flush": Door(
        _batch_flush,
        (HookTrigger.ON_SPEECH_END, HookTrigger.ON_TRANSCRIPTION),
        events=1,
        routes=True,
    ),
    # Two events: AFTER_TTS reads the room as it is once the synthesis played.
    "say": Door(_say, (HookTrigger.BEFORE_TTS, HookTrigger.AFTER_TTS), events=2, routes=False),
    "realtime_final": Door(
        _realtime_final, (HookTrigger.ON_TRANSCRIPTION,), events=1, routes=True
    ),
    "realtime_speech": Door(
        _realtime_speech,
        (HookTrigger.ON_SPEECH_START, HookTrigger.ON_SPEECH_END),
        events=2,
        routes=False,
    ),
}


def _spy_voice_reads(kit: RoomKit) -> list[str]:
    """Record each context read made outside the inbound pipeline, by its caller."""
    reads: list[str] = []
    build = kit._build_context

    async def counted(room_id: str, **kwargs: Any) -> RoomContext:
        caller = sys._getframe(1).f_code.co_name
        if caller not in _INBOUND:
            reads.append(caller)
        return await build(room_id, **kwargs)

    kit._build_context = counted  # type: ignore[method-assign]
    return reads


def _register(kit: RoomKit, triggers: tuple[HookTrigger, ...], seen: list[tuple]) -> None:
    for trigger in triggers:
        if trigger in _SYNC:

            async def gate(event: Any, ctx: RoomContext, t: HookTrigger = trigger) -> HookResult:
                seen.append((t, ctx.room.id))
                return HookResult.allow()

            kit.hook(trigger, name=f"gate-{trigger.value}")(gate)
        else:

            async def observe(event: Any, ctx: RoomContext, t: HookTrigger = trigger) -> None:
                seen.append((t, ctx.room.id))

            kit.hook(trigger, execution=HookExecution.ASYNC, name=f"obs-{trigger.value}")(observe)


async def _transcript_reached(kit: RoomKit, room_id: str) -> bool:
    timeline = await kit.get_timeline(room_id)
    return any(
        isinstance(e.content, TextContent) and e.content.body == "hello there" for e in timeline
    )


@pytest.mark.parametrize("door", list(DOORS.values()), ids=list(DOORS))
async def test_no_hook_no_read(door: Door) -> None:
    kit, room_id, act = await door.setup()
    try:
        reads = _spy_voice_reads(kit)
        await act()
        assert reads == []
        if door.routes:
            assert await _transcript_reached(kit, room_id)
    finally:
        await kit.close()


@pytest.mark.parametrize("door", list(DOORS.values()), ids=list(DOORS))
async def test_hooks_read_once_per_event(door: Door) -> None:
    seen: list[tuple] = []
    kit, room_id, act = await door.setup()
    try:
        _register(kit, door.triggers, seen)
        reads = _spy_voice_reads(kit)
        await act()
        await asyncio.sleep(0.05)  # async observers run off the event
        assert {trigger for trigger, _ in seen} == set(door.triggers)
        assert {room for _, room in seen} == {room_id}
        assert len(reads) == door.events
    finally:
        await kit.close()


@pytest.mark.parametrize("observer", [False, True], ids=["no_hook", "async_observer"])
async def test_an_utterance_whose_room_is_gone_reaches_the_inbound(observer: bool) -> None:
    """An ASYNC hook cannot block (RFC §9.3): its context failing to read costs
    it its run, never the utterance, which the inbound pipeline routes to a
    room it opens under the same id (RFC §10.1 step 2)."""
    kit, room_id, act = await _vad_utterance()
    try:
        seen: list[tuple] = []
        if observer:
            _register(kit, (HookTrigger.ON_SPEECH_END,), seen)
        await kit.store.delete_room(room_id)
        await act()
        assert seen == []
        assert await _transcript_reached(kit, room_id)
    finally:
        await kit.close()


async def test_a_sync_hook_whose_context_fails_to_read_drops_the_utterance() -> None:
    """ON_TRANSCRIPTION fails closed: a chain that cannot run lets nothing pass."""
    kit, room_id, act = await _vad_utterance()
    try:
        seen: list[tuple] = []
        _register(kit, (HookTrigger.ON_TRANSCRIPTION,), seen)
        await kit.store.delete_room(room_id)
        await act()
        assert seen == []
        assert await kit.store.get_room(room_id) is None
    finally:
        await kit.close()
