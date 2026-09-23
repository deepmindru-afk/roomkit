"""SEMANTIC waits for the words a streaming STT is producing (RFC §12.3.13).

A streaming STT commonly needs longer than ``min_speech_ms`` for its first
partial. While it is transcribing the speech, SEMANTIC waits for those words
up to ``transcript_wait_ms`` before judging on duration alone, in VAD mode (a
segment held during playback) and in continuous mode (the energy barge-in
classifies the words of the burst under way).
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
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceCapability, VoiceSession
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.backchannel.base import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.stt.base import STTProvider

_ONSET = VADEvent(type=VADEventType.SPEECH_START, audio_bytes=b"\x01\x00" * 320)
_LOUD = AudioFrame(data=(1000).to_bytes(2, "little", signed=True) * 320)  # 20 ms
_QUIET = AudioFrame(data=b"\x00\x00" * 320)


class _KeywordDetector(BackchannelDetector):
    _WORDS = {"uh-huh", "mm-hmm", "ok", "yeah"}

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    @property
    def name(self) -> str:
        return "keywords"

    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        self.seen.append(context.transcript)
        text = (context.transcript or "").strip().lower()
        return BackchannelDecision(is_backchannel=text in self._WORDS)


class _SlowPartialSTT(STTProvider):
    """Streaming STT whose first partial comes ``delay`` after the audio."""

    def __init__(self, partial: str | None, delay: float) -> None:
        self._partial = partial
        self._delay = delay

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text=self._partial or "", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        await asyncio.sleep(self._delay)
        if self._partial is not None:
            yield TranscriptionResult(text=self._partial, is_final=False)
        async for _ in audio_stream:
            pass


class _RepeatingPartialSTT(STTProvider):
    """Streaming STT that repeats one partial every 100 ms, as providers do."""

    def __init__(self, partial: str) -> None:
        self._partial = partial

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text=self._partial, is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        while True:
            await asyncio.sleep(0.1)
            yield TranscriptionResult(text=self._partial, is_final=False)


async def _room(
    stt: STTProvider,
    *,
    vad: bool,
    transcript_wait_ms: int = 1000,
    capabilities: VoiceCapability = VoiceCapability.NONE,
) -> tuple[RoomKit, VoiceChannel, VoiceSession, MockVoiceBackend, dict[str, list[Any]]]:
    backend = MockVoiceBackend(capabilities=capabilities)
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=MockVADProvider()) if vad else AudioPipelineConfig(),
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            backchannel_detector=_KeywordDetector(),
            min_speech_ms=50,
            transcript_wait_ms=transcript_wait_ms,
        ),
    )
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
    assert channel._continuous_stt is not vad  # noqa: SLF001

    seen: dict[str, list[Any]] = {"barge_in": [], "backchannel": []}

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: Any, context: Any) -> None:
        seen["barge_in"].append(event)

    @kit.hook(HookTrigger.ON_BACKCHANNEL, HookExecution.ASYNC)
    async def on_backchannel(event: Any, context: Any) -> None:
        seen["backchannel"].append(event)

    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id,
        text="Your appointment is confirmed for Tuesday at ten.",
        started_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    return kit, channel, session, backend, seen


async def _talk(
    backend: MockVoiceBackend, session: VoiceSession, seconds: float, *, dip_every: int = 0
) -> None:
    for i in range(int(seconds / 0.02)):
        dip = dip_every and i % dip_every == dip_every - 1
        await backend.simulate_audio_received(session, _QUIET if dip else _LOUD)
        await asyncio.sleep(0.02)


class TestVadMode:
    async def test_a_slow_backchannel_partial_still_decides(self) -> None:
        """The partial lands after min_speech_ms: it is still what decides."""
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT("uh-huh", delay=0.2), vad=True
        )

        channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
        await asyncio.sleep(0.4)

        assert seen["barge_in"] == []
        assert len(seen["backchannel"]) == 1
        await kit.close()

    async def test_no_words_by_the_cap_judges_on_duration(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT(None, delay=0), vad=True, transcript_wait_ms=400
        )

        channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
        await asyncio.sleep(0.12)
        assert seen["barge_in"] == []  # past min_speech_ms, still waiting for words

        await asyncio.sleep(0.45)
        assert len(seen["barge_in"]) == 1
        await kit.close()


class TestContinuousMode:
    async def test_energy_does_not_cut_a_recognized_backchannel(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT("uh-huh", delay=0.2), vad=False
        )

        await _talk(backend, session, 0.5)

        assert seen["barge_in"] == []
        assert len(seen["backchannel"]) == 1
        await kit.close()

    async def test_a_real_interruption_partial_cuts(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT("attends, stop", delay=0.2), vad=False
        )

        await _talk(backend, session, 0.4)

        assert len(seen["barge_in"]) == 1
        assert seen["backchannel"] == []
        await kit.close()

    async def test_energy_without_words_waits_for_the_cap(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT(None, delay=0), vad=False, transcript_wait_ms=450
        )

        await _talk(backend, session, 0.2)
        assert seen["barge_in"] == []  # energy fired and min_speech_ms passed, no words

        await _talk(backend, session, 0.5)
        assert len(seen["barge_in"]) == 1
        await kit.close()

    async def test_pauses_between_words_do_not_split_the_utterance(self) -> None:
        """A quiet frame ends an energy run, not the utterance: its words and
        its verdict hold, ON_BACKCHANNEL fires once and nothing is cut."""
        kit, channel, session, backend, seen = await _room(
            _RepeatingPartialSTT("uh-huh"), vad=False, transcript_wait_ms=200
        )

        await _talk(backend, session, 0.8, dip_every=6)

        assert len(seen["backchannel"]) == 1
        assert seen["barge_in"] == []
        await kit.close()

    async def test_transport_barge_in_reads_the_words_heard(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _RepeatingPartialSTT("uh-huh"),
            vad=False,
            capabilities=VoiceCapability.BARGE_IN,
        )
        await _talk(backend, session, 0.2)
        assert len(seen["backchannel"]) == 1

        await backend.simulate_barge_in(session)
        await asyncio.sleep(0.4)

        assert seen["barge_in"] == []
        assert len(seen["backchannel"]) == 1
        await kit.close()
