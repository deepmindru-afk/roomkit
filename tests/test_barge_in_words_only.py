"""Only words interrupt (RMK-559): a VoiceChannel in continuous mode, during
playback, with PhraseBackchannelDetector(cut_without_words=False)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice import InterruptionConfig, InterruptionStrategy, PhraseBackchannelDetector
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.base import STTProvider

_LOUD = AudioFrame(data=(1000).to_bytes(2, "little", signed=True) * 320)  # 20 ms, RMS 1000
# How long a run of sound waits for its words, on a test's scale: past the
# first 100 ms the STT is handed, so its words come inside the wait.
_WORDS_WAIT_MS = 200


class _HearsAfter(STTProvider):
    """A streaming STT that hears *words* in the first audio it is handed
    (100 ms of it), inside the run's wait for them, or nothing at all."""

    def __init__(self, words: str | None) -> None:
        self._words = words

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        if self._words:
            yield TranscriptionResult(text=self._words, is_final=False)
        async for _ in audio_stream:
            pass


def _semantic(*, cut_without_words: bool) -> InterruptionConfig:
    return InterruptionConfig(
        strategy=InterruptionStrategy.SEMANTIC,
        backchannel_detector=PhraseBackchannelDetector(cut_without_words=cut_without_words),
        min_speech_ms=40,
        transcript_wait_ms=_WORDS_WAIT_MS,
    )


async def _cut(words: str | None, *, cut_without_words: bool) -> bool:
    """Whether loud audio, with *words* heard in it, cuts the agent's reply:
    fed until it does, at most twice the run's wait for words."""
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice",
        stt=_HearsAfter(words),
        backend=backend,
        pipeline=AudioPipelineConfig(),
        interruption=_semantic(cut_without_words=cut_without_words),
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice")
    session = await kit.join("r1", "voice", participant_id="u1")
    channel.bind_session(
        session,
        "r1",
        ChannelBinding(room_id="r1", channel_id="voice", channel_type=ChannelType.VOICE),
    )
    cut = asyncio.Event()

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: Any, ctx: Any) -> None:
        cut.set()

    channel._playing_sessions[session.id] = TTSPlaybackState(
        session_id=session.id,
        text="Here is the full summary of the discussion.",
        started_at=datetime.now(UTC) - timedelta(seconds=2),
    )
    try:
        for _ in range(2 * _WORDS_WAIT_MS // 20):  # loud audio while the agent speaks
            await backend.simulate_audio_received(session, _LOUD)
            await asyncio.sleep(0.02)
            if cut.is_set():
                break
    finally:
        await kit.close()  # a barge-in decided by the last frame reaches its hook first
    return cut.is_set()


async def test_sound_without_words_cuts_by_default() -> None:
    assert await _cut(None, cut_without_words=True) is True


async def test_sound_without_words_does_not_cut_when_only_words_interrupt() -> None:
    assert await _cut(None, cut_without_words=False) is False


async def test_words_still_cut_when_only_words_interrupt() -> None:
    assert await _cut("attends, stop", cut_without_words=False) is True


async def test_an_acknowledgement_still_does_not_cut() -> None:
    assert await _cut("ouais", cut_without_words=False) is False
