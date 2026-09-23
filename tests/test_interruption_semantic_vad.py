"""SEMANTIC in VAD mode classifies words, not an empty speech onset (RFC §12.3.13).

The pipeline VAD's SPEECH_START carries no transcript and no duration, and a
keyword detector cannot find "uh-huh" in ``""``. SEMANTIC therefore holds the
speech: with a streaming STT the held segment is transcribed and each partial
is classified; without one, the second look at ``min_speech_ms`` judges on
duration alone (the CONFIRMED fallback).
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
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceSession
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


class _KeywordDetector(BackchannelDetector):
    """The guide's detector: a short acknowledgement is a backchannel."""

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


class _PartialsSTT(STTProvider):
    """Streaming STT that speaks its partials once audio arrives."""

    def __init__(self, partials: list[str]) -> None:
        self._partials = partials

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text=" ".join(self._partials), is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        for partial in self._partials:
            yield TranscriptionResult(text=partial, is_final=False)
        async for _ in audio_stream:
            pass
        yield TranscriptionResult(text=self._partials[-1], is_final=True)


class _BatchSTT(STTProvider):
    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="hello", is_final=True)


async def _room(
    stt: STTProvider, *, min_speech_ms: int = 300
) -> tuple[RoomKit, VoiceChannel, VoiceSession, _KeywordDetector, dict[str, list[Any]]]:
    detector = _KeywordDetector()
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=MockVADProvider()),
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            backchannel_detector=detector,
            min_speech_ms=min_speech_ms,
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
    assert not channel._continuous_stt  # noqa: SLF001

    seen: dict[str, list[Any]] = {"barge_in": [], "backchannel": [], "transcription": []}

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: Any, context: Any) -> None:
        seen["barge_in"].append(event)

    @kit.hook(HookTrigger.ON_BACKCHANNEL, HookExecution.ASYNC)
    async def on_backchannel(event: Any, context: Any) -> None:
        seen["backchannel"].append(event)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.ASYNC)
    async def on_transcription(event: Any, context: Any) -> None:
        seen["transcription"].append(event)

    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id,
        text="Your appointment is confirmed for Tuesday at ten.",
        started_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    return kit, channel, session, detector, seen


async def test_speech_onset_alone_never_consults_the_detector() -> None:
    kit, channel, session, detector, seen = await _room(_BatchSTT())

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.05)

    assert detector.seen == []
    assert seen["barge_in"] == []
    assert session.id in channel._playing_sessions  # noqa: SLF001
    await kit.close()


async def test_a_backchannel_lets_the_bot_talk_and_its_segment_is_discarded() -> None:
    kit, channel, session, detector, seen = await _room(_PartialsSTT(["uh-huh"]))

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.05)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await asyncio.sleep(0.4)  # past min_speech_ms: the timer must not cut it

    assert detector.seen == ["uh-huh"]
    assert len(seen["backchannel"]) == 1
    assert seen["barge_in"] == []
    assert seen["transcription"] == []
    assert session.id in channel._playing_sessions  # noqa: SLF001
    await kit.close()


async def test_an_acknowledged_utterance_running_long_is_not_cut() -> None:
    """Once "uh-huh" is recognized, speech still going at min_speech_ms is not
    judged again on duration alone."""
    kit, channel, session, detector, seen = await _room(_PartialsSTT(["uh-huh"]), min_speech_ms=50)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.2)

    assert seen["barge_in"] == []
    assert len(seen["backchannel"]) == 1
    await kit.close()


async def test_a_real_interruption_cuts_in_and_becomes_the_turn() -> None:
    kit, channel, session, detector, seen = await _room(_PartialsSTT(["attends", "attends, stop"]))

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.05)

    assert len(seen["barge_in"]) == 1
    assert session.id not in channel._playing_sessions  # noqa: SLF001
    assert detector.seen == ["attends"]  # the first real words decide

    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await asyncio.sleep(0.1)

    # The held stream kept the utterance from its first word.
    assert [e.text for e in seen["transcription"]] == ["attends, stop"]
    await kit.close()


async def test_without_streaming_stt_semantic_falls_back_to_confirmed() -> None:
    kit, channel, session, detector, seen = await _room(_BatchSTT(), min_speech_ms=50)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.02)
    assert seen["barge_in"] == []

    await asyncio.sleep(0.1)  # the speech sustained past min_speech_ms

    assert detector.seen == [""]  # judged on duration alone, once
    assert len(seen["barge_in"]) == 1
    await kit.close()


async def test_without_streaming_stt_a_short_blip_does_not_cut() -> None:
    kit, channel, session, detector, seen = await _room(_BatchSTT(), min_speech_ms=100)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.02)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await asyncio.sleep(0.15)

    assert detector.seen == []
    assert seen["barge_in"] == []
    assert seen["transcription"] == []
    await kit.close()


async def test_a_held_segment_reaches_no_partial_hook() -> None:
    kit, channel, session, detector, seen = await _room(_PartialsSTT(["uh-huh"]))
    partials: list[Any] = []

    @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC)
    async def on_partial(event: Any, context: Any) -> None:
        partials.append(event)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.05)

    assert detector.seen == ["uh-huh"]
    assert partials == []
    await kit.close()


async def test_a_speech_end_right_after_the_cut_in_keeps_the_turn() -> None:
    """The cut-in decision lets the segment through at once: a speech end
    landing before the scheduled barge-in runs is the user's turn, not echo."""
    kit, channel, session, detector, seen = await _room(_PartialsSTT(["attends, stop"]))
    processed: list[bytes] = []

    async def capture(sess: Any, audio: bytes, room_id: str, stream_state: Any, **kw: Any) -> None:
        processed.append(audio)

    channel._on_pipeline_vad_event(session, VADEvent(type=VADEventType.SPEECH_START))  # noqa: SLF001
    channel._process_speech_end = capture  # type: ignore[method-assign]  # noqa: SLF001
    # The partial decides, and the speech ends before any task has run.
    channel._on_held_transcript(session, "r1", "attends, stop")  # noqa: SLF001
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await asyncio.sleep(0.05)

    assert processed == [b"\x11\x22" * 160]
    assert len(seen["barge_in"]) == 1
    await kit.close()
