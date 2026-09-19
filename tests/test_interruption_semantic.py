"""The SEMANTIC strategy classifies the partial transcript it was given (RFC §12.6).

``InterruptionHandler.evaluate`` takes ``speech_text`` for exactly this strategy,
and the continuous-STT loop has the partial transcript in hand when it consults
the handler — but it left the argument at its empty default. A
``BackchannelDetector`` behind ``InterruptionStrategy.SEMANTIC`` therefore always
classified ``""``: an acknowledgement and a real interruption were the same
utterance to it, whatever the detector was worth.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceCapability
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.backchannel.base import BackchannelDecision
from roomkit.voice.pipeline.backchannel.mock import MockBackchannelDetector
from roomkit.voice.stt.base import STTProvider


class _PartialsSTT(STTProvider):
    """Server-endpointing STT that speaks its scripted partials on the first cycle."""

    def __init__(self, partials: list[str]) -> None:
        self._partials = list(partials)

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
        while self._partials:
            yield TranscriptionResult(text=self._partials.pop(0), is_final=False)


# Silence: the energy barge-in path must stay quiet so only the transcript path
# reaches the handler. 3200 bytes is one STT buffer flush.
_SILENT_CHUNK = AudioFrame(data=b"\x00\x00" * 1600)


async def test_semantic_strategy_receives_each_partial_transcript() -> None:
    detector = MockBackchannelDetector(
        decisions=[
            BackchannelDecision(is_backchannel=True, label="acknowledgement"),
            BackchannelDecision(is_backchannel=False),
        ]
    )
    stt = _PartialsSTT(["uh-huh", "wait, stop"])
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),  # no VAD + streaming STT = continuous
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC, backchannel_detector=detector
        ),
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    assert channel._continuous_stt  # noqa: SLF001

    barge_ins: list[object] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        barge_ins.append(event)

    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id, text="Your appointment is confirmed for Tuesday at ten."
    )
    await backend.simulate_audio_received(session, _SILENT_CHUNK)
    await asyncio.sleep(0.3)

    # The detector judged the words the user said, in order.
    assert [c.transcript for c in detector.evaluations] == ["uh-huh", "wait, stop"]
    # The acknowledgement let the bot keep talking; the real interruption did not.
    assert len(barge_ins) == 1

    await channel.close()
    await kit.close()
