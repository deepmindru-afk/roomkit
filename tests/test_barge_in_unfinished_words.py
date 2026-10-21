"""Words still being said decide no barge-in alone (RFC §12.3.13, RMK-889).

A streaming STT writes its partials word by word, or in pieces of words: live,
Voxtral wrote "Ou" before "Ouais", "Je" before "Je vois", "D" before
"D'accord". Judged alone, that first partial is no acknowledgement, and the
agent was cut by a "Ouais." it should have talked through, then answered it as
a turn. A partial that may still become an acknowledgement now waits for its
next words; a final whose partials were left so is judged whole.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceCapability
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.backchannel import PhraseBackchannelDetector
from roomkit.voice.stt.base import STTProvider

_SILENT_CHUNK = AudioFrame(data=b"\x00\x00" * 1600)


class _ScriptSTT(STTProvider):
    """Server-endpointing STT that says its scripted results on the first cycle."""

    def __init__(self, results: list[TranscriptionResult]) -> None:
        self._results = list(results)

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
        while self._results:
            yield self._results.pop(0)


def _partials(*texts: str) -> list[TranscriptionResult]:
    return [TranscriptionResult(text=text, is_final=False) for text in texts]


async def _said_over_the_agent(
    results: list[TranscriptionResult],
) -> tuple[list[str], list[object], list[object]]:
    """*results* from the STT while the agent speaks: the turns, cuts and
    backchannels that came of them."""
    stt = _ScriptSTT(results)
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),  # no VAD + streaming STT = continuous
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            backchannel_detector=PhraseBackchannelDetector(),
        ),
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    channel._session_played_ms[session.id] = 60_000  # noqa: SLF001 — past the echo guard
    heard: list[str] = []
    cuts: list[object] = []
    backchannels: list[object] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        cuts.append(event)

    @kit.hook(HookTrigger.ON_BACKCHANNEL, HookExecution.ASYNC)
    async def on_backchannel(event: object, context: object) -> None:
        backchannels.append(event)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: Any, context: object) -> HookResult:
        heard.append(event.text)
        return HookResult.allow()

    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id, text="Bonne continuation avec tes tests, je suis là."
    )
    await backend.simulate_audio_received(session, _SILENT_CHUNK)
    await asyncio.sleep(0.4)
    await channel.close()
    await kit.close()
    return heard, cuts, backchannels


async def test_a_word_said_in_pieces_does_not_cut_the_agent() -> None:
    # As Voxtral wrote it live: "Ou", then "Ouais".
    heard, cuts, backchannels = await _said_over_the_agent(
        [*_partials("Ou", "Ouais"), TranscriptionResult(text="Ouais.")]
    )

    assert cuts == []
    assert heard == []  # an acknowledgement is no turn
    assert len(backchannels) == 1


async def test_an_acknowledgement_said_word_by_word_does_not_cut_the_agent() -> None:
    heard, cuts, _ = await _said_over_the_agent(
        [*_partials("Je", "Je vois"), TranscriptionResult(text="Je vois.")]
    )

    assert cuts == []
    assert heard == []


async def test_the_next_words_cut_when_they_are_no_acknowledgement() -> None:
    heard, cuts, _ = await _said_over_the_agent(
        [
            *_partials("Je", "Je voudrais", "Je voudrais changer"),
            TranscriptionResult(text="Je voudrais changer l'heure."),
        ]
    )

    assert len(cuts) == 1
    assert heard == ["Je voudrais changer l'heure."]


async def test_a_final_its_partials_left_pending_is_judged_whole() -> None:
    # "No" might have been "noted"; nothing came after it but the final.
    heard, cuts, _ = await _said_over_the_agent(
        [*_partials("No"), TranscriptionResult(text="No.")]
    )

    assert len(cuts) == 1
    assert heard == ["No."]


async def test_a_word_that_opens_no_acknowledgement_still_cuts_at_once() -> None:
    heard, cuts, _ = await _said_over_the_agent(
        [*_partials("Attends"), TranscriptionResult(text="Attends.")]
    )

    assert len(cuts) == 1
    assert heard == ["Attends."]
