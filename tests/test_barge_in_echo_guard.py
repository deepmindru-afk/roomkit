"""A response's own echo does not cut it while the echo canceller learns the room
(RMK-697, RFC §12.3.13).

Each session starts with a fresh canceller: the response's opening words come
back through the microphone, an STT transcribes them ("Salut"), and they cut the
agent off; the final of that echo, seconds later, came in as the person's turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceCapability
from roomkit.voice.interruption import (
    InterruptionConfig,
    InterruptionHandler,
    InterruptionStrategy,
    echoes,
)
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.base import STTProvider

RESPONSE = "Salut ! Ça va très bien, merci. Et toi, comment tu vas ?"
_SILENT_CHUNK = AudioFrame(data=b"\x00\x00" * 1600)


@pytest.mark.parametrize(
    ("heard", "echo"),
    [
        ("Salut", True),
        ("Salut.", True),
        ("salut ! ca va", True),  # spelled by the STT, accents and case its own
        ("Salut ! Ça v", True),  # a partial catches the last word being said
        ("Et toi, comment", True),
        ("Salut, attends", False),  # a word the response does not say there
        ("ça va salut", False),  # its words, not in its order
        ("Bonjour", False),
        ("", False),
    ],
)
def test_speech_echoes_a_response_by_its_words_in_order(heard: str, echo: bool) -> None:
    assert echoes(heard, RESPONSE) is echo


@pytest.mark.parametrize(
    "strategy", [InterruptionStrategy.IMMEDIATE, InterruptionStrategy.DISABLED]
)
def test_an_echo_early_in_the_session_neither_cuts_nor_waits(strategy) -> None:
    handler = InterruptionHandler(InterruptionConfig(strategy=strategy))

    decision = handler.evaluate(
        playback_position_ms=900, speech_text="Salut", response_text=RESPONSE
    )

    assert (decision.is_echo, decision.should_interrupt, decision.queue_speech) == (
        True,
        False,
        False,
    )


@pytest.mark.parametrize(
    ("config", "session_played_ms"),
    [
        (InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE), 6000),
        (InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE, echo_guard_ms=0), 0),
    ],
)
def test_once_the_canceller_has_learned_the_room_its_words_cut_in(
    config: InterruptionConfig, session_played_ms: int
) -> None:
    """ "The sword" to "the sword or the shield?" is an answer, not an echo."""
    decision = InterruptionHandler(config).evaluate(
        playback_position_ms=900,
        speech_text="Salut",
        response_text=RESPONSE,
        session_played_ms=session_played_ms,
    )

    assert decision.should_interrupt and not decision.is_echo


class _StagedSTT(STTProvider):
    """Continuous STT: the partials at once, the final once ``release`` is set."""

    def __init__(self, partials: list[str], final: str) -> None:
        self._partials, self._final = partials, final
        self.release = asyncio.Event()
        self.partials_said = asyncio.Event()

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
        for text in self._partials:
            yield TranscriptionResult(text=text, is_final=False)
        self.partials_said.set()
        await self.release.wait()
        yield TranscriptionResult(text=self._final, is_final=True)
        await asyncio.Event().wait()  # the stream stays open, as a live one does


async def _speak_over(partials: list[str], final: str, *, played_before_ms: int = 0):
    """A response playing, *partials* over it, then *final* once it ended."""
    stt = _StagedSTT(partials, final)
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),  # no VAD + streaming STT = continuous
        interruption=InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE),
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    turns: list[str] = []
    cuts: list[object] = []
    partials: list[str] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        cuts.append(event)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: Any, context: object) -> HookResult:
        turns.append(event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC)
    async def on_partial(event: Any, context: object) -> None:
        partials.append(event.text)

    if played_before_ms:
        channel._session_played_ms[session.id] = played_before_ms  # noqa: SLF001
    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id, text=RESPONSE
    )
    await backend.simulate_audio_received(session, _SILENT_CHUNK)
    await asyncio.wait_for(stt.partials_said.wait(), 2)
    await asyncio.sleep(0.2)
    channel._playing_sessions.pop(session.id, None)  # noqa: SLF001 — the response ended
    stt.release.set()
    await asyncio.sleep(0.3)
    await channel.close()
    await kit.close()
    return turns, cuts, partials


async def test_the_echo_of_a_response_opening_neither_cuts_it_nor_becomes_a_turn() -> None:
    # The final comes after the response ended, as Meta's did 5.5 s later.
    turns, cuts, partials = await _speak_over(["Salut", "Salut ! Ça va"], "Salut")

    assert cuts == []
    assert turns == []
    assert partials == []  # nobody's words: a chat shows none of them


async def test_words_the_response_does_not_say_cut_it_and_are_the_turn() -> None:
    turns, cuts, partials = await _speak_over(
        ["Salut", "Salut, attends"], "Salut, attends une seconde"
    )

    assert len(cuts) == 1
    assert turns == ["Salut, attends une seconde"]
    assert partials == ["Salut, attends"]


async def test_past_the_guard_the_response_words_cut_in_and_are_the_turn() -> None:
    turns, cuts, _ = await _speak_over(["Salut"], "Salut", played_before_ms=10_000)

    assert len(cuts) == 1
    assert turns == ["Salut"]
