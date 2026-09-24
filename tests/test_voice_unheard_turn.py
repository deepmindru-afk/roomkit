"""A response not heard yet waits for the user, who may supersede it (RMK-221, RFC §12.3.12).

"Combien j'ai de bord", pause, "et de cartes": the first half was routed and
its answer was on its way. The answer must not be said over the user, nor said
at all once they add to the question, and the model answers the two once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from roomkit import AIChannel, RoomKit, VoiceChannel
from roomkit.channels._voice_unheard import UnheardTurns
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.delivery import SUPERSEDED
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider

FIRST = "Combien j'ai de bord"
MORE = "Et de cartes"


class _ThinkingAI(AIProvider):
    """Answers each turn once released, and records what it was shown."""

    def __init__(self) -> None:
        self.contexts: list[AIContext] = []
        self.release = asyncio.Event()
        self.answers = ["Tu as 3 boards. ", "Tu as 3 boards et 42 cartes. "]

    @property
    def model_name(self) -> str:
        return "mock-thinking"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:  # pragma: no cover
        return AIResponse(content="unused")

    async def generate_stream(self, context: AIContext) -> AsyncIterator[Any]:
        self.contexts.append(context)
        answer = self.answers[len(self.contexts) - 1]
        await self.release.wait()
        yield answer


class _SentenceTTS(TTSProvider):
    """One audio chunk per sentence."""

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"\x00\x00", sample_rate=16000)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for sentence in text_stream:
            yield AudioChunk(data=_audio(sentence), sample_rate=16000)


def _audio(sentence: str) -> bytes:
    """The sentence as its own PCM, so the transport shows which answer was said."""
    data = sentence.strip().encode()
    return data + b" " * (len(data) % 2)


def _speech(audio: bytes) -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio),
    ]


async def _setup(
    transcripts: list[str],
) -> tuple[RoomKit, MockVoiceBackend, _ThinkingAI, str, VoiceSession]:
    backend, ai = MockVoiceBackend(), _ThinkingAI()
    vad = MockVADProvider(events=_speech(b"\x01\x00" * 4) + _speech(b"\x02\x00" * 4))
    voice = VoiceChannel(
        "voice-1",
        stt=MockSTTProvider(transcripts),
        tts=_SentenceTTS(),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.CONFIRMED, min_speech_ms=300
        ),
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(voice)
    kit.register_channel(AIChannel("ai-1", provider=ai))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    await kit.attach_channel(room.id, "ai-1")
    session = await kit.join(room.id, "voice-1", participant_id="user-0")
    assert isinstance(session, VoiceSession)
    return kit, backend, ai, room.id, session


async def _frame(backend: MockVoiceBackend, session: VoiceSession) -> None:
    await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x00" * 160))


async def _until(condition: Any, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def _said(backend: MockVoiceBackend) -> list[bytes]:
    return [audio for _, audio in backend.sent_audio]


async def _ai_rows(kit: RoomKit, room_id: str) -> list[RoomEvent]:
    events = await kit.store.list_events(room_id, offset=0, limit=50)
    return [e for e in events if e.source.channel_id == "ai-1"]


async def _route_first_turn(backend: MockVoiceBackend, ai: _ThinkingAI, session: Any) -> None:
    await _frame(backend, session)  # speech starts
    await _frame(backend, session)  # speech ends: FIRST is routed
    await _until(lambda: len(ai.contexts) == 1)


class TestResumingBeforeTheResponseIsHeard:
    async def test_the_response_waits_then_the_merged_turn_is_answered_once(self) -> None:
        kit, backend, ai, room_id, session = await _setup([FIRST, MORE])
        await _route_first_turn(backend, ai, session)

        await _frame(backend, session)  # the user resumes
        ai.release.set()  # the first answer is ready to be said
        await asyncio.sleep(0.35)
        assert _said(backend) == []  # nothing is said over the user

        await _frame(backend, session)  # speech ends: MORE
        await _until(lambda: len(_said(backend)) == 1)

        assert _said(backend) == [_audio("Tu as 3 boards et 42 cartes.")]
        second = ai.contexts[1].messages
        assert [(m.role, m.content) for m in second] == [("user", FIRST), ("user", MORE)]
        (cancelled, answered) = await _ai_rows(kit, room_id)
        assert cancelled.metadata["cancelled"] is True
        assert cancelled.metadata["cancellation_reason"] == SUPERSEDED
        assert isinstance(answered.content, TextContent)
        assert answered.content.body == "Tu as 3 boards et 42 cartes. "
        await kit.close()

    async def test_a_cough_releases_the_held_response(self) -> None:
        kit, backend, ai, room_id, session = await _setup([FIRST, ""])
        await _route_first_turn(backend, ai, session)

        await _frame(backend, session)  # a cough starts
        ai.release.set()
        await asyncio.sleep(0.05)
        assert _said(backend) == []
        await _frame(backend, session)  # and ends, with no words

        await _until(lambda: len(_said(backend)) == 1)
        assert _said(backend) == [_audio("Tu as 3 boards.")]
        assert len(ai.contexts) == 1
        (answered,) = await _ai_rows(kit, room_id)
        assert "cancelled" not in answered.metadata
        await kit.close()

    async def test_short_speech_with_words_does_not_supersede(self) -> None:
        kit, backend, ai, room_id, session = await _setup([FIRST, "Hm"])
        await _route_first_turn(backend, ai, session)

        await _frame(backend, session)
        await _frame(backend, session)  # 'Hm', well under min_speech_ms
        ai.release.set()

        # 'Hm' is a turn of its own, as before: the first answer was not dropped for it.
        await _until(lambda: len(ai.contexts) == 2)
        await _until(lambda: len(_said(backend)) >= 1)
        rows = await _ai_rows(kit, room_id)
        assert all(r.metadata.get("cancellation_reason") != SUPERSEDED for r in rows)
        await kit.close()

    async def test_a_segment_suppressed_as_echo_releases_the_response(self) -> None:
        # The turn is routed while a segment that started during other audio
        # is still on: that segment ends suppressed, and must not strand it.
        kit, backend, ai, room_id, session = await _setup([FIRST, MORE])
        voice = kit.get_channel("voice-1")
        assert isinstance(voice, VoiceChannel)
        await _frame(backend, session)  # speech 1 starts
        voice._playing_sessions[session.id] = TTSPlaybackState(session_id=session.id, text="x")
        await _frame(backend, session)  # speech 1 ends: FIRST is routed
        await _frame(backend, session)  # speech 2 starts over the audio: suppressed
        await _until(lambda: len(ai.contexts) == 1)
        voice._playing_sessions.pop(session.id, None)
        ai.release.set()
        await _frame(backend, session)  # speech 2 ends, discarded as echo

        await _until(lambda: len(_said(backend)) == 1)
        assert _said(backend) == [_audio("Tu as 3 boards.")]
        await kit.close()

    async def test_speech_over_audible_playback_is_still_guarded_as_echo(self) -> None:
        kit, backend, ai, room_id, session = await _setup([FIRST, MORE])
        voice = kit.get_channel("voice-1")
        assert isinstance(voice, VoiceChannel)
        await _route_first_turn(backend, ai, session)
        # Something else is playing (a filler, another participant's message).
        voice._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id, text="un instant"
        )
        await _frame(backend, session)
        assert session.id in voice._suppressed_sessions
        voice._playing_sessions.pop(session.id, None)
        ai.release.set()
        await kit.close()


class TestUnheardTurns:
    def _handle(self) -> Any:
        return object()

    async def test_a_heard_response_is_not_held(self) -> None:
        turns, handle = UnheardTurns(), self._handle()
        turns.register("s", handle, speaking_since=None)
        await turns.wait_to_play("s")  # first audio goes out
        assert turns.hold("s") is False  # speech now is a barge-in

    async def test_held_response_plays_once_released(self) -> None:
        turns, handle = UnheardTurns(), self._handle()
        turns.register("s", handle, speaking_since=None)
        assert turns.hold("s") is True
        waiter = asyncio.create_task(turns.wait_to_play("s"))
        await asyncio.sleep(0.01)
        assert not waiter.done()
        turns.release("s")
        await asyncio.wait_for(waiter, 1)

    async def test_speech_already_started_at_routing_holds_it(self) -> None:
        turns, handle = UnheardTurns(), self._handle()
        turns.register("s", handle, speaking_since=time.monotonic() - 0.4)
        waiter = asyncio.create_task(turns.wait_to_play("s"))
        await asyncio.sleep(0.01)
        assert not waiter.done()
        turns.note_speech_end("s")  # measured from the speech's own onset
        assert turns.take_superseded("s", min_speech_ms=300) is handle
        # The gate lets go, and says the audio must not go out.
        assert await asyncio.wait_for(waiter, 1) is False

    async def test_each_segment_is_measured_on_its_own(self) -> None:
        turns, handle = UnheardTurns(), self._handle()
        turns.register("s", handle, speaking_since=time.monotonic() - 1.0)
        turns.hold("s")  # a second short segment starts: the clock restarts
        turns.note_speech_end("s")
        assert turns.take_superseded("s", min_speech_ms=300) is None

    async def test_short_speech_supersedes_nothing(self) -> None:
        turns, handle = UnheardTurns(), self._handle()
        turns.register("s", handle, speaking_since=None)
        turns.hold("s")
        turns.note_speech_end("s")
        assert turns.take_superseded("s", min_speech_ms=300) is None

    async def test_discard_only_forgets_its_own_turn(self) -> None:
        turns, first, second = UnheardTurns(), self._handle(), self._handle()
        turns.register("s", first, speaking_since=None)
        turns.register("s", second, speaking_since=None)
        turns.discard("s", first)
        assert turns.hold("s") is True  # the second turn is still tracked
