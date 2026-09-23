"""A barge-in during a streamed AI response (RMK-189, RFC §12.2 step 13s).

The response already produced is stored with ``metadata.cancelled``, the
generation is closed rather than left running, no tool call starts after the
stop, and the interrupted utterance carries the text handed to TTS.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from roomkit import AIChannel, RoomKit, VoiceChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.interruption import InterruptionConfig
from roomkit.voice.tts.base import TTSProvider

HEARD = ["Sentence one is right here. ", "Sentence two is right here. "]
AFTER = "Sentence three was never said. "


class _HeldAI(AIProvider):
    """Streams two sentences, then holds until released, then calls a tool.

    Held is where a barge-in lands: the next pull is in flight inside the
    provider, which is exactly where a tool call used to slip past the stop.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.held = asyncio.Event()
        self.ended: list[str] = []
        self.tool_reached = False

    @property
    def model_name(self) -> str:
        return "mock-held"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:  # pragma: no cover
        return AIResponse(content="unused")

    async def generate_stream(self, context: AIContext) -> AsyncIterator[Any]:
        try:
            for sentence in HEARD:
                yield sentence
            self.held.set()
            await self.release.wait()
            yield AFTER
            self.tool_reached = True
            yield ToolCallStartMarker(tool_name="book_table", tool_id="t1", arguments={})
            yield ToolCallEndMarker(
                tool_name="book_table", tool_id="t1", arguments={}, result="ok"
            )
        except BaseException as exc:
            self.ended.append(type(exc).__name__)
            raise
        self.ended.append("done")


class _SentenceTTS(TTSProvider):
    """One audio chunk per sentence, synthesized as each sentence is read."""

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
        async for _ in text_stream:
            yield AudioChunk(data=b"\x00\x00", sample_rate=16000)


class _StoppableBackend(MockVoiceBackend):
    """A backend whose playback really stops on ``cancel_audio``.

    The plain mock plays to the end whatever happens, so a barge-in never
    cut a stream short in the other tests.
    """

    def __init__(self) -> None:
        super().__init__(capabilities=VoiceCapability.INTERRUPTION)
        self.cut: dict[str, asyncio.Event] = {}
        self.played: dict[str, int] = {}

    async def send_audio(
        self, session: VoiceSession, audio: bytes | AsyncIterator[AudioChunk]
    ) -> None:
        assert not isinstance(audio, bytes)
        cut = self.cut.setdefault(session.id, asyncio.Event())
        chunks = aiter(audio)
        while True:
            pull = asyncio.ensure_future(anext(chunks))
            stop = asyncio.ensure_future(cut.wait())
            await asyncio.wait({pull, stop}, return_when=asyncio.FIRST_COMPLETED)
            stop.cancel()
            if cut.is_set():
                pull.cancel()
                with contextlib.suppress(BaseException):
                    await pull
                return
            try:
                pull.result()
            except StopAsyncIteration:
                return
            self.played[session.id] = self.played.get(session.id, 0) + 1

    async def cancel_audio(self, session: VoiceSession) -> bool:
        self.cut.setdefault(session.id, asyncio.Event()).set()
        return True


async def _setup(
    *, sessions: int = 1, interruption: InterruptionConfig | None = None
) -> tuple[RoomKit, VoiceChannel, _StoppableBackend, _HeldAI, str, list[VoiceSession]]:
    backend, ai = _StoppableBackend(), _HeldAI()
    voice = VoiceChannel("voice-1", tts=_SentenceTTS(), backend=backend, interruption=interruption)
    kit = RoomKit(voice=backend)
    kit.register_channel(voice)
    kit.register_channel(AIChannel("ai-1", provider=ai))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    await kit.attach_channel(room.id, "ai-1")
    joined: list[VoiceSession] = []
    for i in range(sessions):
        session = await kit.join(room.id, "voice-1", participant_id=f"user-{i}")
        assert isinstance(session, VoiceSession)
        joined.append(session)
    return kit, voice, backend, ai, room.id, joined


def _turn(kit: RoomKit, room_id: str) -> asyncio.Task[Any]:
    message = InboundMessage(
        channel_id="voice-1", sender_id="user-0", content=TextContent(body="hi")
    )
    return asyncio.create_task(kit.process_inbound(message, room_id=room_id))


async def _events(kit: RoomKit, room_id: str) -> list[RoomEvent]:
    return await kit.store.list_events(room_id, offset=0, limit=50)


def _ai_rows(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.source.channel_id == "ai-1"]


def _interrupted(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.metadata.get("interrupted")]


class TestBargeInDuringStreamedResponse:
    async def test_produced_text_is_stored_cancelled(self) -> None:
        kit, voice, _, ai, room_id, (session,) = await _setup()
        turn = _turn(kit, room_id)
        await asyncio.wait_for(ai.held.wait(), 2)

        await voice.interrupt(session, reason="barge_in")
        await asyncio.wait_for(turn, 2)

        (row,) = _ai_rows(await _events(kit, room_id))
        assert row.type == EventType.MESSAGE
        assert isinstance(row.content, TextContent)
        assert row.content.body == "".join(HEARD)
        assert row.metadata.get("cancelled") is True
        await kit.close()

    async def test_generation_is_closed_and_no_tool_call_starts(self) -> None:
        kit, voice, _, ai, room_id, (session,) = await _setup()
        turn = _turn(kit, room_id)
        await asyncio.wait_for(ai.held.wait(), 2)

        await voice.interrupt(session, reason="barge_in")
        await asyncio.wait_for(turn, 2)
        # Were the pull still running off the turn, releasing the provider
        # now would let the tool call through.
        ai.release.set()
        await asyncio.sleep(0.05)

        assert ai.ended == ["CancelledError"]
        assert not ai.tool_reached
        types = {e.type for e in await _events(kit, room_id)}
        assert EventType.TOOL_CALL_START not in types
        await kit.close()

    async def test_interrupted_utterance_carries_the_text_handed_to_tts(self) -> None:
        kit, voice, _, ai, room_id, (session,) = await _setup()
        turn = _turn(kit, room_id)
        await asyncio.wait_for(ai.held.wait(), 2)

        await voice.interrupt(session, reason="barge_in")
        await asyncio.wait_for(turn, 2)

        (row,) = _interrupted(await _events(kit, room_id))
        assert isinstance(row.content, TextContent)
        assert row.content.body == " ".join(s.strip() for s in HEARD)
        assert "played_percentage" not in row.metadata
        await kit.close()

    async def test_one_session_stopping_does_not_stop_the_generation(self) -> None:
        kit, voice, backend, ai, room_id, (first, second) = await _setup(sessions=2)
        turn = _turn(kit, room_id)
        await asyncio.wait_for(ai.held.wait(), 2)

        await voice.interrupt(first, reason="barge_in")
        ai.release.set()
        await asyncio.wait_for(turn, 2)

        assert ai.ended == ["done"]
        rows = [e for e in _ai_rows(await _events(kit, room_id)) if e.type == EventType.MESSAGE]
        assert [r.content.body for r in rows if isinstance(r.content, TextContent)] == [
            "".join(HEARD) + AFTER
        ]
        assert all(not r.metadata.get("cancelled") for r in rows)
        # The session that kept listening heard every sentence.
        assert backend.played[second.id] == len(HEARD) + 1
        await kit.close()

    async def test_without_flush_the_response_plays_and_is_stored_whole(self) -> None:
        kit, voice, _, ai, room_id, (session,) = await _setup(
            interruption=InterruptionConfig(flush_partial_tts=False)
        )
        turn = _turn(kit, room_id)
        await asyncio.wait_for(ai.held.wait(), 2)

        await voice.interrupt(session, reason="barge_in")
        ai.release.set()
        await asyncio.wait_for(turn, 2)

        assert ai.ended == ["done"]
        events = await _events(kit, room_id)
        rows = [e for e in _ai_rows(events) if e.type == EventType.MESSAGE]
        assert all(not r.metadata.get("cancelled") for r in rows)
        assert EventType.TOOL_CALL_START in {e.type for e in events}
        await kit.close()
