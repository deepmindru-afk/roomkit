"""deliver_stream / deliver with several voice sessions on one binding (RMK-188)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from roomkit import RoomKit, VoiceChannel
from roomkit.channels._stream_fanout import StreamBranch, StreamFanOut
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.tts.base import TTSProvider

SENTENCES = ["Hello there, this is a test.", "Second sentence here!"]


def _audio(text: str) -> bytes:
    """Fake 16-bit PCM for *text*: an even byte count, as the pipeline requires."""
    data = f"audio-{text}".encode()
    return data + b"\x00" * (len(data) % 2)


class _RecordingTTS(TTSProvider):
    """Streaming-input TTS: one audio chunk per sentence, as it is read."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.calls.append([text])
        yield AudioChunk(data=_audio(text), sample_rate=16000, is_final=True)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        received: list[str] = []
        self.calls.append(received)
        async for sentence in text_stream:
            received.append(sentence)
            yield AudioChunk(data=_audio(sentence), sample_rate=16000)


class _ScriptedBackend(MockVoiceBackend):
    """Mock backend whose send_audio can fail, stop early, or wait per session."""

    def __init__(self) -> None:
        super().__init__()
        self.fail: set[str] = set()
        self.stop_after: dict[str, int] = {}  # session id -> chunks played
        self.started: dict[str, asyncio.Event] = {}
        self.wait_for: dict[str, str] = {}

    async def send_audio(
        self, session: VoiceSession, audio: bytes | AsyncIterator[AudioChunk]
    ) -> None:
        self.started.setdefault(session.id, asyncio.Event()).set()
        other = self.wait_for.get(session.id)
        if other is not None:
            await self.started.setdefault(other, asyncio.Event()).wait()
        if session.id in self.fail:
            raise RuntimeError(f"transport down for {session.id}")
        limit = self.stop_after.get(session.id)
        if limit is not None and not isinstance(audio, bytes):
            # Barge-in: the transport stops reading after *limit* chunks.
            played = 0
            async for chunk in audio:
                self.sent_audio.append((session.id, chunk.data))
                played += 1
                if played == limit:
                    return
        await super().send_audio(session, audio)


async def _setup(
    backend: MockVoiceBackend, tts: TTSProvider
) -> tuple[RoomKit, VoiceChannel, list[VoiceSession], RoomEvent, ChannelBinding, RoomContext]:
    channel = VoiceChannel("voice-1", tts=tts, backend=backend)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    sessions: list[VoiceSession] = []
    for participant_id in ("user-1", "user-2"):
        session = await kit.join(room.id, "voice-1", participant_id=participant_id)
        assert isinstance(session, VoiceSession)
        sessions.append(session)
    event = RoomEvent(
        room_id=room.id,
        source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
        content=TextContent(body=""),
    )
    binding = ChannelBinding(room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE)
    context = RoomContext(room=Room(id=room.id), bindings=[binding])
    return kit, channel, sessions, event, binding, context


async def _text() -> AsyncIterator[str]:
    yield SENTENCES[0] + " "
    yield SENTENCES[1]


def _by_role(backend: MockVoiceBackend, session: VoiceSession, role: str) -> list[str]:
    return [t for sid, t, r in backend.sent_transcriptions if sid == session.id and r == role]


class TestDeliverStreamSeveralSessions:
    async def test_every_session_hears_every_sentence(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, event, binding, context = await _setup(backend, tts)

        await channel.deliver_stream(_text(), event, binding, context)

        assert tts.calls == [SENTENCES, SENTENCES]
        full = " ".join(SENTENCES)
        for session in sessions:
            assert _by_role(backend, session, "assistant_interim") == SENTENCES
            assert _by_role(backend, session, "assistant") == [full]
            audio = b"".join(d for sid, d in backend.sent_audio if sid == session.id)
            assert audio == b"".join(_audio(s) for s in SENTENCES)
        await kit.close()

    async def test_sessions_are_served_in_parallel(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, event, binding, context = await _setup(backend, tts)
        # Session 1's playback cannot finish before session 2's has started:
        # serving the sessions one after the other deadlocks here.
        backend.wait_for[sessions[0].id] = sessions[1].id

        await asyncio.wait_for(channel.deliver_stream(_text(), event, binding, context), 2.0)

        assert tts.calls == [SENTENCES, SENTENCES]
        await kit.close()

    async def test_session_stopping_early_does_not_cut_the_other(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, event, binding, context = await _setup(backend, tts)
        backend.stop_after[sessions[0].id] = 1

        await channel.deliver_stream(_text(), event, binding, context)

        assert _by_role(backend, sessions[1], "assistant_interim") == SENTENCES
        audio = b"".join(d for sid, d in backend.sent_audio if sid == sessions[1].id)
        assert audio == b"".join(_audio(s) for s in SENTENCES)
        await kit.close()

    async def test_failed_session_gets_no_final_transcript(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, event, binding, context = await _setup(backend, tts)
        backend.fail.add(sessions[0].id)

        await channel.deliver_stream(_text(), event, binding, context)

        assert _by_role(backend, sessions[0], "assistant") == []
        assert _by_role(backend, sessions[1], "assistant") == [" ".join(SENTENCES)]
        await kit.close()

    async def test_every_session_failing_raises(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, event, binding, context = await _setup(backend, tts)
        backend.fail.update(s.id for s in sessions)

        with pytest.raises(RuntimeError, match="transport down"):
            await channel.deliver_stream(_text(), event, binding, context)

        assert [r for _, _, r in backend.sent_transcriptions if r == "assistant"] == []
        await kit.close()

    async def test_barge_in_leaves_the_rest_of_the_response_unread(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, event, binding, context = await _setup(backend, tts)
        await kit.leave(sessions[1])
        backend.stop_after[sessions[0].id] = 2
        spoken = [f"Sentence number {i} is here." for i in range(8)]
        pulled: list[str] = []
        seen: list[str] = []

        async def source() -> AsyncIterator[str]:
            try:
                for sentence in spoken:
                    pulled.append(sentence)
                    yield sentence + " "
            except BaseException as exc:
                seen.append(type(exc).__name__)
                raise

        await channel.deliver_stream(source(), event, binding, context)

        # Pulled at the pace of playback, as a single reader did: not the
        # whole response, and the final transcript is what was played.
        assert len(pulled) <= 4
        (final,) = _by_role(backend, sessions[0], "assistant")
        assert spoken[-1] not in final
        # No pull was in flight when playback stopped, so nothing was
        # cancelled inside the source: closing it is the caller's step
        # (RFC §12.2 step 13s, tests/test_voice_stream_barge_in.py).
        assert seen == []
        await kit.close()

    async def test_source_error_raises_even_when_a_session_was_served(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, event, binding, context = await _setup(backend, tts)
        backend.stop_after[sessions[0].id] = 1

        async def broken() -> AsyncIterator[str]:
            yield SENTENCES[0] + " "
            yield SENTENCES[1] + " "
            raise ValueError("llm down")

        with pytest.raises(ValueError, match="llm down"):
            await channel.deliver_stream(broken(), event, binding, context)

        assert [r for _, _, r in backend.sent_transcriptions if r == "assistant"] == []
        await kit.close()


class TestDeliverSeveralSessions:
    async def test_sessions_are_served_in_parallel(self) -> None:
        backend, tts = _ScriptedBackend(), _RecordingTTS()
        kit, channel, sessions, _, binding, context = await _setup(backend, tts)
        backend.wait_for[sessions[0].id] = sessions[1].id
        event = RoomEvent(
            room_id=binding.room_id,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body="Hello both."),
        )

        await asyncio.wait_for(channel.deliver(event, binding, context), 2.0)

        assert tts.calls == [["Hello both."], ["Hello both."]]
        await kit.close()


async def _items(items: list[str], pulled: list[str]) -> AsyncIterator[str]:
    for item in items:
        pulled.append(item)
        yield item


async def _drain(fan_out: StreamFanOut[str]) -> list[list[str]]:
    """Run the producer while every branch reads to the end."""

    async def read(branch: StreamBranch[str]) -> list[str]:
        return [item async for item in branch]

    producer = asyncio.create_task(fan_out.run())
    results = await asyncio.gather(*(read(b) for b in fan_out.branches))
    await producer
    return list(results)


class TestStreamFanOut:
    async def test_each_branch_gets_every_item(self) -> None:
        fan_out = StreamFanOut(_items(["a", "b"], []), 3)
        assert await _drain(fan_out) == [["a", "b"]] * 3

    async def test_no_branch_pulls_nothing(self) -> None:
        pulled: list[str] = []
        await StreamFanOut(_items(["a"], pulled), 0).run()
        assert pulled == []

    async def test_pulls_only_on_demand(self) -> None:
        pulled: list[str] = []
        fan_out = StreamFanOut(_items(["a", "b", "c"], pulled), 1)
        producer = asyncio.create_task(fan_out.run())
        assert await anext(fan_out.branches[0]) == "a"
        await asyncio.sleep(0)
        assert pulled == ["a"]
        fan_out.branches[0].close()
        await asyncio.wait_for(producer, 1.0)
        assert pulled == ["a"]

    async def test_stops_pulling_once_every_branch_is_closed(self) -> None:
        pulled: list[str] = []
        fan_out = StreamFanOut(_items(["a", "b", "c"], pulled), 2)
        for branch in fan_out.branches:
            branch.close()
        await asyncio.wait_for(fan_out.run(), 1.0)
        assert pulled == []

    async def test_closed_branch_leaves_the_others_running(self) -> None:
        fan_out = StreamFanOut(_items(["a", "b"], []), 2)
        fan_out.branches[0].close()
        assert await _drain(fan_out) == [[], ["a", "b"]]

    async def test_source_error_reaches_every_branch(self) -> None:
        async def broken() -> AsyncIterator[str]:
            yield "a"
            raise ValueError("llm down")

        fan_out = StreamFanOut(broken(), 2)
        producer = asyncio.create_task(fan_out.run())
        for branch in fan_out.branches:
            assert await anext(branch) == "a"
        for branch in fan_out.branches:
            with pytest.raises(ValueError, match="llm down"):
                await anext(branch)
        await producer
        assert isinstance(fan_out.error, ValueError)
