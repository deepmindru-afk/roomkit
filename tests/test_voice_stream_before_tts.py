"""BEFORE_TTS on a streamed response runs on each sentence (RMK-268, RFC §9.3, §12.2 12s.b)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import RoomKit, VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.mock import MockTTSProvider

SENTENCES = [
    "Hello there, this is a test.",
    "Your card is 4111 1111 1111 1111.",
    "Goodbye for now!",
]


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

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        received: list[str] = []
        self.calls.append(received)
        async for sentence in text_stream:
            received.append(sentence)
            yield AudioChunk(data=_audio(sentence), sample_rate=16000)


async def _setup(
    sessions: int = 1,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, _RecordingTTS, list[VoiceSession], Any]:
    backend, tts = MockVoiceBackend(), _RecordingTTS()
    channel = VoiceChannel("voice-1", tts=tts, backend=backend)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    joined: list[VoiceSession] = []
    for i in range(sessions):
        session = await kit.join(room.id, "voice-1", participant_id=f"user-{i}")
        assert isinstance(session, VoiceSession)
        joined.append(session)

    async def deliver() -> None:
        event = RoomEvent(
            room_id=room.id,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body=""),
        )
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        context = RoomContext(room=Room(id=room.id), bindings=[binding])
        await channel.deliver_stream(_text(), event, binding, context)

    return kit, channel, backend, tts, joined, deliver


async def _text() -> AsyncIterator[str]:
    for sentence in SENTENCES:
        yield sentence + " "


def _final(backend: MockVoiceBackend) -> list[str]:
    return [t for _, t, role in backend.sent_transcriptions if role == "assistant"]


def _after_tts(kit: RoomKit) -> list[str]:
    seen: list[str] = []

    @kit.hook(HookTrigger.AFTER_TTS, HookExecution.ASYNC)
    async def record(text: str, context: RoomContext) -> None:
        seen.append(text)

    return seen


class TestStreamedSentenceHook:
    async def test_modify_replaces_the_sentence_the_tts_reads(self) -> None:
        kit, _, backend, tts, _, deliver = await _setup()
        after = _after_tts(kit)

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def redact(text: str, context: RoomContext) -> HookResult:
            if "4111" in text:
                return HookResult.modify("Your card number is redacted.")
            return HookResult.allow()

        await deliver()

        spoken = [SENTENCES[0], "Your card number is redacted.", SENTENCES[2]]
        assert tts.calls == [spoken]
        assert _final(backend) == [" ".join(spoken)]
        assert after == [" ".join(spoken)]
        assert b"4111" not in b"".join(d for _, d in backend.sent_audio)
        await kit.close()

    async def test_block_drops_the_sentence_and_the_next_ones_are_judged_on_their_own(
        self,
    ) -> None:
        kit, _, backend, tts, _, deliver = await _setup()
        after = _after_tts(kit)

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def withhold(text: str, context: RoomContext) -> HookResult:
            return HookResult.block("card number") if "4111" in text else HookResult.allow()

        await deliver()

        spoken = [SENTENCES[0], SENTENCES[2]]
        assert tts.calls == [spoken]
        assert _final(backend) == [" ".join(spoken)]
        assert after == [" ".join(spoken)]
        await kit.close()

    async def test_a_sentence_redacted_to_nothing_is_not_synthesized(self) -> None:
        kit, _, _, tts, _, deliver = await _setup()

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def erase(text: str, context: RoomContext) -> HookResult:
            return HookResult.modify("") if "4111" in text else HookResult.allow()

        await deliver()

        assert tts.calls == [[SENTENCES[0], SENTENCES[2]]]
        await kit.close()

    @pytest.mark.parametrize("failure", ["raises", "times_out", "not_a_result", "wrong_type"])
    async def test_a_failing_hook_fails_closed_for_its_sentence(self, failure: str) -> None:
        kit, _, backend, tts, _, deliver = await _setup()

        @kit.hook(HookTrigger.BEFORE_TTS, timeout=0.05)
        async def broken(text: str, context: RoomContext) -> Any:
            if "4111" not in text:
                return HookResult.allow()
            if failure == "raises":
                raise RuntimeError("redaction service down")
            if failure == "times_out":
                await asyncio.sleep(1)
            if failure == "not_a_result":
                return "not a HookResult"
            return HookResult.modify(42)

        await deliver()

        assert tts.calls == [[SENTENCES[0], SENTENCES[2]]]
        assert all("4111" not in t for t in _final(backend))
        await kit.close()

    async def test_every_sentence_dropped_sends_nothing_and_fires_no_after_tts(self) -> None:
        kit, _, backend, tts, _, deliver = await _setup()
        after = _after_tts(kit)

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def mute(text: str, context: RoomContext) -> HookResult:
            return HookResult.block("silenced")

        await deliver()

        assert tts.calls == [[]]
        assert _final(backend) == []
        assert after == []
        await kit.close()

    async def test_the_hook_runs_once_per_sentence_for_every_session(self) -> None:
        kit, _, backend, tts, sessions, deliver = await _setup(sessions=2)
        judged: list[str] = []

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def count(text: str, context: RoomContext) -> HookResult:
            judged.append(text)
            return HookResult.block("x") if "4111" in text else HookResult.allow()

        await deliver()

        assert judged == SENTENCES
        assert tts.calls == [[SENTENCES[0], SENTENCES[2]]] * 2
        await kit.close()

    async def test_without_a_hook_the_stream_is_untouched(self) -> None:
        kit, _, backend, tts, _, deliver = await _setup()
        after = _after_tts(kit)

        await deliver()

        assert tts.calls == [SENTENCES]
        streamed = "".join(s + " " for s in SENTENCES)
        assert _final(backend) == [streamed]
        assert after == [streamed]
        await kit.close()

    async def test_a_hook_that_allows_everything_leaves_the_text_as_streamed(self) -> None:
        kit, _, backend, tts, _, deliver = await _setup()

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def allow(text: str, context: RoomContext) -> HookResult:
            return HookResult.allow()

        await deliver()

        assert tts.calls == [SENTENCES]
        assert _final(backend) == ["".join(s + " " for s in SENTENCES)]
        await kit.close()


class TestStandardPathUnchanged:
    async def test_a_non_streamed_response_is_judged_whole_once(self) -> None:
        backend, tts = MockVoiceBackend(), MockTTSProvider()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)
        kit = RoomKit(voice=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.join(room.id, "voice-1", participant_id="user-0")
        judged: list[str] = []

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def count(text: str, context: RoomContext) -> HookResult:
            judged.append(text)
            return HookResult.allow()

        body = " ".join(SENTENCES)
        event = RoomEvent(
            room_id=room.id,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body=body),
        )
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        await channel.deliver(event, binding, RoomContext(room=Room(id=room.id)))

        assert judged == [body]
        await kit.close()
