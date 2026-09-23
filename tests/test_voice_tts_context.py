"""TTS conversation context kept by the Voice Channel (RFC §12.2.2, §17.6).

A provider that declares a ``context_level`` receives the dialogue of its
voice session on every streaming call: what the user said (text, and audio
when kept) and what it said itself, cut to what was actually heard.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

from roomkit import (
    HookResult,
    HookTrigger,
    RoomKit,
    TTSContext,
    TTSContextConfig,
    TTSContextLevel,
    VoiceChannel,
)
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.events import TranscriptionEvent
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.dtmf.base import DTMFRedaction
from roomkit.voice.pipeline.dtmf.mock import MockDTMFDetector
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.context import TTSContextStore
from roomkit.voice.tts.mock import MockTTSProvider

_RATE = 16000
_CHUNK_MS = 100
_CHUNK = b"\x01\x00" * (_RATE * _CHUNK_MS // 1000)  # 100 ms of 16-bit mono PCM


class _PacedTTS(TTSProvider):
    """Streams 100 ms PCM chunks, one per sentence word, at a steady pace."""

    def __init__(self, level: TTSContextLevel, *, pace_s: float = 0.0) -> None:
        self._level = level
        self._pace_s = pace_s
        self.contexts: list[TTSContext | None] = []
        self.released: list[str] = []

    @property
    def context_level(self) -> TTSContextLevel:
        return self._level

    @property
    def supports_streaming_input(self) -> bool:
        return True

    def release_context(self, context_id: str) -> None:
        self.released.append(context_id)

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def _chunks(self, text: str) -> AsyncIterator[AudioChunk]:
        for _ in text.split():
            if self._pace_s:
                await asyncio.sleep(self._pace_s)
            yield AudioChunk(data=_CHUNK, sample_rate=_RATE)

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.contexts.append(context)
        async for chunk in self._chunks(text):
            yield chunk

    async def synthesize_stream_input(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncIterator[AudioChunk]:
        self.contexts.append(context)
        async for sentence in text_stream:
            async for chunk in self._chunks(sentence):
                yield chunk


class _LegacyTTS(TTSProvider):
    """A provider written before RFC §12.2.2: its signature has no ``context``."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(  # ty: ignore[invalid-method-override]
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.texts.append(text)
        yield AudioChunk(data=_CHUNK, sample_rate=_RATE)


async def _room(
    tts: TTSProvider,
    *,
    config: TTSContextConfig | None = None,
    stt: MockSTTProvider | None = None,
    pipeline: AudioPipelineConfig | None = None,
    participants: int = 1,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, list[VoiceSession]]:
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline, tts_context=config
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    sessions = [await kit.connect_voice("r1", f"user-{i}", "voice-1") for i in range(participants)]
    return kit, channel, backend, sessions


def _binding() -> ChannelBinding:
    return ChannelBinding(room_id="r1", channel_id="voice-1", channel_type=ChannelType.VOICE)


class TestPassingRules:
    async def test_a_provider_at_none_is_called_as_before(self) -> None:
        tts = _LegacyTTS()
        kit, channel, backend, [session] = await _room(tts)

        await channel.say(session, "hello there")

        assert tts.texts == ["hello there"]
        assert channel._tts_context is None
        await kit.close()

    async def test_the_second_call_hears_the_first(self) -> None:
        tts = _PacedTTS(TTSContextLevel.TEXT)
        kit, channel, backend, [session] = await _room(tts)

        await channel.say(session, "one two three")
        await channel.say(session, "four")

        first, second = tts.contexts
        assert first is not None and first.turns == ()
        assert second is not None and second.context_id == session.id
        [turn] = second.turns
        assert turn.role == "assistant"
        assert turn.text == "one two three"
        assert turn.turn_id == first.next_turn_id
        assert turn.played_ms == 3 * _CHUNK_MS
        assert turn.interrupted is False
        assert turn.audio is None  # TEXT level never carries audio
        await kit.close()

    async def test_audio_is_opt_in_even_at_audio_level(self) -> None:
        tts = _PacedTTS(TTSContextLevel.AUDIO)
        kit, channel, backend, [session] = await _room(tts)

        await channel.say(session, "one two")
        await channel.say(session, "three")

        assert tts.contexts[1] is not None
        assert tts.contexts[1].turns[0].audio is None
        await kit.close()

    async def test_audio_kept_when_asked(self) -> None:
        tts = _PacedTTS(TTSContextLevel.AUDIO)
        kit, channel, backend, [session] = await _room(
            tts, config=TTSContextConfig(include_audio=True)
        )

        await channel.say(session, "one two")
        await channel.say(session, "three")

        audio = tts.contexts[1].turns[0].audio  # type: ignore[union-attr]
        assert audio is not None
        assert audio.data == _CHUNK * 2
        await kit.close()

    async def test_disabled_config_keeps_nothing(self) -> None:
        tts = _PacedTTS(TTSContextLevel.TEXT)
        kit, channel, backend, [session] = await _room(tts, config=TTSContextConfig(enabled=False))

        await channel.say(session, "one")

        assert tts.contexts == [None]
        await kit.close()


class TestOneContextPerSession:
    async def test_each_session_hears_only_its_own_dialogue(self) -> None:
        tts = _PacedTTS(TTSContextLevel.TEXT)
        kit, channel, backend, sessions = await _room(tts, participants=2)

        await channel.say(sessions[0], "only for the first")
        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body="for both"),
        )
        context = await kit._build_context("r1")
        await channel._deliver_voice(event, _binding(), context)

        by_session = {c.context_id: c for c in tts.contexts[1:] if c is not None}
        assert [t.text for t in by_session[sessions[0].id].turns] == ["only for the first"]
        assert by_session[sessions[1].id].turns == ()
        turn = channel._tts_context.turns(sessions[1].id)[0]  # type: ignore[union-attr]
        assert turn.participant_id == "ai-1"
        await kit.close()


class TestUserTurns:
    async def test_the_utterance_is_recorded_after_transcription_hooks(self) -> None:
        tts = _PacedTTS(TTSContextLevel.AUDIO)
        stt = MockSTTProvider(transcripts=["my card is 4111"])
        kit, channel, backend, [session] = await _room(
            tts, config=TTSContextConfig(include_audio=True), stt=stt
        )
        utterance = b"\x02\x00" * 1600

        await channel._process_speech_end(session, utterance, "r1")

        [turn] = channel._tts_context.turns(session.id)  # type: ignore[union-attr]
        assert (turn.role, turn.participant_id, turn.text) == ("user", "user-0", "my card is 4111")
        assert turn.audio is not None and turn.audio.data == utterance
        await kit.close()

    async def test_a_redacted_transcript_keeps_no_audio(self) -> None:
        tts = _PacedTTS(TTSContextLevel.AUDIO)
        stt = MockSTTProvider(transcripts=["my card is 4111"])
        kit, channel, backend, [session] = await _room(
            tts, config=TTSContextConfig(include_audio=True), stt=stt
        )

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def redact(event: TranscriptionEvent, context: Any) -> HookResult:
            return HookResult.modify(replace(event, text="my card is ****"))

        await channel._process_speech_end(session, b"\x02\x00" * 1600, "r1")

        [turn] = channel._tts_context.turns(session.id)  # type: ignore[union-attr]
        assert turn.text == "my card is ****"
        assert turn.audio is None
        await kit.close()

    async def test_dtmf_with_redaction_drops_the_turn_audio(self) -> None:
        tts = _PacedTTS(TTSContextLevel.AUDIO)
        stt = MockSTTProvider(transcripts=["one two"])
        pipeline = AudioPipelineConfig(dtmf=MockDTMFDetector(), dtmf_redaction=DTMFRedaction())
        kit, channel, backend, [session] = await _room(
            tts, config=TTSContextConfig(include_audio=True), stt=stt, pipeline=pipeline
        )

        channel._on_pipeline_dtmf(session, object())
        await channel._process_speech_end(session, b"\x02\x00" * 1600, "r1")

        [turn] = channel._tts_context.turns(session.id)  # type: ignore[union-attr]
        assert turn.text == "one two"
        assert turn.audio is None
        await kit.close()

    async def test_a_routed_batch_flush_is_recorded(self) -> None:
        tts = _PacedTTS(TTSContextLevel.AUDIO)
        stt = MockSTTProvider(transcripts=["dictated note"])
        backend = MockVoiceBackend()
        channel = VoiceChannel(
            "voice-1",
            stt=stt,
            tts=tts,
            backend=backend,
            batch_mode=True,
            tts_context=TTSContextConfig(include_audio=True),
        )
        kit = RoomKit(voice=backend)
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "voice-1")
        session = await kit.connect_voice("r1", "user-0", "voice-1")
        recorded = b"\x03\x00" * 1600
        channel._batch_audio_buffers[session.id].extend(recorded)

        await channel.flush_stt(session, route=True)

        [turn] = channel._tts_context.turns(session.id)  # type: ignore[union-attr]
        assert turn.text == "dictated note"
        assert turn.audio is not None and turn.audio.data == recorded
        await kit.close()


class TestBargeIn:
    async def test_the_assistant_turn_holds_only_what_was_heard(self) -> None:
        tts = _PacedTTS(TTSContextLevel.AUDIO, pace_s=0.05)
        kit, channel, backend, [session] = await _room(
            tts, config=TTSContextConfig(include_audio=True)
        )

        speaking = asyncio.create_task(
            channel.say(session, "a b c d e f g h i j k l m n o p q r s t")
        )
        await asyncio.sleep(0.2)
        await channel.interrupt(session, reason="barge_in")
        await speaking

        [turn] = channel._tts_context.turns(session.id)  # type: ignore[union-attr]
        assert turn.interrupted is True
        assert turn.played_ms is not None and 50 <= turn.played_ms <= 1000
        assert turn.audio is not None
        heard_ms = len(turn.audio.data) / 2 / _RATE * 1000
        assert heard_ms <= turn.played_ms
        await kit.close()

    async def test_the_timeline_records_the_same_played_ms(self) -> None:
        tts = _PacedTTS(TTSContextLevel.TEXT, pace_s=0.05)
        kit, channel, backend, [session] = await _room(tts)

        speaking = asyncio.create_task(channel.say(session, "a b c d e f g h i j"))
        await asyncio.sleep(0.2)
        await channel.interrupt(session, reason="barge_in")
        await speaking

        events = await kit.store.list_events("r1")
        [stored] = [e for e in events if e.metadata.get("interrupted")]
        [turn] = channel._tts_context.turns(session.id)  # type: ignore[union-attr]
        assert stored.metadata["played_ms"] == turn.played_ms
        await kit.close()

    async def test_a_call_cut_before_any_audio_leaves_no_turn(self) -> None:
        tts = _PacedTTS(TTSContextLevel.TEXT, pace_s=0.5)
        kit, channel, backend, [session] = await _room(tts)

        speaking = asyncio.create_task(channel.say(session, "never heard"))
        await asyncio.sleep(0.05)
        await channel.interrupt(session, reason="barge_in")
        speaking.cancel()
        await asyncio.gather(speaking, return_exceptions=True)

        assert channel._tts_context.turns(session.id) == ()  # type: ignore[union-attr]
        await kit.close()


class TestStreamedResponse:
    async def test_a_streamed_response_is_recorded_as_one_turn(self) -> None:
        tts = _PacedTTS(TTSContextLevel.TEXT)
        kit, channel, backend, [session] = await _room(tts)
        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body=""),
        )

        async def tokens() -> AsyncIterator[str]:
            yield "Hello there, how are you today? "
            yield "Fine thanks."

        context = await kit._build_context("r1")
        await channel.deliver_stream(tokens(), event, _binding(), context)

        [turn] = channel._tts_context.turns(session.id)  # type: ignore[union-attr]
        assert turn.role == "assistant"
        assert turn.participant_id == "ai-1"
        assert turn.text == "Hello there, how are you today? Fine thanks."
        assert turn.played_ms == 8 * _CHUNK_MS
        await kit.close()


class TestLifecycle:
    async def test_unbind_releases_the_context(self) -> None:
        tts = _PacedTTS(TTSContextLevel.TEXT)
        kit, channel, backend, [session] = await _room(tts)
        await channel.say(session, "hello")

        channel.unbind_session(session)

        assert tts.released == [session.id]
        assert channel._tts_context.turns(session.id) == ()  # type: ignore[union-attr]
        await kit.close()

    async def test_close_releases_every_context(self) -> None:
        tts = MockTTSProvider(context_level=TTSContextLevel.TEXT)
        kit, channel, backend, [session] = await _room(tts)
        await channel.say(session, "hello")

        await channel.close()

        assert tts.released == [session.id]
        await kit.close()


class TestStoreBounds:
    def test_oldest_turns_go_past_max_turns(self) -> None:
        store = TTSContextStore(TTSContextConfig(max_turns=2), TTSContextLevel.TEXT)
        for text in ("one", "two", "three"):
            store.add_user_turn("s", "u", text)

        assert [t.text for t in store.turns("s")] == ["two", "three"]

    def test_oldest_audio_goes_past_max_audio_seconds_and_its_text_stays(self) -> None:
        store = TTSContextStore(
            TTSContextConfig(include_audio=True, max_audio_seconds=1.5), TTSContextLevel.AUDIO
        )
        one_second = b"\x00\x00" * _RATE
        store.add_user_turn("s", "u", "first", audio=one_second, sample_rate=_RATE)
        store.add_user_turn("s", "u", "second", audio=one_second, sample_rate=_RATE)

        first, second = store.turns("s")
        assert (first.text, first.audio) == ("first", None)
        assert second.audio is not None
        assert store.audio_seconds("s") == pytest.approx(1.0)

    def test_release_forgets_the_session(self) -> None:
        store = TTSContextStore(TTSContextConfig(), TTSContextLevel.TEXT)
        store.add_user_turn("s", "u", "hello")

        store.release("s")

        assert store.turns("s") == ()
        assert store.sessions() == []

    def test_invalid_bounds_are_refused(self) -> None:
        with pytest.raises(ValueError):
            TTSContextConfig(max_turns=0)
        with pytest.raises(ValueError):
            TTSContextConfig(max_audio_seconds=-1)
