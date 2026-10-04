"""A TTS that fails mid-stream is reported on a backend that absorbs its own errors (RMK-448)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.voice.backends.base import PlaybackErrors
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.tts.base import TTSProvider

AUDIO = b"\x00\x01" * 160


class _TTS(TTSProvider):
    """A TTS whose stream plays one chunk, then fails when *fails* is set."""

    def __init__(self, *, fails: bool) -> None:
        self._fails = fails

    @property
    def name(self) -> str:
        return "FlakyTTS"

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def _audio(self) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=AUDIO)
        if self._fails:
            raise RuntimeError("401 from the TTS vendor")

    def synthesize_stream(self, text: str, **kwargs: Any) -> AsyncIterator[AudioChunk]:
        return self._audio()

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], **kwargs: Any
    ) -> AsyncIterator[AudioChunk]:
        async for _sentence in text_stream:
            pass
        async for chunk in self._audio():
            yield chunk


class _Call:
    """A FastRTC voice session over a WebSocket, and what the framework reported."""

    def __init__(self) -> None:
        self.tts_errors: list[dict[str, Any]] = []
        self.after_tts: list[str] = []

    async def open(self, tts: TTSProvider) -> None:
        self.backend = FastRTCVoiceBackend()
        self.channel = VoiceChannel("voice-1", tts=tts, backend=self.backend)
        self.kit = kit = RoomKit(voice=self.backend)
        kit.register_channel(self.channel)

        @kit.on("tts_error")
        async def on_tts_error(event: Any) -> None:
            self.tts_errors.append(event.data)

        @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC)
        async def after_tts(text: Any, ctx: Any) -> None:
            self.after_tts.append(text)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.join(room.id, "voice-1", participant_id="user-1")
        assert isinstance(session, VoiceSession)
        self.session = session
        self.ws = AsyncMock()
        self.ws.client_state = None
        self.backend._register_websocket("ws-1", session.id, self.ws)

    async def deliver_stream(self) -> None:
        async def text() -> AsyncIterator[str]:
            yield "Bonjour."

        room_id = self.session.room_id
        binding = ChannelBinding(
            room_id=room_id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        event = RoomEvent(
            room_id=room_id,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body=""),
        )
        context = RoomContext(room=Room(id=room_id), bindings=[binding])
        await self.channel.deliver_stream(text(), event, binding, context)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


class TestSay:
    async def test_a_failing_tts_is_reported_and_not_said(self) -> None:
        call = _Call()
        await call.open(_TTS(fails=True))

        await call.channel.say(call.session, "Bonjour.")
        await _settle()

        assert call.tts_errors == [
            {
                "provider": "FlakyTTS",
                "error": "401 from the TTS vendor",
                "session_id": call.session.id,
            }
        ]
        assert call.after_tts == []
        await call.kit.close()

    async def test_a_working_tts_is_said(self) -> None:
        call = _Call()
        await call.open(_TTS(fails=False))

        await call.channel.say(call.session, "Bonjour.")
        await _settle()

        assert call.tts_errors == []
        assert call.after_tts == ["Bonjour."]
        media = [
            c.args[0]
            for c in call.ws.send_json.call_args_list
            if c.args[0].get("event") == "media"
        ]
        assert len(media) == 1  # the one chunk, beside the transcription messages
        await call.kit.close()


class TestDeliverStream:
    async def test_a_failing_tts_reaches_the_inbound_stream(self) -> None:
        call = _Call()
        await call.open(_TTS(fails=True))

        with pytest.raises(RuntimeError, match="401 from the TTS vendor"):
            await call.deliver_stream()
        await call.kit.close()


class TestPlaybackErrors:
    """One rule for every backend: the stream's failure leaves, the backend's own stays."""

    async def test_the_stream_failure_leaves_the_block(self) -> None:
        async def chunks() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=AUDIO)
            raise RuntimeError("tts down")

        with (
            pytest.raises(RuntimeError, match="tts down"),
            PlaybackErrors(logging.getLogger("test"), "playing %s", "s1") as play,
        ):
            async for _chunk in play.watch(chunks()):
                pass

    async def test_it_leaves_even_when_the_backend_caught_it(self) -> None:
        async def chunks() -> AsyncIterator[AudioChunk]:
            raise RuntimeError("tts down")
            yield AudioChunk(data=AUDIO)  # pragma: no cover

        with (
            pytest.raises(RuntimeError, match="tts down"),
            PlaybackErrors(logging.getLogger("test"), "playing %s", "s1") as play,
        ):
            try:
                async for _chunk in play.watch(chunks()):
                    pass
            except RuntimeError:
                pass  # a backend helper that swallows what it reads

    async def test_the_backend_failure_is_logged_and_absorbed(self, caplog: Any) -> None:
        async def chunks() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=AUDIO)

        with PlaybackErrors(logging.getLogger("test"), "playing %s", "s1") as play:
            async for _chunk in play.watch(chunks()):
                raise OSError("socket gone")

        assert "playing s1" in caplog.text
        assert "socket gone" in caplog.text

    async def test_a_cancellation_is_not_touched(self) -> None:
        with (
            pytest.raises(asyncio.CancelledError),
            PlaybackErrors(logging.getLogger("test"), "playing %s", "s1"),
        ):
            raise asyncio.CancelledError
