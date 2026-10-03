"""A VoiceChannel refuses a TTS chunk that is not 16-bit PCM (RMK-415, RFC §12.2)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from roomkit import RoomKit, VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.tts.base import TTSProvider

AUDIO = b"ID3\x04\x00\x00"  # an even length: with a pipeline, AudioFrame would take it


class _TTS(TTSProvider):
    """A TTS streaming every chunk in *fmt*, as a vendor configured for it does."""

    def __init__(self, fmt: str) -> None:
        self._fmt = fmt

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=AUDIO, format=self._fmt)
        yield AudioChunk(data=b"", format=self._fmt, is_final=True)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for _sentence in text_stream:
            yield AudioChunk(data=AUDIO, format=self._fmt)
        yield AudioChunk(data=b"", format=self._fmt, is_final=True)


class _Room:
    """One voice session on a VoiceChannel, and the three ways to make it speak."""

    def __init__(self, kit: RoomKit, channel: VoiceChannel, session: VoiceSession) -> None:
        self.kit, self.channel, self.session = kit, channel, session
        room_id = session.room_id
        self.binding = ChannelBinding(
            room_id=room_id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        self.context = RoomContext(room=Room(id=room_id), bindings=[self.binding])

    def event(self, body: str) -> RoomEvent:
        return RoomEvent(
            room_id=self.session.room_id,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body=body),
        )

    async def say(self) -> None:
        await self.channel.say(self.session, "Hello there.")

    async def deliver(self) -> None:
        await self.channel.deliver(self.event("Hello there."), self.binding, self.context)

    async def deliver_stream(self) -> None:
        async def text() -> AsyncIterator[str]:
            yield "Hello there."

        await self.channel.deliver_stream(text(), self.event(""), self.binding, self.context)


async def _room(backend: MockVoiceBackend, tts: TTSProvider, *, pipeline: bool) -> _Room:
    channel = VoiceChannel(
        "voice-1",
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig() if pipeline else None,
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.join(room.id, "voice-1", participant_id="user-1")
    assert isinstance(session, VoiceSession)
    return _Room(kit, channel, session)


Speak = Callable[[_Room], Awaitable[None]]
ENTRY_POINTS = pytest.mark.parametrize(
    "speak",
    [_Room.say, _Room.deliver, _Room.deliver_stream],
    ids=["say", "deliver", "deliver_stream"],
)
# say() and deliver() log a failed synthesis; deliver_stream() raises it to the
# inbound stream, which fires ON_ERROR, when no session was served.
RAISED = {_Room.deliver_stream}
PIPELINE = pytest.mark.parametrize("pipeline", [False, True], ids=["no-pipeline", "pipeline"])


def _played(backend: MockVoiceBackend) -> bytes:
    return b"".join(data for _session_id, data in backend.sent_audio)


@ENTRY_POINTS
@PIPELINE
@pytest.mark.parametrize("fmt", ["mp3", "opus", "ulaw", "mulaw", "alaw"])
async def test_an_encoded_tts_chunk_is_refused_before_a_byte_plays(
    speak: Speak, pipeline: bool, fmt: str, caplog: pytest.LogCaptureFixture
) -> None:
    backend = MockVoiceBackend()
    room = await _room(backend, _TTS(fmt), pipeline=pipeline)
    refusal = f"VoiceChannel expects decoded PCM, got format '{fmt}'"

    if speak in RAISED:
        with pytest.raises(ValueError, match=refusal):
            await speak(room)
    else:
        await speak(room)
        assert refusal in caplog.text

    assert _played(backend) == b""
    await room.kit.close()


@ENTRY_POINTS
@PIPELINE
async def test_pcm_still_plays(speak: Speak, pipeline: bool) -> None:
    backend = MockVoiceBackend()
    room = await _room(backend, _TTS("pcm_s16le"), pipeline=pipeline)

    await speak(room)

    assert _played(backend) == AUDIO
    await room.kit.close()
