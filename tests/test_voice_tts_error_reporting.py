"""A failed synthesis is reported once per session, and is not said (RMK-448, RFC §8.2)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.tts.base import TTSProvider

AUDIO = b"\x00\x01" * 160


class _FailingTTS(TTSProvider):
    """A TTS whose stream plays one chunk, then fails."""

    @property
    def name(self) -> str:
        return "FailingTTS"

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(self, text: str, **kwargs: Any) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=AUDIO)
        raise RuntimeError("429 from the TTS vendor")


class _NoStreamTTS(TTSProvider):
    """A TTS without streaming synthesis, which a voice channel requires."""

    @property
    def name(self) -> str:
        return "NoStreamTTS"

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError


class _WorkingTTS(TTSProvider):
    @property
    def name(self) -> str:
        return "WorkingTTS"

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(self, text: str, **kwargs: Any) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=AUDIO)


class _Backend(MockVoiceBackend):
    """Mock backend whose transport fails for the sessions in ``down``."""

    def __init__(self) -> None:
        super().__init__()
        self.down: set[str] = set()

    async def send_audio(self, session: VoiceSession, audio: Any) -> None:
        if session.id in self.down:
            raise ConnectionError(f"transport down for {session.id}")
        await super().send_audio(session, audio)


class _Room:
    def __init__(self) -> None:
        self.tts_errors: list[dict[str, Any]] = []
        self.after_tts: list[str] = []
        self.sessions: list[VoiceSession] = []

    async def open(self, tts: TTSProvider, participants: int) -> _Room:
        self.backend = _Backend()
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
        for n in range(participants):
            session = await kit.join(room.id, "voice-1", participant_id=f"user-{n}")
            assert isinstance(session, VoiceSession)
            self.sessions.append(session)
        return self

    async def say(self) -> None:
        await self.channel.say(self.sessions[0], "Bonjour.")

    async def deliver(self) -> None:
        room_id = self.sessions[0].room_id
        binding = ChannelBinding(
            room_id=room_id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        event = RoomEvent(
            room_id=room_id,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body="Bonjour."),
        )
        context = RoomContext(room=Room(id=room_id), bindings=[binding])
        await self.channel.deliver(event, binding, context)

    def reported(self) -> list[tuple[str, str | None]]:
        return sorted((e["provider"], e.get("session_id")) for e in self.tts_errors)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


Speak = Callable[[_Room], Awaitable[None]]


@pytest.mark.parametrize("speak", [_Room.say, _Room.deliver], ids=["say", "deliver"])
@pytest.mark.parametrize("tts", [_FailingTTS, _NoStreamTTS], ids=["fails-mid-stream", "no-stream"])
async def test_one_failed_session_is_reported_once_and_not_said(
    speak: Speak, tts: type[TTSProvider]
) -> None:
    room = await _Room().open(tts(), participants=1)

    await speak(room)
    await _settle()

    assert room.reported() == [(tts().name, room.sessions[0].id)]
    assert room.after_tts == []
    await room.kit.close()


@pytest.mark.parametrize("tts", [_FailingTTS, _NoStreamTTS], ids=["fails-mid-stream", "no-stream"])
async def test_a_delivery_reports_every_session_that_failed(tts: type[TTSProvider]) -> None:
    room = await _Room().open(tts(), participants=2)

    await room.deliver()
    await _settle()

    assert room.reported() == sorted((tts().name, s.id) for s in room.sessions)
    assert room.after_tts == []
    await room.kit.close()


async def test_a_session_failing_beside_a_served_one_is_reported() -> None:
    room = await _Room().open(_WorkingTTS(), participants=2)
    room.backend.down.add(room.sessions[1].id)

    await room.deliver()
    await _settle()

    assert room.reported() == [("WorkingTTS", room.sessions[1].id)]
    assert room.after_tts == ["Bonjour."]  # the other session heard it
    await room.kit.close()
