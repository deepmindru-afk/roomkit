"""The Voice Channel closes a provider's audio stream once playback stops.

A backend leaves ``send_audio()`` on a cancel without closing the iterator;
the provider's cleanup (an HTTP response, a GPU thread) must not wait for
garbage collection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.tts.base import TTSProvider


class _ClosingTTS(TTSProvider):
    def __init__(self) -> None:
        self.closed = False

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: Any = None
    ) -> AsyncIterator[AudioChunk]:
        try:
            for _ in range(100):
                yield AudioChunk(data=b"\x01\x00" * 160, sample_rate=16000)
        finally:
            self.closed = True


class _StopsEarlyBackend(MockVoiceBackend):
    """Reads one chunk, then returns without closing, as a cancelled transport does."""

    async def send_audio(self, session: VoiceSession, audio: Any) -> None:
        async for _ in audio:
            return


async def test_the_provider_stream_is_closed_when_the_transport_stops_early() -> None:
    tts = _ClosingTTS()
    backend = _StopsEarlyBackend()
    channel = VoiceChannel("voice-1", tts=tts, backend=backend)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "u", "voice-1")

    await channel.say(session, "a long answer")

    assert tts.closed
    await kit.close()
