"""Mock text-to-speech provider for testing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.context import TTSContextLevel

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext


class MockTTSProvider(TTSProvider):
    """Mock text-to-speech for testing.

    ``context_level`` makes it declare a conversation context level; the
    contexts it receives land in ``contexts`` (one entry per streaming call)
    and the released ones in ``released``.
    """

    def __init__(
        self,
        voice: str = "mock-voice",
        *,
        context_level: TTSContextLevel = TTSContextLevel.NONE,
    ) -> None:
        self._default_voice = voice
        self._context_level = context_level
        self.calls: list[dict[str, str | None]] = []
        self.contexts: list[TTSContext | None] = []
        self.released: list[str] = []

    @property
    def context_level(self) -> TTSContextLevel:
        return self._context_level

    def release_context(self, context_id: str) -> None:
        self.released.append(context_id)

    @property
    def default_voice(self) -> str:
        return self._default_voice

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        from roomkit.models.event import AudioContent as AudioContentModel

        self.calls.append({"text": text, "voice": voice or self._default_voice})
        return AudioContentModel(
            url=f"https://mock.test/audio/{uuid4().hex}.mp3",
            mime_type="audio/mpeg",
            transcript=text,
            duration_seconds=len(text) * 0.05,  # ~50ms per char
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.calls.append({"text": text, "voice": voice or self._default_voice})
        self.contexts.append(context)
        # Simulate streaming with small chunks
        words = text.split()
        for i, word in enumerate(words):
            raw = f"mock-audio-{word}".encode()
            # Ensure even length for AudioFrame alignment (sample_width=2)
            if len(raw) % 2 != 0:
                raw += b"\x00"
            yield AudioChunk(
                data=raw,
                sample_rate=16000,
                is_final=(i == len(words) - 1),
            )
