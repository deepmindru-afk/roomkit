"""Text-to-speech provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING

from roomkit.voice.tts.context import TTSContextLevel
from roomkit.voice.voices import DialogueTurn, VoiceInfo, check_dialogue, filter_voices

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.base import AudioChunk
    from roomkit.voice.tts.context import TTSContext


class TTSProvider(ABC):
    """Text-to-speech provider."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'elevenlabs', 'openai')."""
        return self.__class__.__name__

    @property
    def default_voice(self) -> str | None:
        """Default voice ID. Override in subclasses."""
        return None

    @property
    def max_dialogue_speakers(self) -> int:
        """Distinct speakers one :meth:`synthesize_dialogue` call can voice.

        0, the default, means no dialogue at all (RFC §12.2).
        """
        return 0

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of the voices this provider offers.

        No credentials or network: call it on the class. The base returns an
        empty list; a provider with a fixed or curated set overrides it.
        """
        return []

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """Voices the vendor reports, custom voices the caller owns included.

        Every provider applies the filters the same way, server side or on
        the results (:func:`~roomkit.voice.voices.filter_voices`). The base
        filters :meth:`available_voices`, for a vendor with no voices endpoint.

        Args:
            language: BCP-47 tag or prefix (``"fr"`` matches ``fr-CA``).
            gender: ``"male"``, ``"female"`` or ``"neutral"``.
            query: Case-insensitive text found in the name or description.
        """
        return filter_voices(
            self.available_voices(), language=language, gender=gender, query=query
        )

    async def synthesize_dialogue(
        self, turns: Sequence[DialogueTurn], voices: Mapping[str, str]
    ) -> AudioContent:
        """Voice a scripted exchange in one clip (RFC §12.2).

        Args:
            turns: The lines, in order, each naming its speaker.
            voices: Voice id per speaker.

        Returns:
            AudioContent whose ``transcript`` holds each turn as
            ``"<speaker>: <text>"`` on its own line.

        Raises:
            NotImplementedError: The provider voices no dialogue
                (:attr:`max_dialogue_speakers` is 0).
            ValueError: A turn names an unmapped speaker, or more speakers
                than :attr:`max_dialogue_speakers`.
        """
        check_dialogue(turns, voices, max_speakers=self.max_dialogue_speakers, provider=self.name)
        raise NotImplementedError(
            f"{self.name} announces dialogue but does not implement synthesize_dialogue"
        )

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice ID (uses default_voice if not specified).

        Returns:
            AudioContent with URL to generated audio.
        """
        ...

    @property
    def supports_streaming_input(self) -> bool:
        """Whether this TTS accepts streaming text input."""
        return False

    @property
    def context_level(self) -> TTSContextLevel:
        """What conversation context this TTS consumes (RFC §12.2.2).

        A provider left at NONE never receives a ``context``. Override to
        receive the dialogue of the session on each streaming call.
        """
        return TTSContextLevel.NONE

    def release_context(self, context_id: str) -> None:  # noqa: B027
        """Drop any state held for *context_id* (its voice session ended).

        Called by the Voice Channel when the session is unbound. A provider
        that keeps per-context state (a dialogue KV cache, request handles)
        overrides it; an unknown context is a no-op.
        """

    async def synthesize_stream_input(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio from streaming text chunks.

        Override for providers that accept an async text stream as input.
        ``context`` is the session's dialogue so far, passed only when
        ``context_level`` is not NONE.
        """
        raise NotImplementedError(f"{self.name} does not support streaming text input.")
        yield  # pragma: no cover

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks as they're generated.

        Override for providers that support streaming. ``context`` is the
        session's dialogue so far, passed only when ``context_level`` is not
        NONE.
        Default: synthesizes full audio and yields single chunk.
        """
        raise NotImplementedError(
            f"{self.name} does not support streaming synthesis. Use synthesize() instead."
        )
        # Make this an async generator (unreachable, but required for type)
        yield  # pragma: no cover

    async def warmup(self) -> None:  # noqa: B027
        """Pre-load models so the first call is fast. Override in subclasses."""

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses if needed."""
