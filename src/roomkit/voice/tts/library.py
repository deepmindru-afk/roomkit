"""Custom voices: the ``VoiceLibrary`` interface (RFC §12.2.4).

A voice can be made rather than picked: designed from a description, or
replicated from a recording of a person. Making one manages the vendor's voice
store, not synthesis, so it lives apart from :class:`TTSProvider`; the voice it
returns is spoken through the vendor's TTS like any other, by its ``id``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from roomkit.models.event import AudioContent
from roomkit.voice.voices import VoiceInfo

VoiceAudio = AudioContent | bytes | Path
"""A recording handed to a library: an ``AudioContent`` carrying a ``data:``
URL, the bytes of a WAV file, or the path of one."""


class VoiceConsentError(Exception):
    """The vendor refused a replication because the consent did not hold.

    Raised in place of the vendor's own error so a caller can tell "record the
    consent again" from any other failure; no voice exists afterwards.
    """


class CustomVoice(BaseModel):
    """A voice a :class:`VoiceLibrary` made.

    Attributes:
        voice: The voice; ``voice.id`` is what the vendor's TTS accepts.
        stored: Held by the vendor for the caller's account, listed by its
            TTS and readable with :meth:`VoiceLibrary.get_voice`. An unstored
            voice's id is the only handle on it.
        expires_at: When the vendor stops honouring the id, if it says.
        sample: The vendor's preview of the voice, if it sends one.
    """

    voice: VoiceInfo
    stored: bool
    expires_at: datetime | None = None
    sample: AudioContent | None = None


class VoiceLibrary(ABC):
    """Designs, replicates and deletes custom voices (RFC §12.2.4).

    Replicating a person's voice takes their recorded consent as a required
    argument, and the recordings are handed to the vendor for the call only:
    nothing of them is kept or logged (RFC §17.6).
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def supports_design(self) -> bool:
        """Whether :meth:`design_voice` is offered."""
        return False

    @property
    def supports_replication(self) -> bool:
        """Whether :meth:`replicate_voice` is offered."""
        return False

    async def design_voice(
        self, description: str, *, store: bool = True, name: str | None = None
    ) -> CustomVoice:
        """A voice written from a natural-language description.

        Args:
            description: Who speaks and how (e.g. ``"a warm Montreal narrator
                in her forties, unhurried, a smile in the voice"``).
            store: Ask the vendor to keep the voice for the account.
            name: Display name for a stored voice.

        Raises:
            NotImplementedError: The library does not design voices.
        """
        raise NotImplementedError(f"{self.name} does not design voices")

    async def replicate_voice(
        self,
        sample: VoiceAudio,
        consent: VoiceAudio,
        *,
        store: bool = True,
        name: str | None = None,
    ) -> CustomVoice:
        """A person's voice, from a recording of it and their recorded consent.

        Args:
            sample: The recording of the voice to replicate.
            consent: A recording, by the same person, agreeing to it.
            store: Ask the vendor to keep the voice for the account.
            name: Display name for a stored voice.

        Raises:
            NotImplementedError: The library does not replicate voices.
            VoiceConsentError: The vendor refused the consent.
        """
        raise NotImplementedError(f"{self.name} does not replicate voices")

    @abstractmethod
    async def get_voice(self, voice_id: str) -> CustomVoice | None:
        """The stored voice *voice_id*, or ``None`` if the vendor holds none."""

    @abstractmethod
    async def delete_voice(self, voice_id: str) -> None:
        """Delete a stored voice; an id the vendor does not hold is not an error."""

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses if needed."""
