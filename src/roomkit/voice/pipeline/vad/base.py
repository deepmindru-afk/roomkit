"""Voice Activity Detection provider ABC."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.voice.pipeline.vad")


@unique
class VADEventType(StrEnum):
    """Types of VAD events."""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SILENCE = "silence"
    AUDIO_LEVEL = "audio_level"


@dataclass
class VADEvent:
    """Event produced by a VAD provider."""

    type: VADEventType
    """The type of VAD event."""

    audio_bytes: bytes | None = None
    """Speech audio.  Set on SPEECH_START (pre-roll buffer) and
    SPEECH_END (full accumulated speech including pre-roll)."""

    confidence: float | None = None
    """Confidence score (0.0 to 1.0)."""

    duration_ms: float | None = None
    """Duration in milliseconds (speech or silence)."""

    level_db: float | None = None
    """Audio level in dB (set on AUDIO_LEVEL)."""


@dataclass
class VADConfig:
    """Tuning applied to the pipeline's VAD provider (RFC §12.3.1).

    Every field defaults to ``None``: a field that is set replaces the
    provider's own value, a field left ``None`` keeps it, so the provider's
    defaults hold for everything not set here.
    """

    silence_threshold_ms: int | None = None
    """Milliseconds of silence before triggering SPEECH_END."""

    speech_pad_ms: int | None = None
    """Milliseconds of audio kept from before speech is detected."""

    min_speech_duration_ms: int | None = None
    """Minimum speech duration to trigger events."""

    extra: dict[str, object] = field(default_factory=dict)
    """Provider-specific settings, by the provider's own setting name."""

    def settings(self, provider: str, known: Collection[str]) -> dict[str, object]:
        """The settings to apply, ``extra`` included, checked against *known*.

        A named field that is set wins over the same key in ``extra``.

        Raises:
            ValueError: ``extra`` names a setting *provider* does not have.
        """
        unknown = sorted(set(self.extra) - set(known))
        if unknown:
            raise ValueError(f"{provider} has no VAD setting {', '.join(unknown)}")
        named = {
            "silence_threshold_ms": self.silence_threshold_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "min_speech_duration_ms": self.min_speech_duration_ms,
        }
        return {**self.extra, **{k: v for k, v in named.items() if v is not None}}


class VADProvider(ABC):
    """Abstract base class for Voice Activity Detection providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'silero', 'webrtc')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame, stream: str) -> VADEvent | None:
        """Process an audio frame and optionally return a VAD event.

        Args:
            frame: The audio frame to analyse.
            stream: Identity of the audio stream this frame belongs to. A
                provider keeps its state per stream: a voice session and a
                conference track are separate speakers, and letting one advance
                the other's detection state makes silence from one close the
                other's utterance.

        Returns:
            A VADEvent if a state transition occurred, else None.
        """
        ...

    def configure(self, config: VADConfig) -> None:
        """Apply the pipeline's ``vad_config`` (RFC §12.3.1).

        Called when the pipeline is built, before audio flows. The default
        cannot apply anything and says so rather than ignoring it silently.
        """
        logger.warning("%s does not support VADConfig; vad_config is ignored", self.name)

    def reset(self, stream: str) -> None:  # noqa: B027
        """Drop a stream's state.

        Called when the stream ends, so a long-running room does not accumulate
        the state of every speaker that ever joined.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
