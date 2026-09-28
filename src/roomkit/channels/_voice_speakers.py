"""Speaker attribution from a diarizing STT, as a Voice Channel carries it (RFC §12.2.3).

A diarizing STT labels who said each segment of a final result. The labels
hold within one stream only, so the channel keeps one stream per session and
numbers each new one (the *epoch*); this module turns a label and its epoch
into what the room message carries, and decides when a label is a speaker
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

UNKNOWN_SPEAKER = "Unknown speaker"
"""The name of words the STT attributed to nobody. Without a name the AI
channel would fall back to the stream owner's, for words nobody said as them."""


def default_sender_name(label: str | None, epoch: int) -> str:
    """``"Speaker A"`` in the first stream, ``"Speaker A#1"`` after a reconnect.

    The epoch is in the name from the second stream on, so the same letter
    from two streams never reads as one person to the model.
    """
    if label is None:
        return UNKNOWN_SPEAKER
    return f"Speaker {label}" if epoch == 0 else f"Speaker {label}#{epoch}"


@dataclass(frozen=True)
class SpeakerAttribution:
    """Who said a routed transcript: the STT's label, its epoch, the room name.

    ``sender_name`` starts as :func:`default_sender_name` and is what an
    ``ON_TRANSCRIPTION`` hook left, a label matched to a known voice.
    """

    label: str | None
    epoch: int
    sender_name: str

    @classmethod
    def of(cls, label: str | None, epoch: int) -> SpeakerAttribution:
        return cls(label, epoch, default_sender_name(label, epoch))

    def same_voice(self, other: SpeakerAttribution) -> bool:
        """Same label in the same stream, whatever name a hook gave either."""
        return (self.label, self.epoch) == (other.label, other.epoch)

    def metadata(self) -> dict[str, Any]:
        """The keys the inbound message carries; no label for unattributed words."""
        meta: dict[str, Any] = {"sender_name": self.sender_name, "speaker_epoch": self.epoch}
        if self.label is not None:
            meta["speaker_label"] = self.label
        return meta


class SpeakerTracker:
    """The speaker-change rule for one session, fed each routed segment in order.

    A segment with no label neither fires nor resets. A label fires when it
    differs from the last one routed in the same epoch, and on the first
    labelled segment of each epoch.
    """

    def __init__(self) -> None:
        self._epoch: int | None = None
        self._last: str | None = None
        self._seen: set[str] = set()

    def observe(self, label: str | None, epoch: int) -> bool | None:
        """``None`` when no change fires; otherwise whether the speaker is new."""
        if label is None:
            return None
        if epoch != self._epoch:
            self._epoch, self._last, self._seen = epoch, None, set()
        is_new = label not in self._seen
        self._seen.add(label)
        if label == self._last:
            return None
        self._last = label
        return is_new
