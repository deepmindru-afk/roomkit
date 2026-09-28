"""Speaker attribution from a diarizing STT, as a Voice Channel carries it (RFC §12.2.3).

A diarizing STT labels who said each segment of a final result. The labels
hold within one stream only, so the channel keeps one stream per session and
numbers each new one (the *epoch*); this module turns a label and its epoch
into what the room message carries, and decides when a label is a speaker
change. When the STT labels nothing, the pipeline's diarization stage can
name the utterance's speaker instead (:class:`PipelineSpeakerTally`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from roomkit.voice.base import speaker_label

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.voice")

_CLAIM_WAIT_S = 1.0

_NOBODY = ""
"""The tally's key for a voice the stage heard but matched to no speaker
(sherpa-onnx's ``"unknown"``): it competes like a label and, if heard the
longest, the transcript is :data:`UNKNOWN_SPEAKER`'s."""

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
    source: Literal["stt", "pipeline"] = "stt"

    @classmethod
    def of(
        cls, label: str | None, epoch: int, source: Literal["stt", "pipeline"] = "stt"
    ) -> SpeakerAttribution:
        return cls(label, epoch, default_sender_name(label, epoch), source)

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


class PipelineSpeakerTally:
    """Who the pipeline's diarization stage heard, by seconds of audio, per session.

    The stage marks each frame it identified
    (``frame.metadata["diarization"]["speaker_id"]``); every processed frame is
    added here, after the stage ran. A transcript's speaker is the one heard
    the longest since the count last closed (RFC §12.2.3, "With the pipeline
    stage"); a voice the stage matched to nobody counts too, as
    :data:`UNKNOWN_SPEAKER`. The stage's labels hold for the whole session, so
    the epoch is always 0.

    Behind a VAD the count closes on the SPEECH_END frame itself — the frame a
    stage such as sherpa-onnx's identifies on — and the pipeline fires its
    SPEECH_END callbacks *before* the stage sees that frame. So a speech end
    claims its speaker (:meth:`claim`), answered once the closing frame is
    counted, and awaits the answer (:func:`claimed_speaker`). Without a VAD a
    transcript takes the count as its final lands (:meth:`take`).

    The pipeline may run on a worker thread (``inbound_dsp_threads``), hence
    the lock and the thread-safe futures.
    """

    def __init__(self) -> None:
        self._seconds: dict[str, dict[str, float]] = {}
        self._claims: dict[str, Future[SpeakerAttribution | None]] = {}
        self._lock = threading.Lock()

    def add(self, session_id: str, frame: AudioFrame) -> None:
        verdict = _verdict(frame)
        with self._lock:
            if verdict is not None:
                heard = self._seconds.setdefault(session_id, {})
                heard[verdict] = heard.get(verdict, 0.0) + _seconds_of(frame)
            if not frame.metadata.get("vad_speech_end"):
                return
            speaker = _heard_longest(self._seconds.pop(session_id, None))
            claim = self._claims.pop(session_id, None)
        _answer(claim, speaker)

    def add_frame(self, session: Any, frame: AudioFrame) -> None:
        """The pipeline's processed-frame callback: ``(session, frame)``."""
        self.add(session.id, frame)

    def claim(self, session_id: str) -> Future[SpeakerAttribution | None]:
        """The speaker of the utterance ending now, answered by its closing frame."""
        claim: Future[SpeakerAttribution | None] = Future()
        with self._lock:
            stale = self._claims.pop(session_id, None)
            self._claims[session_id] = claim
        _answer(stale, None)
        return claim

    def take(self, session_id: str) -> SpeakerAttribution | None:
        """The speaker heard the longest since the last take, then a clean slate."""
        with self._lock:
            return _heard_longest(self._seconds.pop(session_id, None))

    def reset(self, session_id: str) -> None:
        """Forget the count; a claim still open is answered with nobody."""
        with self._lock:
            self._seconds.pop(session_id, None)
            claim = self._claims.pop(session_id, None)
        _answer(claim, None)


async def claimed_speaker(
    claim: Future[SpeakerAttribution | None] | None,
) -> SpeakerAttribution | None:
    """The claimed speaker, once the frame closing its utterance is counted.

    That frame is in the very pipeline pass that fired SPEECH_END, so the wait
    is the rest of that pass; the bound only matters if the pass failed midway.
    """
    if claim is None:
        return None
    answer = asyncio.wrap_future(claim)
    done, _ = await asyncio.wait({answer}, timeout=_CLAIM_WAIT_S)
    if not done:
        logger.warning("No pipeline speaker: the utterance's closing frame was never counted")
        return None
    return answer.result()


def _verdict(frame: AudioFrame) -> str | None:
    """The stage's verdict on a frame: a label, :data:`_NOBODY`, or ``None`` for none."""
    identified = frame.metadata.get("diarization")
    if not isinstance(identified, dict):
        return None
    label = speaker_label(identified.get("speaker_id"))
    return _NOBODY if label is None else label


def _seconds_of(frame: AudioFrame) -> float:
    width = max(frame.sample_width * frame.channels, 1)
    return len(frame.data) / (width * frame.sample_rate)


def _heard_longest(heard: dict[str, float] | None) -> SpeakerAttribution | None:
    if not heard:
        return None
    label = max(heard, key=lambda name: heard[name])
    return SpeakerAttribution.of(label or None, 0, source="pipeline")


def _answer(
    claim: Future[SpeakerAttribution | None] | None, speaker: SpeakerAttribution | None
) -> None:
    if claim is None:
        return
    # A waiter gone (its task cancelled) may have cancelled the future.
    with contextlib.suppress(InvalidStateError):
        claim.set_result(speaker)
