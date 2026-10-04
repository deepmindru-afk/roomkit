"""The streams whose session has ended, as the audio pipeline remembers them."""

from __future__ import annotations

import threading

# How many ended streams are remembered. A mark only has to outlive the frames
# still in flight and a backend's last deliveries, a matter of seconds; keeping
# one per session forever would itself leak.
_KEPT = 4096


class EndedStreams:
    """Streams whose session's end has begun, and whether it has been released yet.

    Work still arriving for a marked stream (a frame in flight on a DSP worker,
    a callback queued for the loop, a last TTS chunk) is abandoned, or leaves
    nothing behind once the stream is released. The oldest marks are forgotten
    beyond the bound.

    Thread-safe: a session can end on any thread, and DSP workers read the
    marks for every frame.
    """

    def __init__(self, kept: int = _KEPT) -> None:
        self._kept = kept
        self._released: dict[str, bool] = {}  # oldest first
        self._lock = threading.Lock()

    def mark(self, stream: str, *, released: bool) -> None:
        """Record how far a stream's end has gone, forgetting the oldest beyond the bound."""
        with self._lock:
            self._released.pop(stream, None)
            self._released[stream] = released
            while len(self._released) > self._kept:
                del self._released[next(iter(self._released))]

    def forget(self, stream: str) -> None:
        """Unmark a stream whose session is active again."""
        with self._lock:
            self._released.pop(stream, None)

    def released(self, stream: str) -> bool:
        """Whether a stream's state has been released since its session ended."""
        with self._lock:
            return self._released.get(stream, False)

    def __contains__(self, stream: object) -> bool:
        with self._lock:
            return stream in self._released
