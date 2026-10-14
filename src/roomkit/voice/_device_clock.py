"""The audio devices' own clocks, as PortAudio's callbacks report them.

Browsers and LiveKit's console give WebRTC's echo canceller its render-to-capture
delay from these on every captured block (``set_stream_delay_ms``): how long
until the reference fed last is played, plus how long ago the captured block
was recorded. The nominal stream latencies PortAudio reports are no substitute:
on a MacBook they added up to 296 ms (the input's alone 243 ms) where the clocks
put the whole delay at 42 ms.
"""

from __future__ import annotations

import time
from typing import Any

CAPTURED_AT = "captured_at"
"""``AudioFrame.metadata`` key: ``time.monotonic()`` when the frame's first sample
was captured, from the input device's clock, when the host API reports it."""

_MAX_DEVICE_DELAY_S = 1.0
"""Beyond, a callback's timestamps are not read as a delay (a host API that
reports none gives zeros, or times from another clock)."""


def _delay(later: Any, earlier: Any) -> float | None:
    try:
        delay = float(later) - float(earlier)
    except (TypeError, ValueError):
        return None
    if float(later) <= 0 or float(earlier) <= 0 or not 0 <= delay < _MAX_DEVICE_DELAY_S:
        return None
    return delay


def playout_delay(time_info: Any) -> float | None:
    """Seconds from an output callback to the device playing its block's first
    sample (``outputBufferDacTime - currentTime``); ``None`` when unreported."""
    return _delay(
        getattr(time_info, "outputBufferDacTime", None), getattr(time_info, "currentTime", None)
    )


def capture_metadata(time_info: Any) -> dict[str, float]:
    """The metadata stamping an input callback's frame with ``CAPTURED_AT``:
    ``currentTime - inputBufferAdcTime`` before now; empty when unreported."""
    age = _delay(
        getattr(time_info, "currentTime", None), getattr(time_info, "inputBufferAdcTime", None)
    )
    return {} if age is None else {CAPTURED_AT: time.monotonic() - age}
