"""Shared audio utilities for the voice subsystem."""

from __future__ import annotations

import math
import struct
from typing import Any

_np: Any = None


def _get_np() -> Any:
    """Import numpy lazily, cached at module level.

    NumPy is an optional dependency (voice extras): base installs must
    import roomkit without it, and the DSP hot paths must not pay a
    per-call import.
    """
    global _np
    if _np is None:
        import numpy

        _np = numpy
    return _np


def rms_db(data: bytes) -> float:
    """Compute RMS level in dB from 16-bit little-endian PCM bytes.

    Returns a value in the range [-60.0, 0.0] where 0 dB is full scale
    and -60 dB is silence (or empty input).

    Vectorised with NumPy: this runs per audio chunk on the event loop,
    where a per-sample Python loop holds the GIL long enough to starve
    realtime RTP pacing.
    """
    n = len(data) // 2
    if n == 0:
        return -60.0
    np = _get_np()
    samples = np.frombuffer(data[: n * 2], dtype="<i2").astype(np.float64)
    rms = math.sqrt(float(np.mean(samples * samples))) / 32768.0
    if rms < 1e-10:
        return -60.0
    return max(-60.0, 20.0 * math.log10(rms))


def pcm16_energy(data: bytes) -> int:
    """Sum of the squared samples of 16-bit little-endian PCM.

    Standard library only, for the paths a base install runs per frame. A
    trailing odd byte is ignored. ``math.sumprod`` keeps the sum an exact
    int at about half the cost of a generator; NumPy only wins from about
    200 samples on, which is why ``rms_db`` keeps it for realtime chunks.
    """
    samples = struct.unpack_from(f"<{len(data) // 2}h", data)
    # Int inputs give an exact int; the stubs only know the float signature.
    return int(math.sumprod(samples, samples))


def pcm16_rms(data: bytes) -> float:
    """RMS amplitude of 16-bit little-endian PCM, 0.0 with no whole sample."""
    count = len(data) // 2
    return math.sqrt(pcm16_energy(data) / count) if count else 0.0


def pcm16_has_activity(data: bytes) -> bool:
    """Whether PCM16 exceeds -60 dBFS, without requiring voice/DSP extras.

    This detects output activity, not speech. G.711 decoded silence may be
    nonzero (+/-8); transport samples must still be forwarded and recorded.
    Compare mean-square energy to avoid a square root on each small chunk.
    """
    count = len(data) // 2
    return count > 0 and pcm16_energy(data) > count * (32768 * 0.001) ** 2
