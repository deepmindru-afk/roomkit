"""The PCM16 sums of squares the voice paths share: stdlib, and NumPy for extras (RMK-796)."""

from __future__ import annotations

import math
import random
import struct
from collections.abc import Callable

import pytest

from roomkit.voice.utils import (
    pcm16_energy,
    pcm16_energy_numpy,
    pcm16_has_activity,
    pcm16_rms,
)


def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _generator_energy(data: bytes) -> int:
    """Reference oracle: the per-sample generator sum."""
    count = len(data) // 2
    return sum(s * s for s in struct.unpack(f"<{count}h", data[: count * 2]))


@pytest.mark.parametrize("count", [160, 320, 960])
def test_energy_is_the_exact_sum_the_generator_gave(count: int) -> None:
    rng = random.Random(count)
    samples = [rng.randint(-32768, 32767) for _ in range(count - 5)]
    data = _pcm(*samples, -32768, 32767, 0, -1, 1)

    assert pcm16_energy(data) == _generator_energy(data)
    assert pcm16_energy_numpy(data) == _generator_energy(data)
    assert type(pcm16_energy(data)) is int
    assert type(pcm16_energy_numpy(data)) is int


@pytest.mark.parametrize("energy", [pcm16_energy, pcm16_energy_numpy])
def test_full_scale_frames_do_not_overflow(energy: Callable[[bytes], int]) -> None:
    assert energy(_pcm(*[-32768] * 48000)) == 48000 * 32768**2


@pytest.mark.parametrize("energy", [pcm16_energy, pcm16_energy_numpy])
def test_samples_are_little_endian_whatever_the_host(energy: Callable[[bytes], int]) -> None:
    assert energy(b"\x00\x01") == 256**2  # 0x0100, not 0x0001


@pytest.mark.parametrize(
    ("data", "rms"),
    [
        (_pcm(0, 0, 0, 0), 0.0),
        (_pcm(100, 100, 100, 100), 100.0),
        (_pcm(300, -400), math.sqrt((300**2 + 400**2) / 2)),
        (b"", 0.0),
        (b"\x7f", 0.0),  # no whole sample
        (_pcm(200, 200) + b"\x00", 200.0),  # a trailing odd byte is ignored
    ],
    ids=["silence", "constant", "mixed-sign", "empty", "one-byte", "odd-length"],
)
def test_rms(data: bytes, rms: float) -> None:
    assert pcm16_rms(data) == pytest.approx(rms)
    assert pcm16_energy(data) == pcm16_energy_numpy(data) == _generator_energy(data)


def test_activity_threshold_is_minus_60_dbfs() -> None:
    # -60 dBFS is an RMS of 32.768: 33 crosses it, 32 does not.
    assert pcm16_has_activity(_pcm(*[33] * 160))
    assert not pcm16_has_activity(_pcm(*[32] * 160))
    assert not pcm16_has_activity(_pcm(8, -8, 8, -8))  # decoded G.711 silence
    assert not pcm16_has_activity(b"")
    assert not pcm16_has_activity(b"\x7f")
