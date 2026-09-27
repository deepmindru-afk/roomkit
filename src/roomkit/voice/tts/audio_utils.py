"""Shared audio utilities for TTS providers."""

from __future__ import annotations

import io
import struct
import wave
from typing import Any


def wrap_wav(pcm_data: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM S16LE data in a minimal WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


def wav_duration_seconds(wav: bytes) -> float:
    """Duration of a WAV file held in memory.

    Chunks other than ``fmt `` and ``data`` are skipped, so a file carrying
    metadata after its audio (a provenance manifest, say) measures its audio
    alone.

    Raises:
        ValueError: *wav* is not a readable WAV file.
    """
    try:
        with wave.open(io.BytesIO(wav), "rb") as reader:
            frames, rate = reader.getnframes(), reader.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"not a readable WAV file: {exc}") from exc
    if rate <= 0:
        raise ValueError(f"WAV file declares an invalid sample rate: {rate}")
    return frames / rate


def numpy_to_pcm_s16le(samples: Any) -> bytes:
    """Convert a numpy float32 array in [-1, 1] to PCM signed 16-bit LE bytes."""
    import numpy as np  # optional dependency

    arr = np.clip(samples, -1.0, 1.0)
    int_samples = (arr * 32767).astype(np.int16)
    return bytes(int_samples.tobytes())
