"""Shared audio utilities for TTS providers."""

from __future__ import annotations

import base64
import io
import struct
import wave
from collections.abc import AsyncIterator
from typing import Any

from roomkit.models.event import AudioContent
from roomkit.voice.base import AudioChunk


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


async def collect_wav_content(
    chunks: AsyncIterator[AudioChunk], *, text: str, sample_rate: int
) -> AudioContent:
    """Collect a stream of 16-bit mono PCM chunks into a WAV data URL with its duration."""
    pcm = b"".join([chunk.data async for chunk in chunks])
    wav = wrap_wav(pcm, sample_rate)
    return AudioContent(
        url=f"data:audio/wav;base64,{base64.b64encode(wav).decode()}",
        mime_type="audio/wav",
        transcript=text,
        duration_seconds=len(pcm) / 2 / sample_rate,
    )


def streamed_format(output_format: str) -> str:
    """The format a streamed request asks for: raw ``pcm`` in place of ``wav``.

    A WAV stream opens with a RIFF header, which chunks declared ``pcm_s16le``
    would hand to the transport as audio: a click at the start of every
    sentence. A WAV belongs to ``synthesize()``, which returns a whole file.
    """
    return "pcm" if output_format == "wav" else output_format


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
