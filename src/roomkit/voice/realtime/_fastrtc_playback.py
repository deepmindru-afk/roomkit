"""PCM output pulled by the WebRTC sender, with one queue and one clock.

The generic FastRTC emit/decode queue cannot own realtime interruption: its
resampler and a frame sleeping in AudioCallback both retain old audio. This
FIFO feeds that callback directly, before aiortc's negotiated codec encoder.
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import time

import av
import numpy as np
from aiortc.mediastreams import MediaStreamError

logger = logging.getLogger("roomkit.voice.realtime.fastrtc_playback")


class _PCMPlayback:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate // 50
        self.frame_bytes = self.frame_samples * 2
        self._capacity = sample_rate * 2 * 5
        self._buffer = bytearray()
        self._space = asyncio.Event()
        self._space.set()
        self._closed = asyncio.Event()
        self._generation = 0
        self._next_at: float | None = None
        self._pts = 0
        self._waiting_since: float | None = None
        self._playing = False
        self._ended = False
        self._last_sample = 0
        self._fade_in = True
        self.underruns = 0

    @property
    def buffered_ms(self) -> float:
        return len(self._buffer) * 500 / self.sample_rate

    async def write(self, audio: bytes) -> None:
        if len(audio) % 2:
            raise ValueError("PCM16 audio must contain complete two-byte samples")
        generation = self._generation
        offset = 0
        while offset < len(audio) and not self._closed.is_set():
            if generation != self._generation:
                return
            available = self._capacity - len(self._buffer)
            if not available:
                self._space.clear()
                await self._space.wait()
                continue
            if not self._buffer and not self._playing:
                self._waiting_since = time.monotonic()
            count = min(available, len(audio) - offset)
            self._buffer.extend(audio[offset : offset + count])
            offset += count
            self._ended = False

    def end_response(self) -> None:
        self._ended = True

    def clear(self) -> None:
        self._generation += 1
        self._buffer.clear()
        self._space.set()
        self._playing = False
        self._ended = False
        self._waiting_since = None
        self._fade_in = True

    def close(self) -> None:
        self.clear()
        self._closed.set()

    async def recv(self) -> av.AudioFrame:
        if self._closed.is_set():
            raise MediaStreamError
        now = time.monotonic()
        target = self._next_at if self._next_at is not None else now
        if target > now:
            try:
                async with asyncio.timeout(target - now):
                    await self._closed.wait()
            except TimeoutError:
                pass
        if self._closed.is_set():
            raise MediaStreamError
        now = time.monotonic()
        self._next_at = max(target + 0.02, now)

        # Samples are taken AFTER the wait: clear() also invalidates a frame
        # whose sender is currently sleeping. No second queue can retain it.
        ready = self._playing or len(self._buffer) >= 2 * self.frame_bytes or self._ended
        if self._waiting_since is not None and now - self._waiting_since >= 0.04:
            ready = True  # A short response cannot wait indefinitely for more PCM.
        count = min(self.frame_bytes, len(self._buffer)) if ready else 0
        samples = np.zeros(self.frame_samples, dtype=np.int16)
        if count:
            samples[: count // 2] = np.frombuffer(bytes(self._buffer[:count]), dtype="<i2")
            del self._buffer[:count]
            self._space.set()
            self._playing = True
            if self._fade_in:
                n = min(count // 2, self.sample_rate // 200)
                samples[:n] = (samples[:n] * np.linspace(0, 1, n)).astype(np.int16)
                self._fade_in = False
        if count < self.frame_bytes and self._playing and not self._ended:
            self.underruns += 1
            if self.underruns == 1 or self.underruns % 50 == 0:
                logger.warning("WebRTC PCM playback underrun (total=%d)", self.underruns)

        if count < self.frame_bytes and ready:
            # Smooth only a real discontinuity, never every provider chunk.
            previous = int(samples[count // 2 - 1]) if count else self._last_sample
            n = min(self.frame_samples - count // 2, self.sample_rate // 200)
            samples[count // 2 : count // 2 + n] = np.linspace(previous, 0, n).astype(np.int16)
            self._playing = False
            self._fade_in = True
            self._waiting_since = now if self._buffer else None
        elif not count and self._last_sample:
            n = self.sample_rate // 200
            samples[:n] = np.linspace(self._last_sample, 0, n).astype(np.int16)
        self._last_sample = int(samples[-1])
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._pts += self.frame_samples
        return frame
