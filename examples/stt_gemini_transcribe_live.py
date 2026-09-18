"""RoomKit - live speech-to-text on Gemini 3.5 Transcribe.

    GEMINI_API_KEY=... uv run --extra realtime-gemini \
        python examples/stt_gemini_transcribe_live.py [recording.wav]

``gemini-3.5-transcribe-live`` is a dedicated recogniser reached over the Live
API: it takes a PCM stream and answers with interim transcripts while the
speaker is still talking, then a final one per utterance. That is the opposite
of :mod:`roomkit.voice.stt.gemini`, which sends a finished recording to a
multimodal model and waits for one answer, and the two are for different jobs
rather than being two ways to do the same one.

With no argument this synthesizes a short sentence with Gemini TTS, so the
example runs on one API key. Pass a path to transcribe your own 16-bit mono PCM
WAV file at any rate: it is resampled to the 16 kHz the model takes.

What you should see: several `~` lines (interim, revised as the model hears
more) followed by `=` lines (final). Seeing only `=` lines means the audio
arrived faster than real time, which is normal for a file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import base64
import wave
from collections.abc import AsyncIterator

from shared import require_env, setup_logging

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk
from roomkit.voice.pipeline import LinearResamplerProvider
from roomkit.voice.stt.gemini_transcribe import (
    REQUIRED_SAMPLE_RATE,
    GeminiTranscribeConfig,
    GeminiTranscribeProvider,
)
from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider

logger = setup_logging("roomkit.examples.stt_gemini_transcribe_live")

CHUNK_MS = 100
SPOKEN = (
    "Bonjour, je teste la transcription en direct de RoomKit avec le modele Gemini 3.5 Transcribe."
)


def read_wav(path: Path) -> tuple[bytes, int]:
    """Read a mono PCM WAV file into raw frames plus its rate."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit mono PCM")
        return handle.readframes(handle.getnframes()), handle.getframerate()


async def synthesize(api_key: str) -> tuple[bytes, int]:
    """Speak one sentence so the example needs no recording of its own.

    Gemini TTS answers a base64 WAV data URI at 24 kHz; the 44-byte header is
    dropped to leave raw PCM.
    """
    tts = GeminiTTSProvider(GeminiTTSConfig(api_key=api_key))
    try:
        audio = await tts.synthesize(SPOKEN)
        pcm = base64.b64decode(audio.url.split(",", 1)[1])[44:]
        return pcm, 24000
    finally:
        await tts.close()


def to_required_rate(pcm: bytes, sample_rate: int) -> bytes:
    """Bring the audio to the rate the recogniser documents.

    The model takes 16 kHz, and Gemini TTS produces 24 kHz, so the example
    would otherwise demonstrate the provider's off-rate warning rather than
    its transcription. RoomKit's own resampler does the conversion, which is
    the same one a voice pipeline would apply upstream.
    """
    if sample_rate == REQUIRED_SAMPLE_RATE:
        return pcm
    frame = AudioFrame(data=pcm, sample_rate=sample_rate, channels=1, sample_width=2)
    resampled = LinearResamplerProvider().resample(
        frame, REQUIRED_SAMPLE_RATE, 1, 2, stream="example"
    )
    return resampled.data


async def paced_chunks(pcm: bytes, sample_rate: int) -> AsyncIterator[AudioChunk]:
    """Feed the recogniser in real time, so the interims are worth watching.

    Sending the whole file at once works and is faster, but the model then has
    everything before it answers and there is nothing partial left to see.
    """
    frame_bytes = int(sample_rate * CHUNK_MS / 1000) * 2
    for start in range(0, len(pcm), frame_bytes):
        yield AudioChunk(data=pcm[start : start + frame_bytes], sample_rate=sample_rate)
        await asyncio.sleep(CHUNK_MS / 1000)


async def main() -> None:
    env = require_env("GEMINI_API_KEY")
    api_key = env["GEMINI_API_KEY"]

    if len(sys.argv) > 1:
        pcm, sample_rate = read_wav(Path(sys.argv[1]))
        logger.info(
            "Transcribing %s (%d Hz, %.1f s)",
            sys.argv[1],
            sample_rate,
            len(pcm) / (sample_rate * 2),
        )
    else:
        logger.info("No file given: synthesizing one sentence with Gemini TTS")
        pcm, sample_rate = await synthesize(api_key)

    provider = GeminiTranscribeProvider(
        GeminiTranscribeConfig(
            api_key=api_key,
            # Empty is not a missing value: it asks the model to identify the
            # language itself, which is the point of the 85+ locale coverage.
            language_codes=[],
            custom_vocabulary=["RoomKit", "Gemini"],
        )
    )

    finals: list[str] = []
    try:
        async for result in provider.transcribe_stream(paced_chunks(pcm, sample_rate)):
            marker = "=" if result.is_final else "~"
            logger.info("%s %s", marker, result.text)
            if result.is_final:
                finals.append(result.text)
    finally:
        await provider.close()

    logger.info("Transcript: %s", " ".join(finals).strip() or "(nothing recognised)")


if __name__ == "__main__":
    asyncio.run(main())
