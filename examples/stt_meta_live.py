"""RoomKit - live speech-to-text on Meta Muse Voice Transcribe.

    META_API_KEY=... uv run --extra meta-stt \\
        python examples/stt_meta_live.py [recording.wav] [--push-to-talk]

Streams a 16-bit mono PCM WAV file to ``muse-voice-transcribe-1.0`` in real
time, as a microphone would, then sends the same file once over the REST
endpoint. Any rate works: 16 and 24 kHz go through as they are, anything else
is resampled to 24 kHz by the provider.

With no file, it synthesizes a short French sentence with Gemini TTS first
(24 kHz), which needs GEMINI_API_KEY and ``--extra gemini`` as well.

Two stream modes, chosen by how a VoiceChannel is set up:

* default, ``ENDPOINTING`` — what a channel *without* a pipeline VAD uses: the
  model finds where each turn ends (about 550 ms of silence) and answers one
  final per turn, with ``*`` marking the moment it hears speech begin.
* ``--push-to-talk`` — what a channel *with* a pipeline VAD uses: the VAD
  delimits the utterance, and the model answers one final when it ends.

What you should see: a ``*`` (speech heard), ``~`` lines (interim, revised as
the model hears more), then ``=`` lines (final), and the REST transcript last.

Record a file with, for instance:
    arecord -f S16_LE -r 16000 -c 1 -d 10 recording.wav

Environment variables:
    META_API_KEY    (required) Meta Model API key
    GEMINI_API_KEY  (required without a WAV file) Gemini API key for the sample
    LANGUAGE_BIAS   Language name to bias toward, e.g. French (default: none)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import asyncio
import os
import wave
from collections.abc import AsyncIterator

from shared import pcm_from_wav_url, require_env, setup_logging

from roomkit.voice.base import AudioChunk
from roomkit.voice.stt.meta import MetaSTTConfig, MetaSTTProvider
from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider

logger = setup_logging("roomkit.examples.stt_meta_live")

CHUNK_MS = 40
# Trailing silence, so the model sees the end of the last turn before the
# stream ends — what a live microphone gives it for free.
TAIL_MS = 1500
SPOKEN = "Bonjour, je teste la transcription en direct de RoomKit avec le modèle de Meta."


def read_wav(path: Path) -> tuple[bytes, int]:
    """Read a mono PCM WAV file into raw frames plus its rate."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit mono PCM")
        return handle.readframes(handle.getnframes()), handle.getframerate()


async def synthesize(api_key: str) -> tuple[bytes, int]:
    """Speak one sentence so the example needs no recording of its own."""
    tts = GeminiTTSProvider(GeminiTTSConfig(api_key=api_key))
    try:
        audio = await tts.synthesize(SPOKEN)
        return pcm_from_wav_url(audio.url)
    finally:
        await tts.close()


async def paced_chunks(pcm: bytes, sample_rate: int) -> AsyncIterator[AudioChunk]:
    """Feed the audio in real time, so the interims are worth watching."""
    frame_bytes = int(sample_rate * CHUNK_MS / 1000) * 2
    audio = pcm + b"\x00" * (int(sample_rate * TAIL_MS / 1000) * 2)
    for start in range(0, len(audio), frame_bytes):
        yield AudioChunk(data=audio[start : start + frame_bytes], sample_rate=sample_rate)
        await asyncio.sleep(CHUNK_MS / 1000)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "wav",
        type=Path,
        nargs="?",
        help="16-bit mono PCM WAV file (default: a sentence synthesized with Gemini TTS)",
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="one final for the whole file, as behind a pipeline VAD",
    )
    args = parser.parse_args()

    env = require_env("META_API_KEY")
    if args.wav is not None:
        pcm, sample_rate = read_wav(args.wav)
        source = str(args.wav)
    else:
        gemini_key = require_env("GEMINI_API_KEY")["GEMINI_API_KEY"]
        logger.info("No file given: synthesizing one sentence with Gemini TTS")
        pcm, sample_rate = await synthesize(gemini_key)
        source = "the synthesized sentence"
    logger.info("Streaming %s (%d Hz, %.1f s)", source, sample_rate, len(pcm) / sample_rate / 2)

    bias = os.environ.get("LANGUAGE_BIAS")
    provider = MetaSTTProvider(
        MetaSTTConfig(
            api_key=env["META_API_KEY"],
            mode="PUSH_TO_TALK" if args.push_to_talk else "ENDPOINTING",
            language_bias=[bias] if bias else [],
        )
    )
    try:
        async for result in provider.transcribe_stream(paced_chunks(pcm, sample_rate)):
            if result.is_speech_start:
                logger.info("* speech")
            elif result.is_final:
                logger.info("= %s", result.text)
            else:
                logger.info("~ %s", result.text)

        batch = await provider.transcribe(AudioChunk(data=pcm, sample_rate=sample_rate))
        logger.info("REST: %s", batch.text or "(nothing recognised)")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
