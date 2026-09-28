"""RoomKit - who said what in a recording, with Deepgram's diarizer.

    DEEPGRAM_API_KEY=... uv run --extra deepgram --extra local-audio \\
        python examples/stt_deepgram_diarization.py [--seconds 20]

Records the microphone for a few seconds - take turns speaking with someone -
then sends the recording to Deepgram with ``diarize_model="latest"`` and prints
it as speaker turns. Deepgram labels each word; the provider groups them into
``TranscriptionResult.segments``, one per run of the same voice (RFC §12.2.3).

Batch is where Deepgram's diarizer is at its best: it sees the whole recording
before it decides who is who. Streaming (a ``VoiceChannel`` in continuous mode
takes the same provider) needs about thirty seconds of a conversation before it
tells the voices apart.

Pass ``--wav recording.wav`` to transcribe a file instead of the microphone.

Requires:
    pip install roomkit[deepgram,local-audio]

Environment variables:
    DEEPGRAM_API_KEY  (required) Deepgram API key
    STT_LANGUAGE      Language code (default: fr)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import asyncio
import os
import wave

from shared import require_env, setup_logging

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

logger = setup_logging("roomkit.examples.stt_deepgram_diarization")

SAMPLE_RATE = 16000


async def record(seconds: float) -> bytes:
    """``seconds`` of the default microphone, as 16 kHz mono PCM."""
    backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE, output_sample_rate=SAMPLE_RATE, block_duration_ms=20
    )
    captured = bytearray()

    def on_audio(_session: VoiceSession, frame: AudioFrame) -> None:
        captured.extend(frame.data)

    # Registered before start_listening: PortAudio captures the callback set
    # at stream start.
    backend.on_audio_received(on_audio)
    session = await backend.connect("diarization-demo", "speaker", "stt")
    await backend.start_listening(session)
    logger.info("Recording %.0f s - take turns speaking...", seconds)
    try:
        await asyncio.sleep(seconds)
    finally:
        await backend.stop_listening(session)
        await backend.disconnect(session)
        await backend.close()
    return bytes(captured)


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit mono PCM")
        return handle.readframes(handle.getnframes()), handle.getframerate()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker turns of a recording, by Deepgram.")
    parser.add_argument("--seconds", type=float, default=20.0, help="how long to record")
    parser.add_argument("--wav", type=Path, help="transcribe this 16-bit mono WAV instead")
    args = parser.parse_args()
    env = require_env("DEEPGRAM_API_KEY")

    if args.wav:
        pcm, rate = read_wav(args.wav)
    else:
        pcm, rate = await record(args.seconds), SAMPLE_RATE

    stt = DeepgramSTTProvider(
        DeepgramConfig(
            api_key=env["DEEPGRAM_API_KEY"],
            model="nova-3",
            language=os.environ.get("STT_LANGUAGE", "fr"),
            diarize_model="latest",
        )
    )
    result = await stt.transcribe(AudioChunk(data=pcm, sample_rate=rate))
    await stt.close()

    if not result.segments:
        logger.info("Nothing recognised.")
    for segment in result.segments:
        start = (segment.start_ms or 0) / 1000
        logger.info("[%5.1fs] Speaker %s: %s", start, segment.speaker or "?", segment.text)


if __name__ == "__main__":
    asyncio.run(main())
