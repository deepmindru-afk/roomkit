"""Vui hosted by fluxions.ai: the voices, a streamed reply, a whole render.

``FluxionsTTSProvider`` speaks with the Vui model Fluxions hosts, so it needs an
API key and no GPU. Each text is rendered on its own (no conversation context);
to generate replies inside the dialogue, run Vui locally with ``VuiTTSProvider``
(see ``voice_vui_context.py``).

This example lists the voices, streams one sentence with ``maeve`` (and logs
when the first audio arrived), renders another whole with ``abraham``, and
writes both to ``fluxions.wav`` in a fresh temporary directory.

Requires:
    pip install roomkit[fluxions]

Environment variables:
    FLUXIONS_API_KEY  Fluxions API key

Run with:
    uv run --extra fluxions python examples/voice_fluxions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import base64
import tempfile
import time
import wave

from shared import require_env, setup_logging

from roomkit.voice.tts.fluxions import SAMPLE_RATE, FluxionsTTSConfig, FluxionsTTSProvider

logger = setup_logging("voice_fluxions")

STREAMED = "Hello! This is Vui, hosted by Fluxions, streaming through RoomKit."
WHOLE = "And this one is rendered whole, then handed back as a WAV."


async def main() -> None:
    env = require_env("FLUXIONS_API_KEY")
    tts = FluxionsTTSProvider(FluxionsTTSConfig(api_key=env["FLUXIONS_API_KEY"], voice="maeve"))
    await tts.warmup()

    voices = await tts.list_voices()
    logger.info("%d voices, among them: %s", len(voices), ", ".join(v.id for v in voices[:8]))

    started = time.perf_counter()
    first_audio: float | None = None
    streamed = bytearray()
    async for chunk in tts.synthesize_stream(STREAMED):
        if chunk.data and first_audio is None:
            first_audio = time.perf_counter() - started
        streamed += chunk.data
    logger.info(
        "Streamed %.1f s of audio, first audio after %.0f ms",
        len(streamed) / 2 / SAMPLE_RATE,
        (first_audio or 0) * 1000,
    )

    whole = await tts.synthesize(WHOLE, voice="abraham")
    logger.info("Rendered %.1f s whole with abraham", whole.duration_seconds or 0)
    whole_pcm = base64.b64decode(whole.url.split(",", 1)[1])[44:]

    path = Path(tempfile.mkdtemp(prefix="roomkit_fluxions_")) / "fluxions.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(bytes(streamed) + b"\x00\x00" * (SAMPLE_RATE // 4) + whole_pcm)
    logger.info("Wrote %s", path)
    await tts.close()


if __name__ == "__main__":
    asyncio.run(main())
