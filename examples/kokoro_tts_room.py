"""RoomKit — a room that speaks whatever is written in it, with Kokoro.

Two channels share one room. You type into the CLI channel; the line enters
the room as an ordinary message, the room broadcasts it, and the voice channel
speaks it through the system speakers with Kokoro, on this machine::

    keyboard → CLIChannel → room → broadcast → VoiceChannel → Kokoro (sherpa-onnx) → speakers

Kokoro-82M (Apache 2.0) runs through sherpa-onnx, on the CPU: no GPU, no
network once the model is downloaded. Its v1.0 voices speak American and
British English, Spanish, French, Hindi, Italian, Japanese, Brazilian
Portuguese and Mandarin. A voice is named (``af_heart``, ``ff_siwis``…).

For text that is neither English nor Chinese, set ``KOKORO_LANG`` to its
espeak-ng language (``fr``, ``es``, ``it``…): without it the model reads the
text as English, and French comes out mispronounced. Pick a voice of the same
language, since a voice keeps its accent.

The voice channel runs with ``stt=None``: no microphone transcription, only
delivery. The local backend still opens the mic when the session starts —
audio simply goes nowhere without an STT provider.

Model (download once, while online)::

    python -m roomkit.local_models download kokoro-int8-multi-lang-v1_0

Requires:
    pip install roomkit[sherpa-onnx,local-audio]

Environment variables:
    KOKORO_LANG   espeak-ng language for text other than English or Chinese (default: none)
    KOKORO_VOICE  Voice name (default: af_heart). Some others in v1.0:
                  bf_alice, ef_dora, ff_siwis, if_sara, pf_dora
    ROOMKIT_MODELS_DIR  Where the downloaded models live
                  (default: ~/.cache/roomkit/models)

Run with:
    uv run --extra sherpa-onnx --extra local-audio python examples/kokoro_tts_room.py

    In French:
    KOKORO_LANG=fr KOKORO_VOICE=ff_siwis uv run --extra sherpa-onnx --extra local-audio \\
        python examples/kokoro_tts_room.py

Type a line and press Enter. Ctrl+D (or Ctrl+C) to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import os

from shared import require_local_models, setup_logging

from roomkit import CLIChannel, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.tts import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logger = setup_logging("kokoro_tts_room")

ROOM_ID = "kokoro-tts-room"
KOKORO = "kokoro-int8-multi-lang-v1_0"
SAMPLE_RATE = 24000  # Kokoro's rate


def kokoro_config() -> SherpaOnnxTTSConfig:
    return SherpaOnnxTTSConfig.from_local_model(
        KOKORO,
        voice=os.environ.get("KOKORO_VOICE", "af_heart"),
        lang=os.environ.get("KOKORO_LANG", ""),
    )


async def main() -> None:
    require_local_models(KOKORO)
    tts = SherpaOnnxTTSProvider(kokoro_config())
    await tts.warmup()  # loads the model before the first line, not on it

    kit = RoomKit()
    cli = CLIChannel("cli")
    voice = VoiceChannel(
        "voice",
        tts=tts,
        backend=LocalAudioBackend(output_sample_rate=SAMPLE_RATE),
    )
    kit.register_channel(cli)
    kit.register_channel(voice)

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def announce(text: str, ctx: object) -> HookResult:
        print(f"  ♪ speaking: {text}", flush=True)
        return HookResult.allow()

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "cli")
    # Attaching the voice channel auto-starts a local audio session, because
    # LocalAudioBackend declares auto_connect — that session is what the room
    # speaks into.
    await kit.attach_channel(ROOM_ID, "voice")

    try:
        await cli.run(
            kit,
            room_id=ROOM_ID,
            welcome="Type a line — Kokoro will say it. Ctrl+D to quit.",
        )
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
