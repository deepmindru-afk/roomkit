"""Fish Audio Drama 3: a plain line, a directed one, a fed stream and a dialogue.

``FishAudioTTSProvider`` speaks with Fish Audio's hosted models; Drama 3
(``drama-3``, the default) is directed in the text itself. This example:

1. lists English voices named "Sarah";
2. streams a plain line and logs when the first audio arrived;
3. renders a directed line in expressive mode (``[calm]``, ``<whisper>``);
4. feeds sentences one by one, the way a Voice Channel hands an LLM reply over,
   on the streaming-input socket;
5. voices a two-speaker dialogue, each turn with its own direction;

and writes each to a WAV file in a fresh temporary directory.

API credit is separate from the Fish Audio platform's: a key without it is
refused (402).

Requires:
    pip install roomkit[fish-audio]

Environment variables:
    FISH_API_KEY          Fish Audio API key
    FISH_AUDIO_VOICE_A    (optional) first voice id, default the public voice "Sarah"
    FISH_AUDIO_VOICE_B    (optional) second voice id, default the public voice "Adrian"

Run with:
    uv run --extra fish-audio python examples/voice_fish_audio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncIterator

from shared import pcm_from_wav_url, require_env, setup_logging, wav_bytes

from roomkit.voice import AudioChunk, DialogueTurn
from roomkit.voice.tts import FishAudioTTSConfig, FishAudioTTSProvider

logger = setup_logging("voice_fish_audio")

RATE = 24000
PLAIN = "Hello! This is Drama 3 from Fish Audio, streaming through RoomKit."
DIRECTED = (
    "[calm] I understand, and I am sorry for the wait. [growing impatient] But this is "
    "the third time I explain it! [sigh] <whisper>Let us start over.</whisper>"
)
SENTENCES = [
    "Your order shipped on Tuesday.",
    "[cheerful] It should arrive by Friday, between nine and five.",
    "Is there anything else I can do for you?",
]


async def sentences() -> AsyncIterator[str]:
    """Sentences as an LLM reply produces them, a little apart."""
    for sentence in SENTENCES:
        await asyncio.sleep(0.3)
        yield sentence


async def streamed(chunks: AsyncIterator[AudioChunk], label: str) -> bytes:
    started = time.perf_counter()
    first: float | None = None
    pcm = bytearray()
    async for chunk in chunks:
        if chunk.data and first is None:
            first = time.perf_counter() - started
        pcm += chunk.data
    logger.info(
        "%s: %.1f s of audio, first audio after %.0f ms",
        label,
        len(pcm) / 2 / RATE,
        (first or 0) * 1000,
    )
    return bytes(pcm)


async def main() -> None:
    env = require_env("FISH_API_KEY")
    voice_a = os.environ.get("FISH_AUDIO_VOICE_A", "933563129e564b19a115bedd57b7406a")
    voice_b = os.environ.get("FISH_AUDIO_VOICE_B", "bf322df2096a46f18c579d0baa36f41d")
    plain = FishAudioTTSProvider(FishAudioTTSConfig(api_key=env["FISH_API_KEY"], voice=voice_a))
    directed = FishAudioTTSProvider(
        FishAudioTTSConfig(api_key=env["FISH_API_KEY"], voice=voice_a, expressive=True)
    )
    out = Path(tempfile.mkdtemp(prefix="roomkit_fish_audio_"))

    voices = await plain.list_voices(language="en", query="Sarah")
    logger.info("Voices named Sarah: %s", ", ".join(f"{v.name} ({v.id})" for v in voices[:5]))

    clips = {
        "plain": await streamed(plain.synthesize_stream(PLAIN), "Plain line"),
        "directed": await streamed(directed.synthesize_stream(DIRECTED), "Directed"),
        "fed": await streamed(
            directed.synthesize_stream_input(sentences()), "Fed sentence by sentence"
        ),
    }
    dialogue = await directed.synthesize_dialogue(
        [
            DialogueTurn(speaker="agent", text="Hello, how can I help you?", style="politely"),
            DialogueTurn(
                speaker="caller", text="My order still has not arrived!", style="annoyed"
            ),
            DialogueTurn(speaker="agent", text="Let me check that for you right away."),
        ],
        {"agent": voice_a, "caller": voice_b},
    )
    logger.info("Dialogue: %.1f s", dialogue.duration_seconds or 0)
    clips["dialogue"], _ = pcm_from_wav_url(dialogue.url)

    for name, pcm in clips.items():
        (out / f"{name}.wav").write_bytes(wav_bytes(pcm, RATE))
    logger.info("Wrote %s", ", ".join(str(out / f"{name}.wav") for name in clips))
    await plain.close()
    await directed.close()


if __name__ == "__main__":
    asyncio.run(main())
