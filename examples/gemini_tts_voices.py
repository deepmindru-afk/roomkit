"""RoomKit -- find voices in Gemini's catalog and voice a dialogue with two of them.

Gemini's catalog holds about 2,000 voices, 68 of them Québec French. This
example searches it (``list_voices``), picks two ``fr-CA`` voices, and has them
play a short support call in one clip (``synthesize_dialogue``), each line with
its own direction. The WAV is written to a temporary folder.

Requirements:
    pip install roomkit[gemini]

Run with:
    GEMINI_API_KEY=... uv run python examples/gemini_tts_voices.py

Environment variables:
    GEMINI_API_KEY  (required) Gemini API key
    VOICE_LANGUAGE  Catalog language (default: fr-CA)
"""

from __future__ import annotations

import asyncio
import base64
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging, voice_language

from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider
from roomkit.voice.voices import DialogueTurn, filter_voices

logger = setup_logging("roomkit.examples.gemini_tts_voices")

SCRIPT = [
    ("Client", "Bonjour, j'appelle pour ma facture, elle me semble trop élevée.", None),
    ("Conseillère", "Je comprends. Laissez-moi vérifier ça avec vous.", "calme et rassurante"),
    ("Client", "D'accord, merci beaucoup.", "soulagé"),
]


async def main() -> None:
    env = require_env("GEMINI_API_KEY")
    language = voice_language("fr-CA")
    tts = GeminiTTSProvider(GeminiTTSConfig(api_key=env["GEMINI_API_KEY"], language=language))

    # One catalog read (three pages); the split by gender is local.
    voices = await tts.list_voices(language=language)
    women = filter_voices(voices, gender="female")
    men = filter_voices(voices, gender="male")
    logger.info("%s: %d female and %d male voices", language, len(women), len(men))
    for voice in women[:3] + men[:3]:
        logger.info("  %-18s %-10s %s", voice.id, voice.accent or "", voice.description or "")
    if not (women and men):
        logger.error("The catalog has no voice pair for %s", language)
        return

    turns = [DialogueTurn(speaker=who, text=text, style=style) for who, text, style in SCRIPT]
    audio = await tts.synthesize_dialogue(turns, {"Client": men[0].id, "Conseillère": women[0].id})
    await tts.close()

    path = Path(tempfile.mkdtemp(prefix="roomkit_voices_")) / "dialogue.wav"
    path.write_bytes(base64.b64decode(audio.url.split(",", 1)[1]))
    logger.info("Dialogue (%.1f s) written to %s", audio.duration_seconds or 0, path)


if __name__ == "__main__":
    asyncio.run(main())
