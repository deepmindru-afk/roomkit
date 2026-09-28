"""RoomKit -- make a Gemini voice from a description, or from your own recording.

Design: describe a voice in plain words, Google makes it (about 15 s), and the
TTS speaks with it by its id. Google stores every designed voice for a year;
this example deletes it at the end unless KEEP_VOICE=1.

Replication (optional): set SAMPLE_WAV and CONSENT_WAV to clone your own voice.
Both are 16-bit, 24 kHz mono WAV recordings of you: the sample, 10 to 30 s of
natural speech, and the consent, the statement printed below read word for
word. Google checks both; a refusal is raised as ``VoiceConsentError``. The
cloned voice is not stored (it expires after seven days).

Requirements:
    pip install roomkit[gemini]

Run with:
    GEMINI_API_KEY=... uv run python examples/gemini_voice_design.py

Environment variables:
    GEMINI_API_KEY  (required) Gemini API key
    DESCRIPTION     The voice to design (default: a warm Montreal narrator)
    VOICE_LANGUAGE  fr-CA (default), fr-FR or en-US
    KEEP_VOICE      1 to keep the designed voice
    SAMPLE_WAV      Your voice sample, to replicate it
    CONSENT_WAV     Your consent recording
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import env_bool, require_env, setup_logging, voice_language

from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider
from roomkit.voice.tts.gemini_library import (
    CONSENT_STATEMENTS,
    GeminiVoiceLibrary,
    GeminiVoiceLibraryConfig,
)
from roomkit.voice.tts.library import VoiceConsentError

logger = setup_logging("roomkit.examples.gemini_voice_design")

DEFAULT_DESCRIPTION = (
    "Une narratrice québécoise dans la quarantaine, chaleureuse, débit posé, "
    "un léger sourire dans la voix."
)
LINE = "Bonjour, bienvenue chez RoomKit. Comment puis-je vous aider aujourd'hui?"


async def speak(tts: GeminiTTSProvider, voice_id: str, out: Path, name: str) -> None:
    audio = await tts.synthesize(LINE, voice=voice_id)
    path = out / f"{name}.wav"
    path.write_bytes(base64.b64decode(audio.url.split(",", 1)[1]))
    logger.info("  %s speaks: %s", voice_id, path)


async def main() -> None:
    env = require_env("GEMINI_API_KEY")
    language = voice_language("fr-CA")
    library = GeminiVoiceLibrary(GeminiVoiceLibraryConfig(api_key=env["GEMINI_API_KEY"]))
    tts = GeminiTTSProvider(GeminiTTSConfig(api_key=env["GEMINI_API_KEY"], language=language))
    out = Path(tempfile.mkdtemp(prefix="roomkit_voice_design_"))

    logger.info("Designing a voice (about 15 s)...")
    designed = await library.design_voice(
        os.environ.get("DESCRIPTION", DEFAULT_DESCRIPTION), name="roomkit-example"
    )
    try:
        await speak(tts, designed.voice.id, out, "designed")
    finally:
        if env_bool("KEEP_VOICE", default=False):
            logger.info("Kept %s (expires %s)", designed.voice.id, designed.expires_at)
        else:
            await library.delete_voice(designed.voice.id)
            logger.info("Deleted %s", designed.voice.id)

    sample, consent = os.environ.get("SAMPLE_WAV"), os.environ.get("CONSENT_WAV")
    if not (sample and consent):
        logger.info("To clone your voice, record this statement word for word as CONSENT_WAV:")
        logger.info("  %s", CONSENT_STATEMENTS.get(language, CONSENT_STATEMENTS["en-US"]))
    else:
        try:
            cloned = await library.replicate_voice(Path(sample), Path(consent), store=False)
            await speak(tts, cloned.voice.id, out, "cloned")
        except VoiceConsentError as refused:
            logger.error("%s", refused)

    await tts.close()
    await library.close()


if __name__ == "__main__":
    asyncio.run(main())
