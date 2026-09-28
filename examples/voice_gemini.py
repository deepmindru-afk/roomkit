"""RoomKit -- a voice assistant that is Gemini end to end, and acts its lines.

Talk to Gemini through your microphone; it answers through your speakers:

  Mic -> gemini-3.5-transcribe-live (STT) -> gemini-3.8-flash (LLM)
      -> gemini-3.8-flash-lite-tts (TTS) -> Speaker

One API key covers the three stages. The system prompt lets the model write
audio tags into its replies -- <laugh>, <sigh>, <short pause>, [whispers] --
and Gemini TTS performs them instead of reading them out, which is the part a
conventional TTS cannot do. Every line is logged as text too, tags included,
so you can read what the model asked for while you hear what the voice made of it.

How a turn works: there is no local VAD. The live recogniser decides when you
have finished a sentence and sends a final transcript; the room hands it to
the model, and the whole reply is spoken once it is written.

The mic is muted while the assistant speaks (the LocalAudioBackend default),
so plain speakers work without echo cancellation. The flip side: you cannot
interrupt it mid-sentence.

Speak French, English or any of the recogniser's 85+ languages: it detects
the language and the model answers in it. Set VOICE_LANGUAGE to pin one.

Requirements:
    pip install roomkit[gemini,local-audio]

Run with:
    GEMINI_API_KEY=... uv run python examples/voice_gemini.py

Environment variables:
    GEMINI_API_KEY    (required) Gemini API key
    VOICE_LANGUAGE    BCP-47 code for both STT and TTS, e.g. fr-FR
                      (default: detected from what you say)
    GEMINI_TTS_MODEL  gemini-3.8-flash-lite-tts (default, built for voice
                      agents) or gemini-3.8-flash-tts (more acting range,
                      about half a second slower to start)
    GEMINI_VOICE      Prebuilt voice (default: Kore)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, run_until_stopped, setup_logging, voice_language

from roomkit import AIChannel, ChannelCategory, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig
from roomkit.voice.backends.base import VoiceBackend
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.stt.gemini_transcribe import (
    REQUIRED_SAMPLE_RATE,
    GeminiTranscribeConfig,
    GeminiTranscribeProvider,
)
from roomkit.voice.tts.gemini import OUTPUT_SAMPLE_RATE, GeminiTTSConfig, GeminiTTSProvider

logger = setup_logging("roomkit.examples.voice_gemini")

ROOM_ID = "gemini-voice"

SYSTEM_PROMPT = """\
You are a warm, lively voice assistant. Everything you write is spoken aloud,
so write the way people talk: short sentences, no lists, no markdown, no emoji.
Answer in the language the user speaks. Keep replies to two or three sentences
unless asked for more.

Your voice can perform audio tags. Put one inline, right where the sound or the
change of delivery belongs, when the moment calls for it -- not in every reply:

  <laugh>  <chuckle>  <sigh>  <gasp>  <breath>
  <short pause>  <long pause>
  [whispers]  [excitedly]  [slowly]

Write the tags in English whatever language you answer in, exactly as above.
Example: "Oh, that one got me <laugh> Okay, [whispers] here is the secret..."
"""


def build_channels(api_key: str, backend: VoiceBackend) -> tuple[VoiceChannel, AIChannel]:
    """The two channels of the room: the voice, and the model behind it."""
    language = voice_language(None)

    stt = GeminiTranscribeProvider(
        GeminiTranscribeConfig(api_key=api_key, language_codes=[language] if language else [])
    )
    tts = GeminiTTSProvider(
        GeminiTTSConfig(
            api_key=api_key,
            model=os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts"),
            voice=os.environ.get("GEMINI_VOICE", "Kore"),
            language=language,
        )
    )
    # No tts_filter: StripBrackets would remove the [whispers]-style tags
    # before the TTS ever saw them.
    voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend)

    ai = AIChannel(
        "ai",
        provider=GeminiAIProvider(
            # "low" is the lowest thinking level 3.8 Flash takes: a voice reply
            # is not worth seconds of reasoning.
            GeminiConfig(api_key=api_key, model="gemini-3.8-flash", thinking_level="low")
        ),
        system_prompt=SYSTEM_PROMPT,
    )
    return voice, ai


async def main() -> None:
    env = require_env("GEMINI_API_KEY")

    kit = RoomKit()
    backend = LocalAudioBackend(
        # Capture at the rate the recogniser documents, play at the rate the
        # TTS answers: no resampling anywhere.
        input_sample_rate=REQUIRED_SAMPLE_RATE,
        output_sample_rate=OUTPUT_SAMPLE_RATE,
    )
    voice, ai = build_channels(env["GEMINI_API_KEY"], backend)
    kit.register_channel(voice)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def show_heard(event, ctx) -> HookResult:
        logger.info("you:    %s", event.text)
        return HookResult.allow()

    # The reply as written, tags included, just before the voice performs it.
    @kit.hook(HookTrigger.BEFORE_TTS)
    async def show_reply(text: str, ctx) -> HookResult:
        logger.info("gemini: %s", text)
        return HookResult.allow()

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "ai", category=ChannelCategory.INTELLIGENCE)
    # Attached last: attaching the voice channel opens the microphone.
    await kit.attach_channel(ROOM_ID, "voice")

    logger.info("Speak into your microphone, then pause. Ctrl+C to stop.")
    await run_until_stopped(kit)


if __name__ == "__main__":
    asyncio.run(main())
