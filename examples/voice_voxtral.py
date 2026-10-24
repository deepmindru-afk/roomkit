"""RoomKit -- a voice agent on Mistral end to end: talk to it through your microphone.

    Voxtral Mini Transcribe Realtime 3  hears you  (VoxtralSTTProvider)
    Mistral Small                       answers    (MistralAIProvider)
    Voxtral Speech 3                    speaks     (VoxtralTTSProvider)

The realtime model detects the turns itself, so the pipeline runs no VAD: the
channel streams all audio (continuous mode), each turn the model ends is a
final, and your words, as they arrive, can cut the agent off. The example
runs Mistral's faster turn preset with a 240 ms streaming delay: the final
comes about 0.7 s after you stop, and a short pause ends your turn more often
(a pause after a question can make the agent answer twice). With Mistral's
defaults, ``TURNS=default STREAMING_DELAY_MS=480``, it comes about 1.7 s after
(measured 2026-10-21).

One Mistral API key serves the three models.

Requirements:
    pip install roomkit[mistral-voice,mistral,local-audio] aec-audio-processing

Environment variables:
    MISTRAL_API_KEY       (required) Mistral API key

    --- Voice (optional) ---
    VOICE_LANGUAGE        Language to transcribe, e.g. fr (default: detected)
    VOICE                 Voxtral voice (default: voice_478, Sheila, US English;
                          voice_3137 is Manon, French)
    MODEL                 Mistral chat model (default: mistral-small-latest)

    --- Audio (optional) ---
    TURNS                 aggressive | default (default: aggressive)
    STREAMING_DELAY_MS    How long the model listens past a word, 80-1200 or
                          2400 (default: 240)
    AEC                   webrtc | speex | 0 (default: webrtc)
    INTERRUPTION          semantic | words | confirmed | immediate | disabled
                          (default: semantic: an "okay" does not stop the agent)

Run with:
    MISTRAL_API_KEY=... uv run --extra mistral-voice --extra mistral \\
        --extra local-audio --extra webrtc-aec python examples/voice_voxtral.py

Use headphones, or keep AEC on: without echo cancellation the agent hears
itself. Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    build_aec,
    build_interruption,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
    voice_language,
)

from roomkit import AIChannel, ChannelCategory, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.providers.mistral import MistralAIProvider, MistralConfig
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt import VoxtralSTTConfig, VoxtralSTTProvider, VoxtralTurnDetection
from roomkit.voice.tts import StripEmoji, VoxtralTTSConfig, VoxtralTTSProvider

logger = setup_logging("voice_voxtral")

SAMPLE_RATE = 16000
"""Microphone rate: the realtime model's default, sent as is."""

OUTPUT_RATE = 24000
"""Voxtral Speech renders 24 kHz."""


def turn_detection() -> VoxtralTurnDetection:
    """The turn settings ``TURNS`` names: Mistral's faster preset, or the service's defaults."""
    turns = os.environ.get("TURNS", "aggressive").lower()
    if turns == "aggressive":
        return VoxtralTurnDetection.aggressive()
    if turns != "default":
        sys.exit(f"TURNS must be aggressive or default, got {turns!r}")
    return VoxtralTurnDetection()


async def main() -> None:
    env = require_env("MISTRAL_API_KEY")
    api_key = env["MISTRAL_API_KEY"]

    # --- Audio: local mic + speakers, AEC so the agent does not hear itself ----
    aec = build_aec(SAMPLE_RATE)
    backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=OUTPUT_RATE,
        channels=1,
        block_duration_ms=20,
        aec=aec,
        mute_mic_during_playback=aec is None,
    )
    # No VAD: the realtime model finds the turns (continuous mode). No aec= on
    # the pipeline either: the backend feeds the echo reference itself.
    pipeline = AudioPipelineConfig()

    # --- Voxtral Transcribe Realtime: the model's own turn detection -----------
    language = voice_language(None)
    stt = VoxtralSTTProvider(
        VoxtralSTTConfig(
            api_key=api_key,
            language=language,
            streaming_delay_ms=int(os.environ.get("STREAMING_DELAY_MS", "240")),
            turn_detection=turn_detection(),
        )
    )

    # --- Voxtral Speech: each sentence rendered and streamed --------------------
    # The config's default voice applies when VOICE is unset.
    tts_voice = {"voice": os.environ["VOICE"]} if "VOICE" in os.environ else {}
    tts = VoxtralTTSProvider(VoxtralTTSConfig(api_key=api_key, **tts_voice))
    logger.info("Hearing in %s, speaking with %s", language or "any language", tts.default_voice)

    # --- Channels ---------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
        # The model adds emoji despite the prompt; spoken, they sound wrong.
        tts_filter=StripEmoji(),
        # The agent keeps talking through an "okay" and stops for anything else.
        interruption=build_interruption(),
    )
    ai = AIChannel(
        "ai",
        provider=MistralAIProvider(
            MistralConfig(
                api_key=api_key,
                model=os.environ.get("MODEL", "mistral-small-latest"),
                max_tokens=300,
                # A reasoning trace would come before every spoken answer.
                reasoning_effort="none",
            )
        ),
        system_prompt=(
            "You are a friendly voice assistant. Answer in the language the user "
            "spoke, in one or two short sentences, without markdown or lists."
        ),
    )

    kit = RoomKit()
    console_cleanup = setup_console(kit)
    kit.register_channel(voice)
    kit.register_channel(ai)
    await kit.create_room(room_id="voxtral-demo")
    await kit.attach_channel("voxtral-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You said: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("voxtral-demo", "voice")

    logger.info("")
    logger.info("Speak: Voxtral hears you, Mistral Small answers, Voxtral Speech speaks.")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
