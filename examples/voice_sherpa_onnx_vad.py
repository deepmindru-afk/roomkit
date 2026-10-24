"""RoomKit -- Neural VAD + Denoiser with sherpa-onnx.

Demonstrates SherpaOnnxVADProvider and (optionally) SherpaOnnxDenoiserProvider
with a local microphone. The neural VAD replaces simple energy thresholding
with a proper speech detection model, giving much better accuracy in noisy
environments. The GTCRN denoiser cleans up the audio before VAD processing.

Supported VAD models:
  - TEN-VAD:  fast, low latency, good for real-time voice assistants
  - Silero VAD: widely used, slightly heavier

Prerequisites:
    pip install roomkit[local-audio,sherpa-onnx]

Models (download once, while online):
    python -m roomkit.local_models download ten-vad gtcrn-simple
    python -m roomkit.local_models download silero-vad   # for VAD_MODEL=silero-vad

Run with VAD only:
    uv run python examples/voice_sherpa_onnx_vad.py

Run with VAD + Denoiser:
    DENOISE=1 uv run python examples/voice_sherpa_onnx_vad.py

Or with Silero:
    VAD_MODEL=silero-vad uv run python examples/voice_sherpa_onnx_vad.py

Environment variables:
    VAD_MODEL       VAD local model: ten-vad | silero-vad (default: ten-vad)
    VAD_THRESHOLD   Speech probability threshold 0-1 (default: 0.35)
    DENOISE         1 enables the GTCRN denoiser (gtcrn-simple) (default: 0)
    ROOMKIT_MODELS_DIR  Where the downloaded models live
                    (default: ~/.cache/roomkit/models)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    env_bool,
    require_local_models,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import AIChannel, ChannelCategory, HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.providers.ai import MockAIProvider
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig, SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt import MockSTTProvider
from roomkit.voice.tts import MockTTSProvider

logger = setup_logging("voice_sherpa_onnx_vad")


async def main() -> None:
    vad_model = os.environ.get("VAD_MODEL", "ten-vad")
    denoise = env_bool("DENOISE", default=False)
    require_local_models(vad_model, *(["gtcrn-simple"] if denoise else []))
    threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    sample_rate = 16000

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=24000,
        channels=1,
        block_duration_ms=20,
    )

    # --- Neural VAD -----------------------------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig.from_local_model(
            vad_model,
            threshold=threshold,
            silence_threshold_ms=500,
            min_speech_duration_ms=250,
            sample_rate=sample_rate,
        )
    )
    logger.info("VAD: sherpa-onnx (model=%s, threshold=%.2f)", vad_model, threshold)

    # --- Denoiser (optional, GTCRN via sherpa-onnx) ----------------------------
    denoiser = None
    if denoise:
        from roomkit.voice.pipeline import SherpaOnnxDenoiserConfig, SherpaOnnxDenoiserProvider

        denoiser = SherpaOnnxDenoiserProvider(
            SherpaOnnxDenoiserConfig.from_local_model("gtcrn-simple")
        )
        logger.info("Denoiser: sherpa-onnx GTCRN (model=gtcrn-simple)")

    # --- Pipeline config ------------------------------------------------------
    pipeline = AudioPipelineConfig(vad=vad, denoiser=denoiser)

    # --- STT + TTS (mock — swap with real providers for production) -----------
    stt = MockSTTProvider(transcripts=["Hello from sherpa-onnx VAD!"])
    tts = MockTTSProvider()

    # --- Channels -------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    ai = AIChannel(
        "ai",
        provider=MockAIProvider(responses=["Neural VAD detected your speech!"]),
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="vad-demo")
    await kit.attach_channel("vad-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started (neural VAD)")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended (neural VAD)")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        from roomkit import HookResult

        logger.info("Transcription: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        from roomkit import HookResult

        logger.info("AI says: %s", text)
        return HookResult.allow()

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("vad-demo", "voice")

    logger.info("")
    logger.info("Speak into your microphone!")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
