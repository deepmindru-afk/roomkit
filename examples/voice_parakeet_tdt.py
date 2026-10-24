"""RoomKit -- Local voice assistant with NVIDIA Parakeet TDT STT.

Uses the NVIDIA Parakeet TDT 0.6B v3 model for high-accuracy offline
speech-to-text via sherpa-onnx.  Parakeet TDT is an offline model —
VAD segments speech, then each utterance is transcribed in a single
batch call.  It supports 25 European languages with automatic language
detection, punctuation, and capitalization.

Audio pipeline:

  Mic → [Denoiser] → VAD → Parakeet TDT STT → Local LLM → TTS → Speaker

Prerequisites:
    pip install roomkit[local-audio,vllm,sherpa-onnx]

Models (download once, while online; Parakeet is about 490 MB):
    python -m roomkit.local_models download parakeet-tdt-0.6b-v3-int8 ten-vad \\
        vits-piper-en_US-amy-low
    # with VAD_MODEL_TYPE=silero: silero-vad; with DENOISE=1: gtcrn-simple

Run:
    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    uv run python examples/voice_parakeet_tdt.py

Environment variables:
    LLM_MODEL       (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL    Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY     API key if server requires auth (default: none)
    LLM_MAX_TOKENS  Max response tokens (default: 256)
    VAD_MODEL_TYPE  VAD model: ten (ten-vad) | silero (silero-vad) (default: ten)
    VAD_THRESHOLD   Speech probability threshold 0-1 (default: 0.35)
    TTS_SPEAKER_ID  Speaker ID for multi-speaker models (default: 0)
    DENOISE         1 enables the GTCRN denoiser (gtcrn-simple) (default: 0)
    ONNX_PROVIDER   ONNX provider for STT/TTS: cpu | cuda (default: cpu)
    CONSOLE         1 shows the RoomKit console dashboard (default: 0)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    require_env,
    require_local_models,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)
from roomkit.voice.stt import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logger = setup_logging("voice_parakeet_tdt")


async def main() -> None:
    vad_model = f"{os.environ.get('VAD_MODEL_TYPE', 'ten')}-vad"
    denoise = os.environ.get("DENOISE", "0") == "1"
    require_local_models(
        "parakeet-tdt-0.6b-v3-int8",
        vad_model,
        "vits-piper-en_US-amy-low",
        *(["gtcrn-simple"] if denoise else []),
    )
    env = require_env("LLM_MODEL")

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    onnx_provider = os.environ.get("ONNX_PROVIDER", "cpu")
    sample_rate = 16000
    # Built first: the speaker plays at the voice's own rate (16 kHz for amy).
    tts_config = SherpaOnnxTTSConfig.from_local_model(
        "vits-piper-en_US-amy-low",
        speaker_id=int(os.environ.get("TTS_SPEAKER_ID", "0")),
        provider=onnx_provider,
    )

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_config.sample_rate,
        channels=1,
        block_duration_ms=20,
        mute_mic_during_playback=True,
    )

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig.from_local_model(
            vad_model,
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.35")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            sample_rate=sample_rate,
        )
    )

    # --- Denoiser (optional) --------------------------------------------------
    denoiser = None
    if denoise:
        denoiser = SherpaOnnxDenoiserProvider(
            SherpaOnnxDenoiserConfig.from_local_model("gtcrn-simple")
        )
        logger.info("Denoiser: GTCRN")

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(vad=vad, denoiser=denoiser)

    # --- STT: Parakeet TDT (offline, via sherpa-onnx) -------------------------
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig.from_local_model(
            "parakeet-tdt-0.6b-v3-int8",
            sample_rate=sample_rate,
            provider=onnx_provider,
        )
    )
    logger.info("STT: Parakeet TDT 0.6B v3 (offline, provider=%s)", onnx_provider)

    # --- TTS (sherpa-onnx VITS/Piper) -----------------------------------------
    tts = SherpaOnnxTTSProvider(tts_config)

    # --- LLM (local via OpenAI-compatible API) --------------------------------
    ai_provider = create_vllm_provider(
        VLLMConfig(
            model=env["LLM_MODEL"],
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("LLM_API_KEY", "none"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
        )
    )
    logger.info("LLM: %s", env["LLM_MODEL"])

    # --- Channels -------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=(
            "You are a friendly voice assistant. Keep your responses "
            "short and conversational — one or two sentences at most. "
            "Answer directly without thinking, reasoning, or internal monologue. "
            "/no_think"
        ),
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="parakeet-demo")
    await kit.attach_channel("parakeet-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended — transcribing with Parakeet TDT...")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You said: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Warmup ---------------------------------------------------------------
    logger.info("Loading Parakeet TDT model (~640 MB)...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded — ready!")

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("parakeet-demo", "voice")

    logger.info("")
    logger.info("Speak into your microphone. Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
