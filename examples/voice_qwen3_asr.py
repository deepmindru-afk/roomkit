"""RoomKit -- Voice assistant with Qwen3-ASR speech recognition.

Uses Qwen3-ASR for high-quality local speech recognition with automatic
language detection, a local LLM for the replies and sherpa-onnx (VITS/Piper)
to speak them: a complete voice assistant with state-of-the-art local ASR.

Audio flow:
    Mic -> [Pipeline] -> VAD -> Qwen3-ASR (STT) -> LLM -> sherpa-onnx TTS -> Speaker

Requirements:
    pip install roomkit[local-audio,vllm,sherpa-onnx,qwen-asr]

    System dependencies:
    - CUDA GPU with 3-5GB VRAM (for Qwen3-ASR-0.6B model)
    - For the 4B model, 8GB+ VRAM is recommended

    A local LLM server:
      Ollama: ollama pull qwen3:8b && ollama serve

    VAD model:
      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx

    TTS model (VITS/Piper voice):
      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
      tar xf vits-piper-en_US-amy-low.tar.bz2

Run:
    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    VAD_MODEL=ten-vad.onnx \\
    TTS_MODEL=vits-piper-en_US-amy-low/en_US-amy-low.onnx \\
    TTS_TOKENS=vits-piper-en_US-amy-low/tokens.txt \\
    TTS_DATA_DIR=vits-piper-en_US-amy-low/espeak-ng-data \\
    uv run python examples/voice_qwen3_asr.py

Environment variables:
    --- Qwen3-ASR ---
    ASR_MODEL_ID        HuggingFace model ID (default: Qwen/Qwen3-ASR-0.6B)
    ASR_BACKEND         Inference backend: transformers | vllm (default: transformers)
    ASR_DEVICE_MAP      Torch device mapping (default: auto)
    ASR_DTYPE           Model dtype: bfloat16, float16, float32 (default: bfloat16)
    ASR_LANGUAGE        Language code, e.g. en, zh (default: auto-detect)

    --- LLM ---
    LLM_MODEL           (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL        Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY         API key if needed (default: none)
    LLM_MAX_TOKENS      Max response tokens (default: 256)
    SYSTEM_PROMPT       Custom system prompt

    --- TTS (sherpa-onnx) ---
    TTS_MODEL           (required) Path to VITS/Piper .onnx model
    TTS_TOKENS          (required) Path to TTS tokens.txt
    TTS_DATA_DIR        Path to TTS data directory (espeak-ng-data)
    TTS_SAMPLE_RATE     Output sample rate of the voice (default: 22050)

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           (required) Path to VAD .onnx model
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)

    --- Other ---
    CONSOLE             1 shows the RoomKit console dashboard (default: 0)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, run_until_stopped, setup_console, setup_logging

from roomkit import ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.qwen3 import Qwen3ASRConfig, Qwen3ASRProvider
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logger = setup_logging("voice_qwen3_asr")


async def main() -> None:
    env = require_env("LLM_MODEL", "VAD_MODEL", "TTS_MODEL", "TTS_TOKENS")

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    sample_rate = 16000
    tts_sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "22050"))

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_sample_rate,
        channels=1,
        block_duration_ms=20,
    )

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad_model = env["VAD_MODEL"]
    vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=vad_model,
            threshold=vad_threshold,
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            sample_rate=sample_rate,
            provider="cpu",
        )
    )
    logger.info("VAD: sherpa-onnx (threshold=%.2f, model=%s)", vad_threshold, vad_model)

    pipeline_config = AudioPipelineConfig(vad=vad)

    # --- STT (Qwen3-ASR) -----------------------------------------------------
    asr_language = os.environ.get("ASR_LANGUAGE") or None
    stt_config = Qwen3ASRConfig(
        model_id=os.environ.get("ASR_MODEL_ID", "Qwen/Qwen3-ASR-0.6B"),
        backend=os.environ.get("ASR_BACKEND", "transformers"),
        device_map=os.environ.get("ASR_DEVICE_MAP", "auto"),
        dtype=os.environ.get("ASR_DTYPE", "bfloat16"),
        language=asr_language,
    )
    stt = Qwen3ASRProvider(stt_config)
    logger.info(
        "STT: Qwen3-ASR (model=%s, backend=%s, language=%s)",
        stt_config.model_id,
        stt_config.backend,
        stt_config.language or "auto-detect",
    )

    # --- TTS (sherpa-onnx) ----------------------------------------------------
    tts = SherpaOnnxTTSProvider(
        SherpaOnnxTTSConfig(
            model=env["TTS_MODEL"],
            tokens=env["TTS_TOKENS"],
            data_dir=os.environ.get("TTS_DATA_DIR", ""),
            sample_rate=tts_sample_rate,
        )
    )
    logger.info("TTS: sherpa-onnx (model=%s, rate=%d)", env["TTS_MODEL"], tts_sample_rate)

    # --- LLM (local via OpenAI-compatible API) --------------------------------
    llm_model = env["LLM_MODEL"]
    llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    ai_provider = create_vllm_provider(
        VLLMConfig(
            model=llm_model,
            base_url=llm_base_url,
            api_key=os.environ.get("LLM_API_KEY", "none"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
        )
    )
    logger.info("LLM: %s (base_url=%s)", llm_model, llm_base_url)

    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly voice assistant. Keep your responses "
        "short and conversational — one or two sentences at most. "
        "Answer directly without thinking, reasoning, or internal monologue. "
        "/no_think",
    )

    # --- Channels -------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="qwen3-asr")
    await kit.attach_channel("qwen3-asr", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You said: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Warmup: pre-load model -----------------------------------------------
    logger.info("Loading Qwen3-ASR and TTS models (may take a moment on first run)...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Model loaded — ready!")

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("qwen3-asr", "voice")

    logger.info("")
    logger.info(
        "Qwen3-ASR active — model=%s, language=%s",
        stt_config.model_id,
        stt_config.language or "auto-detect",
    )
    logger.info("Speak into your microphone. Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
