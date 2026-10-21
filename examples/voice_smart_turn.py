"""RoomKit -- Audio-native turn detection with smart-turn.

Demonstrates SmartTurnDetector for natural conversational turn-taking.
Instead of relying on silence duration to decide when the user is done
speaking, the smart-turn model analyzes prosody and intonation from raw
audio to distinguish mid-utterance pauses from actual turn completions.

This example uses a fully local stack:
  - sherpa-onnx for STT (streaming Zipformer) + TTS (VITS/Piper)
  - sherpa-onnx neural VAD (TEN-VAD)
  - smart-turn ONNX model for turn detection
  - Local LLM via any OpenAI-compatible server (Ollama, vLLM, etc.)

How it works:
  1. VAD detects speech start/end and collects the audio segment
  2. STT transcribes the speech to text
  3. SmartTurnDetector receives the raw audio (not text) and predicts
     whether the user's turn is complete based on prosodic cues
  4. If complete → route to AI. If incomplete → wait for more speech.
     Accumulated audio and text are combined when the turn completes.

Requirements:
    pip install roomkit[local-audio,vllm,sherpa-onnx,smart-turn]

Models (download once, while online):
    python -m roomkit.local_models download \\
        smart-turn-v3.2-cpu ten-vad zipformer-en-kroko-2025-08-06 vits-piper-en_US-amy-low

Run:
    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    uv run python examples/voice_smart_turn.py

Environment variables:
    --- Smart turn detection ---
    SMART_TURN_THRESHOLD  Completion probability threshold 0-1 (default: 0.5)
                          Lower values make the detector more eager to complete turns.
                          Higher values wait longer for natural turn endings.

    --- LLM ---
    LLM_MODEL             (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL          Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY           API key if the server requires auth (default: none)
    LLM_MAX_TOKENS        Max response tokens (default: 256)
    SYSTEM_PROMPT         Custom system prompt

    --- Speech (sherpa-onnx local models) ---
    STT_MODEL             Local STT model (default: zipformer-en-kroko-2025-08-06;
                          python -m roomkit.local_models list --kind sherpa-onnx-stt)
    TTS_MODEL             Local TTS model (default: vits-piper-en_US-amy-low;
                          python -m roomkit.local_models list --kind sherpa-onnx-tts)

    --- VAD (sherpa-onnx) ---
    VAD_THRESHOLD         Speech probability threshold 0-1 (default: 0.35)

    --- Other ---
    ROOMKIT_MODELS_DIR    Where the downloaded models live (default: ~/.cache/roomkit/models)
    CONSOLE               1 shows the RoomKit console dashboard (default: 0)

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

from roomkit import ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logger = setup_logging("voice_smart_turn")

SMART_TURN = "smart-turn-v3.2-cpu"
VAD_MODEL = "ten-vad"


async def main() -> None:
    env = require_env("LLM_MODEL")
    stt_model = os.environ.get("STT_MODEL", "zipformer-en-kroko-2025-08-06")
    tts_model = os.environ.get("TTS_MODEL", "vits-piper-en_US-amy-low")
    require_local_models(SMART_TURN, VAD_MODEL, stt_model, tts_model)

    kit = RoomKit()
    console_cleanup = setup_console(kit)
    sample_rate = 16000
    tts_config = SherpaOnnxTTSConfig.from_local_model(tts_model)
    tts_sample_rate = tts_config.sample_rate

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_sample_rate,
        channels=1,
        block_duration_ms=20,
    )

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig.from_local_model(
            VAD_MODEL,
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.35")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            sample_rate=sample_rate,
            provider="cpu",
        )
    )

    # --- Smart turn detector --------------------------------------------------
    smart_turn_threshold = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))
    turn_detector = SmartTurnDetector(
        SmartTurnConfig.from_local_model(SMART_TURN, threshold=smart_turn_threshold)
    )
    logger.info(
        "Turn detection: smart-turn (model=%s, threshold=%.2f)",
        SMART_TURN,
        smart_turn_threshold,
    )

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        turn_detector=turn_detector,
    )

    # --- STT (sherpa-onnx) ----------------------------------------------------
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig.from_local_model(stt_model, sample_rate=sample_rate)
    )

    # --- TTS (sherpa-onnx) ----------------------------------------------------
    tts = SherpaOnnxTTSProvider(tts_config)

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
    await kit.create_room(room_id="smart-turn-demo")
    await kit.attach_channel("smart-turn-demo", "ai", category=ChannelCategory.INTELLIGENCE)

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

    @kit.hook(HookTrigger.ON_TURN_COMPLETE, execution=HookExecution.ASYNC)
    async def on_turn_complete(event, ctx):
        logger.info(
            "Turn COMPLETE (confidence=%.2f): %s",
            event.confidence,
            event.text,
        )

    @kit.hook(HookTrigger.ON_TURN_INCOMPLETE, execution=HookExecution.ASYNC)
    async def on_turn_incomplete(event, ctx):
        logger.info(
            "Turn INCOMPLETE (confidence=%.2f) — waiting for more: %s",
            event.confidence,
            event.text,
        )

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Warmup ---------------------------------------------------------------
    logger.info("Loading models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded — ready!")

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("smart-turn-demo", "voice")

    logger.info("")
    logger.info("Smart turn detection active!")
    logger.info("Try speaking with natural pauses — the detector will wait")
    logger.info("for you to finish your thought before responding.")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
