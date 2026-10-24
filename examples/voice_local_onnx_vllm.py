"""RoomKit -- Fully local voice assistant with sherpa-onnx + local LLM.

Everything runs locally — no cloud APIs required:
  - sherpa-onnx for speech-to-text (Zipformer transducer streaming, or Whisper offline)
  - Local LLM via any OpenAI-compatible server (Ollama, vLLM, LM Studio, etc.)
  - sherpa-onnx for text-to-speech (VITS/Piper models)
  - sherpa-onnx neural VAD (TEN-VAD or Silero)
  - Optional: sherpa-onnx GTCRN denoiser, WebRTC or Speex echo cancellation,
    smart-turn turn detection, WAV recording
  - All neural models support CUDA acceleration

Audio flows through the full pipeline:

  Mic → [Resampler] → [Recorder tap] → [AEC] → [Denoiser] → VAD
  → sherpa-onnx STT → Local LLM → sherpa-onnx TTS → [Recorder tap] → Speaker

Requirements:
    pip install roomkit[local-audio,vllm,sherpa-onnx]
    Optional: roomkit[webrtc-aec] for AEC=webrtc, roomkit[smart-turn] for
    SMART_TURN=1
    A local LLM server — pick one:
      Ollama:   ollama pull qwen3:8b && ollama serve
      vLLM:     vllm serve Qwen/Qwen3-8B --port 8000

    Recommended LLM by GPU VRAM:
      24GB+ VRAM  →  qwen3:30b  (MoE, only ~3B active — fast + best quality)
      12GB VRAM   →  qwen3:8b   (best balance for most setups)
      8GB VRAM    →  qwen3:4b   (fast, good quality, leaves room for ONNX CUDA)
      CPU only    →  qwen3:4b   (still usable, ~1-2s latency)

    System (optional): libspeexdsp (apt install libspeexdsp1) for AEC=speex

GPU acceleration (CUDA) for sherpa-onnx:
    The default sherpa-onnx pip package is CPU-only. For GPU support:

    1. Install cuDNN 9 (system library, once):
       # Add NVIDIA repo (Ubuntu 24.04 — change for your version)
       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
       sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
       sudo apt-get -y install cudnn9-cuda-12

    2. Install the CUDA 12 build of the sherpa-onnx version you have
       (1.12.26 or later): pick its `+cuda12.cudnn9...` wheel on
       https://k2-fsa.github.io/sherpa/onnx/cuda.html and install it with
       uv pip install "sherpa-onnx==<that version>" \\
           -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

    3. Set the env var:
       export ONNX_PROVIDER=cuda

    See https://www.roomkit.live/docs/guides/sherpa-onnx/ for troubleshooting.

Models (download once, while online):
    # VAD (TEN-VAD), STT (Kroko zipformer, streaming), TTS (Piper amy)
    python -m roomkit.local_models download \\
        ten-vad zipformer-en-kroko-2025-08-06 vits-piper-en_US-amy-low
    # Optional: denoiser (DENOISE=1), turn detection (SMART_TURN=1)
    python -m roomkit.local_models download gtcrn-simple smart-turn-v3.2-cpu

    `python -m roomkit.local_models list` shows the other models the
    STT_MODEL, TTS_MODEL and VAD_MODEL variables take.

Run with Ollama (recommended):
    ollama pull qwen3:8b

    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    uv run python examples/voice_local_onnx_vllm.py

Run with vLLM:
    LLM_MODEL=Qwen/Qwen3-8B \\
    LLM_BASE_URL=http://localhost:8000/v1 \\
    uv run python examples/voice_local_onnx_vllm.py

Environment variables:
    --- LLM (any OpenAI-compatible server) ---
    LLM_MODEL           (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL        Server endpoint (default: http://localhost:11434/v1 — Ollama)
    LLM_API_KEY         API key if server requires auth (default: none)
    LLM_MAX_TOKENS      Max response tokens (default: 256)
    SYSTEM_PROMPT       Custom system prompt

    --- Models (local models, by name) ---
    ROOMKIT_MODELS_DIR  Where the downloaded models live
                        (default: ~/.cache/roomkit/models)

    --- STT (sherpa-onnx) ---
    STT_MODEL           STT local model (default: zipformer-en-kroko-2025-08-06);
                        e.g. whisper-tiny.en-int8 or whisper-base-int8 (offline)
    STT_LANGUAGE        Language code for a Whisper model (default: the model's)

    --- TTS (sherpa-onnx) ---
    TTS_MODEL           TTS local model (default: vits-piper-en_US-amy-low)
    TTS_VOICE           Voice name or speaker ID (default: the model's)
    TTS_SPEED           Speech speed multiplier (default: 1.0)
    TTS_SAMPLE_RATE     Output sample rate (default: the model's)

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           VAD local model: ten-vad | silero-vad (default: ten-vad)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)
                        Lower values improve sensitivity for short utterances.
                        The GTCRN denoiser slightly alters spectral features,
                        which reduces TEN-VAD confidence — 0.35 compensates.
                        Without denoiser you can raise to 0.5 for fewer false
                        positives.

    --- Pipeline (optional) ---
    DENOISE             1 enables the GTCRN denoiser (gtcrn-simple) (default: 0)
    AEC                 Echo cancellation: webrtc | speex | 1 (=webrtc) | 0 (default: 0)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto, off with AEC)
    RECORDING_DIR       Record the call as WAV files into this directory
                        (default: unset, no recording)
    RECORDING_ENCRYPTED_AT_REST
                        Required with RECORDING_DIR: set it to 1 to state
                        that RECORDING_DIR is on encrypted storage. RoomKit
                        refuses plaintext recordings (RFC 17.6) and cannot
                        check the claim itself.
    RECORDING_MODE      Channel mode: mixed | separate | stereo (default: stereo)
    DEBUG_TAPS_DIR      Directory for pipeline debug taps (disabled if unset)
    DEBUG_TAPS_STAGES   Comma-separated stages to capture (default: all)
    ONNX_PROVIDER       ONNX execution provider for STT/TTS: cpu | cuda (default: cpu)
                        VAD and denoiser always use CPU (per-frame overhead makes
                        CUDA slower for these tiny models)

    --- Turn detection (optional) ---
    SMART_TURN          1 enables audio-native turn detection (smart-turn-v3.2-cpu)
                        (default: 0). Requires: pip install roomkit[smart-turn]
    SMART_TURN_THRESHOLD  Completion probability threshold 0-1 (default: 0.5)

    --- Other ---
    CONSOLE             1 shows the RoomKit console dashboard (default: 0)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    env_bool,
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
    PipelineDebugTaps,
    RecordingChannelMode,
    RecordingConfig,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
    WavFileRecorder,
)
from roomkit.voice.stt import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logger = setup_logging("voice_local_onnx_vllm")

CHANNEL_MODES = {
    "mixed": RecordingChannelMode.MIXED,
    "separate": RecordingChannelMode.SEPARATE,
    "stereo": RecordingChannelMode.STEREO,
}


def build_recording() -> tuple[WavFileRecorder | None, RecordingConfig | None]:
    """WAV recording of the call, off unless RECORDING_DIR is set.

    RFC 17.6 makes encryption at rest a MUST, so WavFileRecorder refuses to
    start without a RecordingEncryption or a statement that the storage
    encrypts at rest. RECORDING_ENCRYPTED_AT_REST=1 is that statement: yours,
    about RECORDING_DIR, which the example cannot verify.
    """
    recording_dir = os.environ.get("RECORDING_DIR", "")
    if not recording_dir:
        return None, None
    if not env_bool("RECORDING_ENCRYPTED_AT_REST", default=False):
        print(
            "Error: RECORDING_DIR needs RECORDING_ENCRYPTED_AT_REST=1, stating that "
            f"{recording_dir} is on encrypted storage (RFC 17.6 refuses plaintext recordings)"
        )
        sys.exit(1)
    mode_name = os.environ.get("RECORDING_MODE", "stereo").lower()
    config = RecordingConfig(
        storage=recording_dir,
        storage_encrypted_at_rest=True,  # stated by RECORDING_ENCRYPTED_AT_REST=1
        channels=CHANNEL_MODES.get(mode_name, RecordingChannelMode.STEREO),
    )
    logger.info("Recording to %s (mode=%s)", recording_dir, mode_name)
    return WavFileRecorder(), config


async def main() -> None:
    env = require_env("LLM_MODEL")
    stt_model = os.environ.get("STT_MODEL", "zipformer-en-kroko-2025-08-06")
    tts_model = os.environ.get("TTS_MODEL", "vits-piper-en_US-amy-low")
    vad_model = os.environ.get("VAD_MODEL", "ten-vad")
    denoise = env_bool("DENOISE", default=False)
    smart_turn = env_bool("SMART_TURN", default=False)
    require_local_models(
        stt_model,
        tts_model,
        vad_model,
        *(["gtcrn-simple"] if denoise else []),
        *(["smart-turn-v3.2-cpu"] if smart_turn else []),
    )

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # --- ONNX execution providers ----------------------------------------------
    # CUDA accelerates STT and TTS (batch inference on larger models).
    # VAD and denoiser run per-frame (every 20ms) on tiny models — the GPU
    # transfer overhead makes them *slower* on CUDA, so they always use CPU.
    onnx_provider = os.environ.get("ONNX_PROVIDER", "cpu")
    if onnx_provider == "cuda":
        logger.info("ONNX execution provider: CUDA (GPU) for STT/TTS, CPU for VAD/denoiser")
    else:
        logger.info("ONNX execution provider: CPU")

    # --- Audio settings -------------------------------------------------------
    sample_rate = 16000
    block_ms = 20
    frame_size = sample_rate * block_ms // 1000  # 320 samples

    # The TTS config first: its sample rate sets up the speaker output.
    tts_config = SherpaOnnxTTSConfig.from_local_model(
        tts_model,
        voice=os.environ.get("TTS_VOICE") or None,
        speed=float(os.environ.get("TTS_SPEED", "1.0")),
        provider=onnx_provider,
    )
    if os.environ.get("TTS_SAMPLE_RATE"):
        tts_config = replace(tts_config, sample_rate=int(os.environ["TTS_SAMPLE_RATE"]))
    tts_sample_rate = tts_config.sample_rate

    # --- AEC (optional, echo cancellation) ------------------------------------
    aec = None
    aec_mode = os.environ.get("AEC", "0").lower()
    if aec_mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec import WebRTCAECProvider

        aec = WebRTCAECProvider(sample_rate=sample_rate)
        logger.info("AEC enabled (WebRTC AEC3)")
    elif aec_mode == "speex":
        from roomkit.voice.pipeline import SpeexAECProvider

        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 10,
            sample_rate=sample_rate,
        )
        logger.info("AEC enabled (Speex, filter=%d samples)", frame_size * 10)

    # --- Backend: local mic + speakers ----------------------------------------
    # The backend owns the AEC: it feeds the speaker signal as the echo
    # reference block-aligned with playback, and reports NATIVE_AEC so the
    # pipeline does not run a second one.
    # When AEC is active it removes speaker echo from the mic signal, so we
    # can keep the mic open during playback and allow barge-in interruption.
    # Without AEC the mic is muted during playback to prevent feedback loops.
    # Override with MUTE_MIC=0|1 for testing (e.g. MUTE_MIC=0 to test AEC).
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_sample_rate,
        channels=1,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )

    # --- Denoiser (optional, sherpa-onnx GTCRN) -------------------------------
    denoiser = None
    if denoise:
        from roomkit.voice.pipeline import SherpaOnnxDenoiserConfig, SherpaOnnxDenoiserProvider

        denoiser = SherpaOnnxDenoiserProvider(
            SherpaOnnxDenoiserConfig.from_local_model("gtcrn-simple", provider="cpu")
        )
        logger.info("Denoiser: sherpa-onnx GTCRN (model=gtcrn-simple)")

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig.from_local_model(
            vad_model,
            threshold=vad_threshold,
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            sample_rate=sample_rate,
            provider="cpu",
        )
    )
    logger.info("VAD: sherpa-onnx (model=%s, threshold=%.2f)", vad_model, vad_threshold)

    # --- WAV recorder (opt-in) ------------------------------------------------
    recorder, recording_config = build_recording()

    # --- Debug taps (optional pipeline stage capture) -------------------------
    debug_taps = None
    debug_taps_dir = os.environ.get("DEBUG_TAPS_DIR", "")
    if debug_taps_dir:
        stages_env = os.environ.get("DEBUG_TAPS_STAGES", "all")
        stages = [s.strip() for s in stages_env.split(",")]
        debug_taps = PipelineDebugTaps(output_dir=debug_taps_dir, stages=stages)
        logger.info("Debug taps: %s (stages=%s)", debug_taps_dir, stages)

    # --- Smart turn detection (optional, audio-native) -----------------------
    turn_detector = None
    if smart_turn:
        from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector

        smart_turn_threshold = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))
        turn_detector = SmartTurnDetector(
            SmartTurnConfig.from_local_model("smart-turn-v3.2-cpu", threshold=smart_turn_threshold)
        )
        logger.info(
            "Turn detection: smart-turn (model=smart-turn-v3.2-cpu, threshold=%.2f)",
            smart_turn_threshold,
        )

    # --- Pipeline config ------------------------------------------------------
    # No aec= here: the backend runs it (see above).
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
        debug_taps=debug_taps,
        turn_detector=turn_detector,
    )

    # --- STT (sherpa-onnx) ----------------------------------------------------
    stt_config = SherpaOnnxSTTConfig.from_local_model(
        stt_model, sample_rate=sample_rate, provider=onnx_provider
    )
    if os.environ.get("STT_LANGUAGE"):
        stt_config = replace(stt_config, language=os.environ["STT_LANGUAGE"])
    stt = SherpaOnnxSTTProvider(stt_config)
    logger.info("STT: sherpa-onnx (model=%s, mode=%s)", stt_model, stt_config.mode)

    # --- TTS (sherpa-onnx) ----------------------------------------------------
    tts = SherpaOnnxTTSProvider(tts_config)
    logger.info(
        "TTS: sherpa-onnx (model=%s, voice=%s, rate=%d)",
        tts_model,
        tts.default_voice,
        tts_sample_rate,
    )

    # --- LLM (local via OpenAI-compatible API: Ollama, vLLM, etc.) ------------
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

    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=system_prompt,
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="local-voice")
    await kit.attach_channel("local-voice", "ai", category=ChannelCategory.INTELLIGENCE)

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

    @kit.hook(HookTrigger.ON_TURN_COMPLETE, execution=HookExecution.ASYNC)
    async def on_turn_complete(event, ctx):
        logger.info("Turn complete (confidence=%.2f): %s", event.confidence, event.text)

    @kit.hook(HookTrigger.ON_TURN_INCOMPLETE, execution=HookExecution.ASYNC)
    async def on_turn_incomplete(event, ctx):
        logger.info("Turn incomplete (confidence=%.2f): %s", event.confidence, event.text)

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
    async def on_rec_started(event, ctx):
        logger.info("Recording started: %s", event.id)

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
    async def on_rec_stopped(event, ctx):
        logger.info(
            "Recording stopped: %s (%.1fs, files=%s)",
            event.id,
            event.duration_seconds,
            event.urls,
        )

    # --- Warmup: pre-load models (avoids delay on first interaction) ----------
    logger.info("Loading models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded — ready!")

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("local-voice", "voice")

    logger.info("")
    logger.info("All local — no cloud APIs needed!")
    logger.info("Speak into your microphone. Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        if recording_config is not None:
            logger.info("Recordings saved to: %s", recording_config.storage)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
