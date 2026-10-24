#!/usr/bin/env python3
"""RoomKit — SIP calls routed to a fully local AI agent (no cloud APIs).

Incoming SIP calls from a PBX/trunk are answered automatically and processed
through a fully local pipeline:

  Phone → SIP/RTP → [Resampler] → [Denoiser] → VAD
  → sherpa-onnx STT → Local LLM → sherpa-onnx TTS → SIP/RTP → Phone

No echo canceller here: the caller's handset cancels its own acoustic echo.

Everything runs on your server — no cloud APIs, no data leaves the box.
Ideal for privacy-sensitive deployments, air-gapped environments, or
on-premise telephony AI.

Every call joins the ``support`` room.

Requirements:
    pip install roomkit[sip,openai,sherpa-onnx]
    A local LLM server — pick one:
      Ollama:   ollama pull qwen3:8b && ollama serve
      vLLM:     vllm serve Qwen/Qwen3-8B --port 8000

Models (download once, while online):
    python -m roomkit.local_models download \\
        ten-vad zipformer-en-kroko-2025-08-06 vits-piper-en_US-amy-low
    # Denoiser (optional, recommended for noisy phone lines; DENOISE=1)
    python -m roomkit.local_models download gtcrn-simple

Usage:
    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    python examples/voice_sip_local_agent.py

Environment variables:
    --- SIP ---
    SIP_HOST            Listening address (default: 0.0.0.0)
    SIP_PORT            SIP port (default: 5060)
    RTP_IP              RTP bind address (default: 0.0.0.0)
    RTP_PORT_START      RTP port range start (default: 10000)
    RTP_PORT_END        RTP port range end (default: 20000)

    --- LLM (any OpenAI-compatible server) ---
    LLM_MODEL           (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL        Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY         API key if server requires auth (default: none)
    LLM_MAX_TOKENS      Max response tokens (default: 256)
    SYSTEM_PROMPT       Custom system prompt

    --- STT (sherpa-onnx) ---
    STT_MODEL           Local STT model (default: zipformer-en-kroko-2025-08-06;
                        python -m roomkit.local_models list --kind sherpa-onnx-stt)

    --- TTS (sherpa-onnx) ---
    TTS_MODEL           Local TTS model (default: vits-piper-en_US-amy-low;
                        python -m roomkit.local_models list --kind sherpa-onnx-tts)
    TTS_SPEAKER_ID      Speaker ID or voice name for multi-speaker models
                        (default: the model's)
    TTS_SPEED           Speech speed multiplier (default: 1.0)

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           Local VAD model: ten-vad | silero-vad (default: ten-vad)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)

    --- Pipeline (optional) ---
    DENOISE             1 enables the GTCRN denoiser (gtcrn-simple) (default: 0)
    RECORDING_DIR       Record calls as WAV files there (default: no recording)
    RECORDING_ENCRYPTED_AT_REST
                        Set to 1 to declare RECORDING_DIR is on encrypted
                        storage — required with RECORDING_DIR, the WAV
                        recorder refuses plaintext storage (RFC §17.6)
    RECORDING_MODE      Channel mode: mixed | separate | stereo (default: stereo)
    CONSOLE             Set to 1 for the live console dashboard
    ONNX_PROVIDER       ONNX execution provider: cpu | cuda (default: cpu)
    ROOMKIT_MODELS_DIR  Where the downloaded models live (default: ~/.cache/roomkit/models)
"""

from __future__ import annotations

import asyncio
import os
import sys
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

logger = setup_logging("sip_local_agent")

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
    VoiceChannel,
)
from roomkit.models import ProtocolTrace
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.voice.backends import SIPVoiceBackend
from roomkit.voice.pipeline import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
    RecordingChannelMode,
    RecordingConfig,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
    WavFileRecorder,
)
from roomkit.voice.stt import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = os.environ.get("SIP_HOST", "0.0.0.0")  # nosec B104
SIP_PORT = int(os.environ.get("SIP_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")  # nosec B104
RTP_PORT_START = int(os.environ.get("RTP_PORT_START", "10000"))
RTP_PORT_END = int(os.environ.get("RTP_PORT_END", "20000"))

CHANNEL_MODES = {
    "mixed": RecordingChannelMode.MIXED,
    "separate": RecordingChannelMode.SEPARATE,
    "stereo": RecordingChannelMode.STEREO,
}


async def main() -> None:
    require_env("LLM_MODEL")
    stt_model = os.environ.get("STT_MODEL", "zipformer-en-kroko-2025-08-06")
    tts_model = os.environ.get("TTS_MODEL", "vits-piper-en_US-amy-low")
    vad_model = os.environ.get("VAD_MODEL", "ten-vad")
    denoise = env_bool("DENOISE", default=False)
    require_local_models(vad_model, stt_model, tts_model, *(["gtcrn-simple"] if denoise else []))

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # --- ONNX execution providers ----------------------------------------
    onnx_provider = os.environ.get("ONNX_PROVIDER", "cpu")
    if onnx_provider == "cuda":
        logger.info("ONNX: CUDA (GPU) for STT/TTS, CPU for VAD/denoiser")
    else:
        logger.info("ONNX: CPU")

    # --- Audio settings --------------------------------------------------
    sample_rate = 16000

    # --- SIP backend (answers incoming calls) ----------------------------
    sip = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
        user_agent="RoomKit/LocalAgent",
        server_name="RoomKit",
    )

    # --- Denoiser (optional, sherpa-onnx GTCRN) --------------------------
    denoiser = None
    if denoise:
        from roomkit.voice.pipeline import SherpaOnnxDenoiserConfig, SherpaOnnxDenoiserProvider

        denoiser = SherpaOnnxDenoiserProvider(
            SherpaOnnxDenoiserConfig.from_local_model("gtcrn-simple", provider="cpu")
        )
        logger.info("Denoiser: sherpa-onnx GTCRN (gtcrn-simple)")

    # --- VAD (sherpa-onnx neural VAD) ------------------------------------
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
    logger.info("VAD: %s (threshold=%.2f)", vad_model, vad_threshold)

    # --- WAV recorder (opt-in: RECORDING_DIR) ----------------------------
    # Recording a call is opt-in. RoomKit ships no default cipher, so the
    # operator declares the directory sits on encrypted storage.
    recorder = None
    recording_config = None
    recording_dir = os.environ.get("RECORDING_DIR", "")
    if recording_dir:
        if not env_bool("RECORDING_ENCRYPTED_AT_REST", default=False):
            print(
                "Error: RECORDING_DIR must be on encrypted storage; "
                "set RECORDING_ENCRYPTED_AT_REST=1 to declare it"
            )
            sys.exit(1)
        rec_mode_name = os.environ.get("RECORDING_MODE", "stereo").lower()
        recorder = WavFileRecorder()
        recording_config = RecordingConfig(
            storage=recording_dir,
            storage_encrypted_at_rest=True,  # declared by RECORDING_ENCRYPTED_AT_REST
            channels=CHANNEL_MODES.get(rec_mode_name, RecordingChannelMode.STEREO),
        )
        logger.info("Recording to %s (mode=%s)", recording_dir, rec_mode_name)

    # --- Pipeline contract -----------------------------------------------
    # SIP negotiates G.722 (16 kHz) or G.711 (8 kHz). TTS may output at a
    # different rate (e.g. 22050 Hz for Piper). The contract tells the
    # pipeline to resample outbound audio to match the SIP codec rate.
    sip_rate = 16000  # G.722 preferred; change to 8000 if only G.711
    contract = AudioPipelineContract(
        transport_inbound_format=AudioFormat(sample_rate=sip_rate),
        transport_outbound_format=AudioFormat(sample_rate=sip_rate),
        internal_format=AudioFormat(sample_rate=sample_rate),
    )

    # --- Pipeline config -------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
        contract=contract,
    )

    # --- STT (sherpa-onnx) -----------------------------------------------
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig.from_local_model(
            stt_model, sample_rate=sample_rate, provider=onnx_provider
        )
    )
    logger.info("STT: sherpa-onnx (%s)", stt_model)

    # --- TTS (sherpa-onnx) -----------------------------------------------
    tts_config = SherpaOnnxTTSConfig.from_local_model(
        tts_model,
        voice=os.environ.get("TTS_SPEAKER_ID"),
        speed=float(os.environ.get("TTS_SPEED", "1.0")),
        provider=onnx_provider,
    )
    tts = SherpaOnnxTTSProvider(tts_config)
    logger.info("TTS: sherpa-onnx (%s, rate=%d)", tts_model, tts_config.sample_rate)

    # --- LLM (local via OpenAI-compatible API) ---------------------------
    llm_model = os.environ["LLM_MODEL"]
    llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    ai_provider = create_vllm_provider(
        VLLMConfig(
            model=llm_model,
            base_url=llm_base_url,
            api_key=os.environ.get("LLM_API_KEY", "none"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
        )
    )
    logger.info("LLM: %s (%s)", llm_model, llm_base_url)

    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly phone assistant. Keep your responses "
        "short and conversational — one or two sentences at most. "
        "Answer directly without thinking, reasoning, or internal monologue. "
        "/no_think",
    )

    # --- Channels --------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=sip,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
    kit.register_channel(ai)

    # -------------------------------------------------------------------
    # Room setup — create once at startup
    # -------------------------------------------------------------------

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "voice")
    await kit.attach_channel("support", "ai", category=ChannelCategory.INTELLIGENCE)

    # -------------------------------------------------------------------
    # Hooks — same hooks work for text AND voice
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx: RoomContext) -> HookResult:
        """Log what the caller says."""
        logger.info("Caller: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text: str, ctx: RoomContext) -> HookResult:
        """Log what the AI responds."""
        logger.info("Agent: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
    async def on_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
        """Log SIP signaling at the room level."""
        logger.info(
            "ON_PROTOCOL_TRACE [room=%s] %s %s: %s",
            ctx.room.id,
            trace.direction,
            trace.protocol,
            trace.summary,
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def log_events(event: RoomEvent, ctx: RoomContext) -> None:
        """Log every event that flows through the room."""
        logger.info(
            "AFTER_BROADCAST — type=%s channel=%s room=%s provider=%s",
            event.type,
            event.source.channel_id,
            ctx.room.id,
            event.source.provider,
        )

    # -------------------------------------------------------------------
    # Incoming call → join session to room
    # -------------------------------------------------------------------

    @sip.on_call
    async def handle_call(session):
        # kit.join() emits no session_started event, so a BEFORE_BROADCAST
        # hook never sees the call: screen callers here, before joining.
        await kit.join("support", "voice", session=session)
        logger.info(
            "Call connected — session=%s caller=%s room=support",
            session.id,
            session.metadata.get("caller", "unknown"),
        )

    # -------------------------------------------------------------------
    # Remote hangup — the voice channel unbinds the session by itself
    # -------------------------------------------------------------------

    @sip.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("Call ended — session=%s", session.id)

    # -------------------------------------------------------------------
    # Warmup models + start
    # -------------------------------------------------------------------

    logger.info("Loading ONNX models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded!")

    await sip.start()
    logger.info(
        "SIP + Local Agent ready — SIP %s:%d, RTP %d-%d",
        SIP_HOST,
        SIP_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )
    logger.info("LLM: %s — STT: sherpa-onnx — TTS: sherpa-onnx — no cloud APIs", llm_model)
    logger.info("Waiting for incoming SIP calls...")

    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await sip.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
