"""RoomKit -- Cloud voice assistant with local microphone.

Talk to Claude through your microphone with cloud AI services:
  - Deepgram for speech-to-text (streaming)
  - Claude (Anthropic) for AI responses
  - ElevenLabs for text-to-speech
  - WebRTC or Speex AEC for echo cancellation
  - RNNoise or sherpa-onnx GTCRN for noise suppression
  - sherpa-onnx neural VAD (TEN-VAD or Silero) for speech detection
  - WavFileRecorder for debug audio capture (opt-in)

Audio flows through the full pipeline:

  Mic -> [Resampler] -> [Recorder tap] -> [AEC] -> [Denoiser] -> VAD
  -> Deepgram STT -> Claude -> ElevenLabs TTS -> [Recorder tap] -> Speaker

Requirements:
    pip install roomkit[local-audio,anthropic,deepgram,elevenlabs,sherpa-onnx,webrtc-aec]
    System (optional): libspeexdsp (apt install libspeexdsp1) for Speex AEC
    System (optional): librnnoise (apt install librnnoise0) for the default
                       RNNoise denoiser -- without it the denoiser is skipped

Run with:
    ANTHROPIC_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    VAD_MODEL=path/to/ten-vad.onnx \\
    uv run python examples/voice_cloud.py

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    ELEVENLABS_API_KEY  (required) ElevenLabs API key
    ELEVENLABS_VOICE_ID Voice ID (default: Rachel)
    VOICE_LANGUAGE      Language code for STT (default: en)
    DEEPGRAM_MODEL      Deepgram STT model (default: nova-3)
    DEEPGRAM_KEYTERMS   Comma-separated key terms to boost (default: none)
    SYSTEM_PROMPT       Custom system prompt for Claude
    CONSOLE             1 shows the RoomKit console dashboard (default: 0)

    --- VAD (sherpa-onnx) ---
    VAD                 0 disables local VAD: continuous STT, Deepgram
                        endpointing decides the turns (default: 1)
    VAD_MODEL           Path to sherpa-onnx VAD .onnx model file
                        (unset: energy VAD)
    VAD_MODEL_TYPE      Model type: ten | silero (default: ten)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)
                        Lower values improve sensitivity for short utterances.
                        The GTCRN denoiser slightly alters spectral features,
                        which reduces TEN-VAD confidence -- 0.35 compensates.
                        Without denoiser you can raise to 0.5 for fewer false
                        positives.

    --- Interruption / barge-in (optional) ---
    INTERRUPTION        Strategy: immediate | confirmed | semantic | disabled
                        (unset = the channel's own default, IMMEDIATE)
                          immediate — cut the assistant as soon as speech starts
                          confirmed — cut it only after MIN_SPEECH_MS of
                                      sustained speech; shorter bursts are
                                      treated as echo and dropped
                          disabled  — never cut; speech is queued and handled
                                      once the assistant has finished
    MIN_SPEECH_MS       Sustained speech required by confirmed (default: 300)
    ALLOW_DURING_FIRST_MS
                        Ignore barge-in during the first N ms of playback
                        (default: 0)

    --- Pipeline (optional) ---
    AEC                 Echo cancellation: webrtc | speex | 1 (=webrtc) | 0
                        (default: webrtc)
    AEC_NS              WebRTC AEC3 noise suppression: 1 | 0 (default: 0).
                        Attenuates what the echo canceller leaves behind —
                        worth enabling on open speakers, where the residual
                        can otherwise clear the VAD's speech threshold.
    AEC_AGC             WebRTC AEC3 gain control: 1 | 0 (default: 0)
    DENOISE             Noise suppression: rnnoise | sherpa | webrtc |
                        1 (=rnnoise) | 0 (default: rnnoise; skipped with a
                        warning when its library is missing)
    DENOISE_MODEL       GTCRN .onnx model for DENOISE=sherpa
                        (default: gtcrn_simple.onnx)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto,
                        off with AEC)
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

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    build_denoiser,
    env_bool,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
    voice_language,
)

from roomkit import ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    DenoiserProvider,
    PipelineDebugTaps,
    RecordingChannelMode,
    RecordingConfig,
    WavFileRecorder,
)
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logger = setup_logging("voice_cloud")

# Channel mode mapping
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
    env = require_env("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # --- Audio settings -------------------------------------------------------
    sample_rate = 16000
    block_ms = 20
    frame_size = sample_rate * block_ms // 1000  # 320 samples
    output_rate = 24000  # ElevenLabs native rate

    # --- AEC (echo cancellation) ----------------------------------------------
    # Unified format: AEC=webrtc | speex | 1 (=webrtc) | 0
    aec = None
    aec_mode = os.environ.get("AEC", "webrtc").lower()
    if aec_mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        # AEC3 carries its own noise suppressor and gain control, both off by
        # default. On open speakers the echo AEC3 leaves behind can clear the
        # VAD's speech threshold, and with no denoiser in the pipeline nothing
        # else attenuates it — so this is the knob that keeps the assistant
        # from hearing itself.
        aec_ns = os.environ.get("AEC_NS", "0") == "1"
        aec_agc = os.environ.get("AEC_AGC", "0") == "1"
        aec = WebRTCAECProvider(sample_rate=sample_rate, enable_ns=aec_ns, enable_agc=aec_agc)
        logger.info("AEC enabled (WebRTC AEC3, ns=%s, agc=%s)", aec_ns, aec_agc)
    elif aec_mode == "speex":
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 10,  # 200ms echo tail
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
    # Override with MUTE_MIC=0|1 for testing.
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=output_rate,
        channels=1,
        block_duration_ms=block_ms,
        aec=aec,
        mute_mic_during_playback=mute_mic,
    )
    logger.info(
        "Backend: LocalAudio (in=%dHz, out=%dHz, mute_mic=%s)",
        sample_rate,
        output_rate,
        mute_mic,
    )

    # --- Denoiser (RNNoise by default, or sherpa-onnx GTCRN / WebRTC NS) ------
    # The shared builder returns `object` to keep its imports lazy.
    denoiser = cast("DenoiserProvider | None", build_denoiser(sample_rate, default="rnnoise"))

    # --- WAV recorder (opt-in) ------------------------------------------------
    recorder, recording_config = build_recording()

    # --- VAD (sherpa-onnx neural VAD or energy fallback) ----------------------
    # VAD=0 disables local VAD → continuous STT mode (Deepgram handles endpointing)
    vad_disabled = os.environ.get("VAD", "1") == "0"
    vad_model = os.environ.get("VAD_MODEL", "")
    if vad_disabled:
        vad = None
        logger.info("VAD: disabled (continuous STT mode)")
    elif vad_model:
        vad_model_type = os.environ.get("VAD_MODEL_TYPE", "ten")
        vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))
        vad = SherpaOnnxVADProvider(
            SherpaOnnxVADConfig(
                model=vad_model,
                model_type=vad_model_type,
                threshold=vad_threshold,
                silence_threshold_ms=600,
                min_speech_duration_ms=200,
                sample_rate=sample_rate,
            )
        )
        logger.info(
            "VAD: sherpa-onnx (type=%s, threshold=%.2f, model=%s)",
            vad_model_type,
            vad_threshold,
            vad_model,
        )
    else:
        vad = EnergyVADProvider(
            energy_threshold=300.0,
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
        )
        logger.info("VAD: EnergyVAD (set VAD_MODEL for neural VAD)")

    # --- Debug taps (pipeline stage audio capture) ----------------------------
    debug_taps = None
    debug_taps_dir = os.environ.get("DEBUG_TAPS_DIR", "")
    if debug_taps_dir:
        stages_env = os.environ.get("DEBUG_TAPS_STAGES", "all")
        stages = [s.strip() for s in stages_env.split(",")]
        debug_taps = PipelineDebugTaps(
            output_dir=debug_taps_dir,
            stages=stages,
        )
        logger.info("Debug taps: %s (stages=%s)", debug_taps_dir, stages)

    # --- Pipeline config ------------------------------------------------------
    # No aec= here: the backend runs it (see above).
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
        debug_taps=debug_taps,
    )

    # --- Deepgram STT ---------------------------------------------------------
    language = voice_language("en") or "en"
    stt_model = os.environ.get("DEEPGRAM_MODEL", "nova-3")
    keyterms = [k.strip() for k in os.environ.get("DEEPGRAM_KEYTERMS", "").split(",") if k.strip()]
    stt = DeepgramSTTProvider(
        config=DeepgramConfig(
            api_key=env["DEEPGRAM_API_KEY"],
            model=stt_model,
            language=language,
            keyterm=keyterms,
            punctuate=True,
            smart_format=True,
            endpointing=300,
        )
    )
    logger.info("STT: Deepgram %s (language=%s)", stt_model, language)

    # --- ElevenLabs TTS -------------------------------------------------------
    tts_format = f"pcm_{output_rate}"
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    tts = ElevenLabsTTSProvider(
        config=ElevenLabsConfig(
            api_key=env["ELEVENLABS_API_KEY"],
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format=tts_format,
            optimize_streaming_latency=3,
        )
    )
    logger.info("TTS: ElevenLabs (voice=%s, format=%s)", voice_id, tts_format)

    # --- Claude AI ------------------------------------------------------------
    ai_provider = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-opus-5",
            max_tokens=256,
        )
    )
    logger.info("AI: Claude (claude-opus-5)")

    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly voice assistant. Keep your responses "
        "short and conversational — one or two sentences at most.",
    )

    # --- Interruption strategy ------------------------------------------------
    # Unset leaves the channel's own default (IMMEDIATE: cut the bot the moment
    # speech is detected). Set it to compare how the strategies behave when you
    # talk over the assistant.
    strategy_name = os.environ.get("INTERRUPTION", "").strip().lower()
    interruption: InterruptionConfig | None = None
    if strategy_name:
        interruption = InterruptionConfig(
            strategy=InterruptionStrategy(strategy_name),
            min_speech_ms=int(os.environ.get("MIN_SPEECH_MS", "300")),
            allow_during_first_ms=int(os.environ.get("ALLOW_DURING_FIRST_MS", "0")),
        )

    # --- Voice channel --------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
        interruption=interruption,
    )
    if interruption is None:
        logger.info("Interruption: channel default (immediate)")
    else:
        logger.info(
            "Interruption: strategy=%s, min_speech_ms=%d, allow_during_first_ms=%d",
            interruption.strategy.value,
            interruption.min_speech_ms,
            interruption.allow_during_first_ms,
        )
    kit.register_channel(voice)

    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=system_prompt,
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="voice-demo")
    await kit.attach_channel("voice-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------

    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended")

    @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
    async def on_barge_in(event, ctx):
        # How far into its answer the assistant was when you cut it. With a
        # duration-based strategy this fires only once the speech has lasted
        # min_speech_ms, so the position is that much later than speech onset.
        logger.info(">>> BARGE-IN: cut the assistant at %dms of playback", event.audio_position_ms)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You said: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Claude says: %s", text)
        return HookResult.allow()

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

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("voice-demo", "voice")

    logger.info("")
    logger.info("Speak into your microphone!")
    logger.info("Press Ctrl+C to stop.")
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
