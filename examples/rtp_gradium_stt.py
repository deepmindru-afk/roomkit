"""RoomKit -- RTP receiver with Gradium STT transcription.

Receive voice audio over RTP and transcribe it to text using Gradium
speech-to-text. Transcriptions are printed to the console in real time.

No AI provider or TTS is used — this is a pure listen-and-transcribe example.

Audio flow:
    RTP (8kHz PCMU) → Pipeline (resample to 16kHz) → Gradium STT → print

Gradium handles speech detection and turn endpointing server-side.  An
inactivity watchdog monitors the RTP stream — when no frames arrive for
500 ms, the last partial transcription is promoted to a final result.

Prerequisites:
    pip install roomkit[rtp,gradium]

Run with:
    GRADIUM_API_KEY=... uv run python examples/rtp_gradium_stt.py

Then send RTP audio to the local port (default 10000).  With aiortp:

    python examples/send_wav.py audio.wav 127.0.0.1 10000

Or with ffmpeg:

    ffmpeg -re -i audio.wav -ar 8000 -ac 1 -acodec pcm_mulaw \\
        -f rtp rtp://127.0.0.1:10000

Environment variables:
    GRADIUM_API_KEY     (required) Gradium API key
    RTP_LOCAL_PORT      Local port to bind RTP (default: 10000)
    GRADIUM_REGION      API region (default: us)
    GRADIUM_STT_MODEL   STT model name (default: default)
    VOICE_LANGUAGE      Language code for STT (default: en)

    --- Debug ---
    DEBUG               Set to 1 for verbose pipeline/STT logging
    RECORD_DIR          Directory to save audio WAVs (default: no recording):
                        - transport_8khz.wav  (raw 8kHz audio from RTP, pre-pipeline)
                        - {session}_{ts}_inbound.wav (16kHz post-pipeline via recorder)
    RECORDING_ENCRYPTED_AT_REST
                        Set to 1 to declare RECORD_DIR is on encrypted storage —
                        required with RECORD_DIR, the WAV recorder refuses
                        plaintext storage (RFC §17.6)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    env_bool,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
    voice_language,
)

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.backends.rtp import RTPVoiceBackend
from roomkit.voice.pipeline import AudioFormat, AudioPipelineConfig, AudioPipelineContract
from roomkit.voice.pipeline.recorder import (
    RecordingChannelMode,
    RecordingConfig,
    RecordingMode,
    WavFileRecorder,
)
from roomkit.voice.pipeline.resampler import SincResamplerProvider
from roomkit.voice.stt.gradium import GradiumSTTConfig, GradiumSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logger = setup_logging("rtp_gradium_stt")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit.voice").setLevel(logging.DEBUG)
    logging.getLogger("roomkit.voice.stt.gradium").setLevel(logging.DEBUG)
    logging.getLogger("roomkit.voice.pipeline").setLevel(logging.DEBUG)


async def main() -> None:
    env = require_env("GRADIUM_API_KEY")
    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # --- Configuration --------------------------------------------------------
    local_port = int(os.environ.get("RTP_LOCAL_PORT", "10000"))

    # --- RTP backend ----------------------------------------------------------
    backend = RTPVoiceBackend(
        local_addr=("0.0.0.0", local_port),
        remote_addr=("127.0.0.1", 9),  # discard port — we only receive, never send
        payload_type=0,  # PCMU (G.711 mu-law)
        clock_rate=8000,
    )

    # --- Pipeline (no VAD — Gradium handles endpointing server-side) ----------
    # Contract tells the pipeline to resample 8kHz RTP audio to 16kHz internally.
    # Without this, raw 8kHz frames reach STT and cause resampling artifacts.
    contract = AudioPipelineContract(
        transport_inbound_format=AudioFormat(sample_rate=8000, channels=1),
        transport_outbound_format=AudioFormat(sample_rate=8000, channels=1),
    )
    # --- Optional recording (RECORD_DIR=/tmp/recordings) -----------------------
    record_dir = os.environ.get("RECORD_DIR")
    recorder = None
    recording_config = None
    if record_dir:
        # RoomKit ships no default cipher: the operator declares the
        # directory sits on encrypted storage.
        if not env_bool("RECORDING_ENCRYPTED_AT_REST", default=False):
            print(
                "Error: RECORD_DIR must be on encrypted storage; "
                "set RECORDING_ENCRYPTED_AT_REST=1 to declare it"
            )
            sys.exit(1)
        recorder = WavFileRecorder()
        recording_config = RecordingConfig(
            mode=RecordingMode.INBOUND_ONLY,
            channels=RecordingChannelMode.SEPARATE,
            storage=record_dir,
            storage_encrypted_at_rest=True,  # declared by RECORDING_ENCRYPTED_AT_REST
        )
        logger.info("Recording inbound audio to %s", record_dir)

    pipeline = AudioPipelineConfig(
        contract=contract,
        resampler=SincResamplerProvider(),
        recorder=recorder,
        recording_config=recording_config,
    )

    # --- Gradium STT ----------------------------------------------------------
    region = os.environ.get("GRADIUM_REGION", "us")
    language = voice_language("en")
    stt = GradiumSTTProvider(
        config=GradiumSTTConfig(
            api_key=env["GRADIUM_API_KEY"],
            region=region,
            model_name=os.environ.get("GRADIUM_STT_MODEL", "default"),
            input_format="pcm",
            language=language,
        )
    )
    logger.info("STT: Gradium (region=%s, lang=%s)", region, language)

    # --- Voice channel (no TTS — transcribe only) -----------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=MockTTSProvider(),  # placeholder — not used
        backend=backend,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="rtp-stt")
    await kit.attach_channel("rtp-stt", "voice")

    # --- Hooks: print transcriptions ------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session_arg, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session_arg, ctx):
        logger.info("Speech ended")

    @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def on_partial(result, ctx):
        logger.info("Partial: %r", result.text)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        print(f"\n>>> {event.text}\n")
        # Block further processing (no AI provider to forward to)
        return HookResult.block("transcribe-only, no AI provider")

    # --- Start RTP session ----------------------------------------------------
    session = await backend.connect("rtp-stt", "rtp-caller", "voice")
    await kit.join("rtp-stt", "voice", session=session)

    # --- Transport-level WAV recording (raw 8kHz, pre-pipeline) -----------------
    transport_wav: wave.Wave_write | None = None
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        transport_wav_path = os.path.join(record_dir, "transport_8khz.wav")
        transport_wav = wave.open(transport_wav_path, "wb")  # noqa: SIM115
        transport_wav.setnchannels(1)
        transport_wav.setsampwidth(2)
        transport_wav.setframerate(8000)
        logger.info("Transport recording: %s", transport_wav_path)

        # Wrap the backend's audio callback to capture raw transport audio
        # before pipeline processing.  RTPVoiceBackend has no public way to
        # add a listener next to the channel's (SIPVoiceBackend has
        # subscribe_audio_received), so this patches its private callback.
        orig_cb = backend._audio_received_callback

        def _record_transport(sess, frame):
            transport_wav.writeframes(frame.data)
            return orig_cb(sess, frame)

        backend._audio_received_callback = _record_transport

    logger.info("RTP listening on 0.0.0.0:%d", local_port)
    logger.info("Send audio with e.g.:")
    logger.info(
        "  ffmpeg -re -i audio.wav -ar 8000 -ac 1 -acodec pcm_mulaw -f rtp rtp://127.0.0.1:%d",
        local_port,
    )
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C --------------------------------------------
    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        if transport_wav is not None:
            transport_wav.close()
            logger.info("Transport recording saved.")
        await kit.leave(session)  # also disconnects the backend session

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
