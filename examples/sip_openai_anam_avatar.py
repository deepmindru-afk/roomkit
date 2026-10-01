"""RoomKit — SIP + OpenAI Realtime + Anam Avatar.

Accept SIP video calls. OpenAI Realtime handles the conversation
(speech-to-speech), Anam renders the avatar face (passthrough mode).

Architecture:
    SIP phone → audio → RealtimeAVBridge → OpenAI Realtime (STT → LLM → TTS)
                                                 ↓
                                          TTS audio out
                                           ↓          ↓
                                    SIP speaker    AnamAvatarProvider
                                    (caller)         ↓
                                                  Anam Cloud (lip-sync)
                                                     ↓
                                               video frames
                                                     ↓
                                          pipeline → H.264 → SIP video

Prerequisites:
    pip install roomkit[anam,sip,video,local-video,realtime-openai]
    (local-video brings OpenCV for the watermark and the "connecting" frame)

Run with:
    export OPENAI_API_KEY="sk-..."
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    uv run python examples/sip_openai_anam_avatar.py

Environment variables:
    OPENAI_API_KEY       OpenAI API key (required)
    OPENAI_MODEL         Realtime model (default: gpt-realtime-2.1-mini)
    OPENAI_VOICE         Voice preset (default: alloy)
    SYSTEM_PROMPT        System prompt override
    ANAM_API_KEY         Anam API key (required)
    ANAM_AVATAR_ID       Avatar from lab.anam.ai (required)
    ANAM_VOICE_ID        Voice from lab.anam.ai (optional; unused in passthrough,
                         where OpenAI speaks)
    ANAM_LLM_ID          LLM from lab.anam.ai (optional; unused in passthrough,
                         where OpenAI answers)
    SIP_PORT             SIP listener port (default: 5060)
    RTP_IP               IP to bind RTP on (default: 0.0.0.0; the SDP then
                         advertises the resolved local IP)
    RTP_PORT_START       First RTP port to allocate, below 20000 (default: 10000)
    DEBUG                Set to 1 for verbose logging

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio
import logging
import os
import signal

from shared import require_env, setup_logging

from roomkit.providers.anam import AnamConfig
from roomkit.providers.anam.avatar import AnamAvatarProvider
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.pipeline import VideoPipelineConfig
from roomkit.video.pipeline.encoder.pyav import PyAVVideoEncoder
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.video.utils import make_text_frame
from roomkit.voice.realtime.bridge import RealtimeAVBridge

logger = setup_logging("sip_openai_anam")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)


async def main() -> None:
    # --- Validate environment -------------------------------------------------
    env = require_env("OPENAI_API_KEY", "ANAM_API_KEY", "ANAM_AVATAR_ID")
    openai_key = env["OPENAI_API_KEY"]
    anam_key = env["ANAM_API_KEY"]

    avatar_id = env["ANAM_AVATAR_ID"]
    # Optional: in passthrough mode OpenAI speaks and answers, Anam only animates.
    voice_id = os.environ.get("ANAM_VOICE_ID") or None
    llm_id = os.environ.get("ANAM_LLM_ID") or None

    # --- SIP backend ----------------------------------------------------------
    sip = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", int(os.environ.get("SIP_PORT", "5060"))),
        local_rtp_ip=os.environ.get("RTP_IP", "0.0.0.0"),
        rtp_port_start=int(os.environ.get("RTP_PORT_START", "10000")),
        supported_video_codecs=["H264"],
    )

    # --- OpenAI Realtime (speech-to-speech, audio only) -----------------------
    openai_provider = OpenAIRealtimeProvider(
        api_key=openai_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-realtime-2.1-mini"),
    )

    # --- Anam avatar (passthrough — lip-sync only, no STT/LLM) ---------------
    avatar = AnamAvatarProvider(
        AnamConfig(
            api_key=anam_key,
            avatar_id=avatar_id,
            voice_id=voice_id,
            llm_id=llm_id,
            enable_audio_passthrough=True,
        ),
        audio_sample_rate=24000,
    )

    # --- Bridge: SIP ↔ OpenAI + Anam avatar -----------------------------------
    bridge = RealtimeAVBridge(
        openai_provider,
        sip,
        avatar=avatar,
        video_pipeline=VideoPipelineConfig(
            filters=[
                WatermarkFilter(
                    "RoomKit | OpenAI + Anam | {timestamp}",
                    position="bottom-left",
                    font_scale=0.4,
                ),
            ],
        ),
        encoder=PyAVVideoEncoder(fps=25, bitrate=3_000_000, preset="medium"),
        connecting_frame=make_text_frame("Connecting...\nPlease wait"),
        provider_sample_rate=24000,
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a helpful AI assistant on a video call. "
            "Respond in the same language as the user. "
            "Keep responses conversational and concise.",
        ),
        voice=os.environ.get("OPENAI_VOICE", "alloy"),
        on_transcription=lambda role, text, _: logger.info("[%s] %s", role.upper(), text),
    )

    # --- Start ----------------------------------------------------------------
    await sip.start()
    logger.info(
        "SIP + OpenAI Realtime + Anam Avatar on 0.0.0.0:%s",
        os.environ.get("SIP_PORT", "5060"),
    )
    logger.info("Call this SIP endpoint with a video phone.")
    logger.info("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()

    def _signal() -> None:
        if stop.is_set():
            raise SystemExit(1)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal)
    await stop.wait()

    logger.info("Shutting down (Ctrl+C again to force)...")
    try:
        await asyncio.wait_for(bridge.close(), timeout=5.0)
    except TimeoutError:
        logger.warning("Bridge close timed out")
    await sip.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
