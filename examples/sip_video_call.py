"""RoomKit — SIP audio+video call handler.

Accept incoming SIP calls with audio and video, transcribe speech with
a mock STT provider, and deliver H.264 video frames to a vision callback.
Audio-only calls are handled transparently — video is added when the remote
party offers it.

The mock STT does not listen to the audio: each utterance the energy VAD
detects is "transcribed" as a canned text ("Hello", then "How can I help
you?"), which shows the audio path end to end without any API key.

Audio flow:
    SIP INVITE → SDP negotiation (A/V) → RTP audio → Pipeline (energy VAD)
        → mock STT → ON_TRANSCRIPTION → print

Video flow:
    SIP INVITE → SDP negotiation → RTP video → H.264 NAL → video tap → print

Prerequisites:
    pip install roomkit[sip,video]
    (video brings numpy, which the voice channel's audio-level path needs,
    and PyAV for RECORDING_DIR)

Run with:
    uv run python examples/sip_video_call.py

Then send a SIP INVITE (with m=audio + m=video) to port 5060.

Environment variables:
    SIP_PORT         SIP listener port (default: 5060)
    RTP_IP           IP to bind RTP on (default: 0.0.0.0; the SDP then
                     advertises the resolved local IP)
    RTP_PORT_START   First RTP port to allocate, below 20000 (default: 10000)
    RECORDING_DIR    Record each call (MP4) into this directory;
                     unset = no recording
    DEBUG            Set to 1 for verbose logging

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio
import logging
import os

from shared import run_until_stopped, setup_logging

from roomkit import (
    AudioVideoChannel,
    HookResult,
    HookTrigger,
    RoomKit,
)
from roomkit.recorder.base import (
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.pyav import PyAVMediaRecorder
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig, EnergyVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logger = setup_logging("sip_video_call")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)


async def main() -> None:
    kit = RoomKit()

    # --- Configuration --------------------------------------------------------
    sip_port = int(os.environ.get("SIP_PORT", "5060"))
    rtp_ip = os.environ.get("RTP_IP", "0.0.0.0")
    rtp_port_start = int(os.environ.get("RTP_PORT_START", "10000"))

    # --- SIP A/V backend ------------------------------------------------------
    # SIPVideoBackend extends SIPVoiceBackend: audio-only calls work normally,
    # video is added when the remote party includes m=video in the INVITE.
    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", sip_port),
        local_rtp_ip=rtp_ip,
        rtp_port_start=rtp_port_start,
        supported_video_codecs=["H264", "VP8", "VP9"],
    )

    # --- Video callback — print frame info ------------------------------------
    frame_count = 0

    def on_video(session, frame):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 1:  # log every ~1s at 30fps
            logger.info(
                "Video frame #%d: codec=%s %s seq=%d ts=%.1fms",
                frame_count,
                frame.codec,
                "KEY" if frame.keyframe else "   ",
                frame.sequence,
                frame.timestamp_ms or 0,
            )

    # --- A/V channel (mock STT/TTS for demo) ------------------------------------
    # The VAD cuts the audio into utterances; without one the STT is never called.
    av = AudioVideoChannel(
        "voice",
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=EnergyVADProvider()),
    )
    av.add_video_media_tap(on_video)
    kit.register_channel(av)

    # --- Hooks: print transcriptions ------------------------------------------
    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        print(f"\n>>> {event.text}\n")
        return HookResult.block("demo — no AI provider")

    # --- Room recorder (opt-in) -----------------------------------------------
    recording_dir = os.environ.get("RECORDING_DIR", "")
    recorders: list[RoomRecorderBinding] = []
    if recording_dir:
        recorders.append(
            RoomRecorderBinding(
                recorder=PyAVMediaRecorder(),
                config=MediaRecordingConfig(storage=recording_dir, video_codec="libx264"),
            )
        )

    # --- Route incoming SIP calls to rooms ------------------------------------
    async def on_call(session: VoiceSession) -> None:
        room_id = session.metadata.get("room_id", session.id)
        has_video = session.metadata.get("has_video", False)
        caller = session.metadata.get("caller", "unknown")

        logger.info(
            "Incoming call: room=%s, caller=%s, video=%s",
            room_id,
            caller,
            has_video,
        )

        await kit.create_room(room_id=room_id, recorders=recorders)
        await kit.attach_channel(room_id, "voice")
        await kit.join(room_id, "voice", session=session)

    backend.on_call(on_call)

    # on_call_disconnected fires on every BYE; SIPVideoBackend.on_client_disconnected
    # only reports the end of a call's video session, so an audio-only call would
    # never close its room.
    def on_call_ended(session: VoiceSession) -> None:
        """Close the room (and finalize any recording) when the SIP call ends."""
        logger.info("Call ended: session=%s, video_frames=%d", session.id, frame_count)
        asyncio.create_task(kit.close_room(session.room_id))

    backend.on_call_disconnected(on_call_ended)

    # --- Start ----------------------------------------------------------------
    await backend.start()

    logger.info("SIP A/V backend listening on 0.0.0.0:%d", sip_port)
    if recording_dir:
        logger.info("Recording each call into %s", Path(recording_dir).resolve())
    else:
        logger.info("Recording off (set RECORDING_DIR to record calls).")
    logger.info("Send a SIP INVITE with m=audio + m=video to test.")
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C --------------------------------------------
    async def cleanup() -> None:
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
