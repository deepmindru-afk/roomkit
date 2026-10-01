"""RoomKit -- SIP send video: answer a SIP call and stream a test pattern to the caller.

Answers incoming SIP calls and sends the caller a moving H.264 test pattern
(a background colour that changes every two seconds and a white bar sweeping
across the picture), so you can see the outbound video path work live:

    numpy RGB frame -> PyAVVideoEncoder (H.264 NAL units)
        -> SIPVideoBackend.send_video() -> RTP -> caller

Each call gets its own encoder: an H.264 stream is stateful, so callers
cannot share one. The caller's audio and video are ignored and no audio
is sent back: the caller hears silence. A call that offers no H.264 video
(no ``m=video`` line) is answered but receives nothing.

The SIP backend is used on its own: there is no room, channel or AI here.
See sip_video_call.py for routing a SIP video call into a RoomKit room.

Prerequisites:
    pip install roomkit[sip,video]

Run with:
    uv run python examples/sip_send_video.py

Then call ``sip:video@<this-host>:5060`` over UDP from a softphone with
H.264 video enabled (e.g. Linphone with the H.264 codec on) and start
the call with video.

Environment variables:
    SIP_PORT         SIP listener port, UDP (default: 5060)
    RTP_IP           IP to bind RTP on (default: 0.0.0.0; the SDP then
                     advertises the resolved local IP)
    RTP_PORT_START   Start of the UDP port range for RTP (default: 10000)
    RTP_PORT_END     End of that range, exclusive (default: 20000); each call
                     takes an even port for audio and one for video, plus the
                     odd port above each for RTCP

Press Ctrl+C to stop: active calls are hung up.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio
import os
import signal
from collections.abc import AsyncIterator

import numpy as np
from shared import setup_logging

from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.base import VideoChunk, VideoSession, VideoSessionState
from roomkit.video.pipeline.encoder import PyAVVideoEncoder
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import VoiceSession

logger = setup_logging("sip_send_video")

WIDTH, HEIGHT, FPS = 640, 480, 15
COLORS = [(200, 30, 30), (30, 160, 30), (30, 60, 200), (200, 170, 20)]
H264_IDR = 5


def draw_test_pattern(index: int) -> bytes:
    """Return frame *index* of the test pattern as RGB24 bytes."""
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = COLORS[(index // (2 * FPS)) % len(COLORS)]
    bar_x = (index * 8) % WIDTH
    frame[:, bar_x : bar_x + 16] = 255
    return frame.tobytes()


async def test_pattern_chunks(session: VideoSession) -> AsyncIterator[VideoChunk]:
    """Encode the test pattern in real time, one H.264 NAL unit per chunk.

    Runs until the call ends. The encoder belongs to this call only.
    """
    encoder = PyAVVideoEncoder(width=WIDTH, height=HEIGHT, fps=FPS)
    loop = asyncio.get_running_loop()
    next_frame_at = loop.time()
    index = 0
    try:
        while session.state is VideoSessionState.ACTIVE:
            raw = VideoFrame(
                data=draw_test_pattern(index), codec="raw_rgb24", width=WIDTH, height=HEIGHT
            )
            # One chunk per frame: its NAL units as an Annex B access unit,
            # which the backend sends together (marker bit on the last packet).
            nals = encoder.encode(raw)
            if nals:
                yield VideoChunk(
                    data=b"".join(b"\x00\x00\x00\x01" + nal for nal in nals),
                    width=WIDTH,
                    height=HEIGHT,
                    timestamp_ms=index * 1000 / FPS,
                    keyframe=any((nal[0] & 0x1F) == H264_IDR for nal in nals),
                )
            index += 1
            next_frame_at += 1 / FPS
            await asyncio.sleep(max(0.0, next_frame_at - loop.time()))
    finally:
        encoder.close()
        logger.info("Call %s: sent %d frames", session.id[:8], index)


async def main() -> None:
    sip_port = int(os.environ.get("SIP_PORT", "5060"))
    rtp_ip = os.environ.get("RTP_IP", "0.0.0.0")  # nosec B104
    rtp_port_start = int(os.environ.get("RTP_PORT_START", "10000"))
    rtp_port_end = int(os.environ.get("RTP_PORT_END", "20000"))

    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", sip_port),  # nosec B104
        local_rtp_ip=rtp_ip,
        rtp_port_start=rtp_port_start,
        rtp_port_end=rtp_port_end,
        supported_video_codecs=["H264"],  # the only codec the encoder produces
    )
    senders: set[asyncio.Task[None]] = set()

    def on_call(session: VoiceSession) -> None:
        caller = session.metadata.get("caller", "unknown")
        video_session = backend.get_video_session(session.id)
        if video_session is None:
            logger.info("Call from %s offers no H.264 video: nothing to send", caller)
            return
        logger.info("Call %s from %s: sending the test pattern", session.id[:8], caller)
        task = asyncio.create_task(
            backend.send_video(video_session, test_pattern_chunks(video_session))
        )
        senders.add(task)
        task.add_done_callback(senders.discard)

    backend.on_call(on_call)
    await backend.start()

    logger.info("Listening for SIP calls on UDP port %d", sip_port)
    logger.info("Video: H.264 test pattern %dx%d @ %d fps", WIDTH, HEIGHT, FPS)
    logger.info("Press Ctrl+C to stop.")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    logger.info("Stopping...")
    await backend.close()  # hangs up active calls
    for task in senders:
        task.cancel()
    await asyncio.gather(*senders, return_exceptions=True)
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
