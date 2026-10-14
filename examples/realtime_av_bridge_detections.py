"""RoomKit — what a RealtimeAVBridge's video filters detect, handed to the application.

A ``RealtimeAVBridge`` runs the provider's video through its ``video_pipeline``
before encoding it for the caller. A filter that inspects frames (face touch,
an object in view) emits detections; ``on_video_detection`` receives each one,
its ``session`` set to the call it was seen on. Without the callback the
detections are drained and dropped.

This example drives a bridge with a mock audio+video provider, a mock voice
backend and a mock filter that detects on chosen frames — no API key, no
camera, no SIP endpoint — and logs what reaches the callback.

Run with:
    uv run python examples/realtime_av_bridge_detections.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging

from roomkit.video.events import VideoDetectionEvent
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.filter.mock_face_touch import MockFaceTouchFilter
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.bridge import RealtimeAVBridge
from roomkit.voice.realtime.mock import MockRealtimeAudioVideoProvider

logger = setup_logging("realtime_av_bridge_detections")

FRAMES = 6


def detection(region: str) -> VideoDetectionEvent:
    return VideoDetectionEvent(session=None, kind="face_touch", labels=[region], confidence=0.9)


def frame(sequence: int) -> VideoFrame:
    return VideoFrame(data=b"\x80" * 3, codec="raw_rgb24", width=1, height=1, sequence=sequence)


async def on_video_detection(event: VideoDetectionEvent) -> None:
    """Called for each detection; may be sync or async."""
    call = event.session.id if event.session is not None else "?"
    logger.info("Detected %s (%s) on call %s", event.kind, ", ".join(event.labels), call)


async def main() -> None:
    provider = MockRealtimeAudioVideoProvider()
    # The mock filter detects a face touch on frames 1 and 4.
    face_touch = MockFaceTouchFilter({1: [detection("cheek")], 4: [detection("chin")]})
    bridge = RealtimeAVBridge(
        provider,
        MockVoiceBackend(),
        video_pipeline=VideoPipelineConfig(filters=[face_touch]),
        on_video_detection=on_video_detection,
    )
    call = await bridge.connect(
        VoiceSession(id="call-1", room_id="room-1", participant_id="caller", channel_id="voice")
    )
    for sequence in range(FRAMES):
        await provider.simulate_video(call, frame(sequence))
    await asyncio.sleep(0)  # let the async callback run
    await bridge.close()
    logger.info("%d frames sent, the bridge is closed", FRAMES)


if __name__ == "__main__":
    asyncio.run(main())
