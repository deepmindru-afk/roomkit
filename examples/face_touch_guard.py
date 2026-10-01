"""Face touch guard — alerts when you touch your face.

Captures webcam video via a local camera backend and detects hand-to-face
contact using MediaPipe landmarks.  Fires ON_VIDEO_DETECTION hooks that
log warnings.  Inspired by FaceTouchGuard
(https://github.com/timpratim/faceguard).

Requires:
    pip install roomkit[local-video,mediapipe]
    A webcam (device 0).

Models:
    On the first analysed frame the filter downloads two MediaPipe models
    from Google Cloud Storage (the ``float16/latest`` builds, unpinned, so
    the bytes can change upstream) into ``~/.cache/roomkit/mediapipe/``:
    ``face_landmarker.task`` (~3.7 MB) and ``hand_landmarker.task``
    (~7.8 MB). Later runs reuse the cache; nothing is written to the
    current directory. To pre-fetch them, or to run offline:

        mkdir -p ~/.cache/roomkit/mediapipe && cd ~/.cache/roomkit/mediapipe
        curl -LO https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
        curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

    To use files you have checked yourself, pass their paths as
    ``FaceTouchConfig(face_model=..., hand_model=...)``: an existing file is
    used as-is and nothing is downloaded.

Run with:
    uv run python examples/face_touch_guard.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys

from shared import run_until_stopped, setup_logging

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.video import VideoChannel
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.video.events import VideoDetectionEvent
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.filter.mediapipe_face_touch import (
    FaceTouchConfig,
    FaceTouchFilter,
    FaceTouchSensitivity,
    FaceZone,
)

logger = setup_logging("face_touch_guard")


async def main() -> None:
    # The filter imports MediaPipe on the first frame; without it every
    # analysed frame would log the same ImportError. Fail once, up front.
    if importlib.util.find_spec("mediapipe") is None:
        sys.exit("mediapipe is not installed: pip install roomkit[mediapipe]")

    kit = RoomKit()

    # Local webcam backend
    backend = LocalVideoBackend(device=0, fps=15)

    # Configure face touch detection with high sensitivity
    face_touch_filter = FaceTouchFilter(
        FaceTouchConfig(
            sensitivity=FaceTouchSensitivity.HIGH,
            zones=frozenset(
                {
                    FaceZone.LEFT_CHEEK,
                    FaceZone.RIGHT_CHEEK,
                    FaceZone.CHIN,
                    FaceZone.MOUTH,
                }
            ),
            every_n_frames=3,
        )
    )

    pipeline_config = VideoPipelineConfig(filters=[face_touch_filter])

    video = VideoChannel(
        "video-cam",
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(video)

    # React to face touches
    @kit.hook(HookTrigger.ON_VIDEO_DETECTION, execution=HookExecution.ASYNC)
    async def on_detection(event: VideoDetectionEvent, ctx: dict) -> None:
        if event.kind != "face_touch":
            return
        zone = event.metadata.get("zone", "unknown")
        hand = event.metadata.get("hand", "unknown")
        count = event.metadata.get("touch_count", 0)
        logger.warning(
            "FACE TOUCH detected! zone=%s hand=%s confidence=%.2f count=%d",
            zone,
            hand,
            event.confidence,
            count,
        )

    # Create a room, attach channel, join session, start capture
    room = await kit.create_room("face-guard-room")
    await kit.attach_channel(room.id, "video-cam")
    session = await kit.join(room.id, "video-cam", participant_id="local-user")
    await backend.start_capture(session)

    logger.info("Face touch guard running — touch your face to trigger alerts")
    logger.info("Press Ctrl+C to stop")

    await run_until_stopped(kit)


if __name__ == "__main__":
    asyncio.run(main())
