"""RoomKit -- Webcam censor with recording.

Demonstrates the video pipeline filter + room-level recording.
Vision periodically analyzes frames — when "person" is detected,
the censor filter replaces frames with black.  The recording
captures post-filter frames, so censored content never reaches
the MP4 file.

Pipeline:  Camera → [CensorFilter → CensorStateProbe → Watermark] → taps (recorder, vision)

With --yolo:
    Camera → [YOLODetectorFilter → CensorFilter → ...] → taps (recorder, vision)
    YOLO detects objects every frame, updating labels_detected.
    CensorFilter reads labels_detected and censors immediately.

CensorStateProbe is a small filter defined below: it reads the public
``FilterContext.censoring`` flag the censor filter sets, counts censored
frames, and emits an ON_VIDEO_DETECTION event (kind ``"censor"``) each
time censoring starts or stops.

Output:  <temp dir>/roomkit-webcam-censor/room_*.mp4 (censored sections
are black).  Set RECORDING_DIR to choose another directory; the path is
printed at start.

Prerequisites:
    pip install roomkit[local-video,video]

    # For Gemini vision (--gemini):
    pip install roomkit[gemini]
    export GEMINI_API_KEY=AIza...

    # For YOLO detection (--yolo):
    pip install roomkit[yolo]
    # The weights (yolo26n.pt) are downloaded on first use into
    # <temp dir>/roomkit-yolo/, or to the path in YOLO_WEIGHTS.

Environment variables:
    GEMINI_API_KEY   (--gemini only) Gemini API key
    RECORDING_DIR    (optional) where the MP4 goes
    YOLO_WEIGHTS     (optional, --yolo) path of the YOLO weights file

Run with:
    uv run python examples/webcam_censor.py                           # mock vision
    uv run python examples/webcam_censor.py --gemini                  # real vision
    uv run python examples/webcam_censor.py --yolo                    # YOLO detection
    uv run python examples/webcam_censor.py --effect cartoon          # cartoon effect
    uv run python examples/webcam_censor.py --yolo --effect sepia     # YOLO + sepia

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import asyncio
import os
import tempfile

from shared import require_env, run_until_stopped, setup_logging

from roomkit import (
    FrameworkEvent,
    HookExecution,
    HookTrigger,
    RoomKit,
    VideoChannel,
    VideoDetectionEvent,
)
from roomkit.models.session_event import SessionStartedEvent
from roomkit.recorder.base import MediaRecordingConfig, RoomRecorderBinding
from roomkit.recorder.pyav import PyAVMediaRecorder
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.video.pipeline import VideoPipelineConfig, VideoTransformProvider
from roomkit.video.pipeline.filter.base import FilterContext, FilterEvent, VideoFilterProvider
from roomkit.video.pipeline.filter.censor import CensorVideoFilter
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.video.pipeline.filter.yolo import YOLODetectorFilter
from roomkit.video.pipeline.transform.effects import VideoEffectTransform
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import VisionProvider
from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider
from roomkit.video.vision.mock import MockVisionProvider

setup_logging("webcam_censor")


class CensorStateProbe(VideoFilterProvider):
    """Report the censor filter's state through its public surface.

    Placed right after :class:`CensorVideoFilter`, it reads
    ``FilterContext.censoring`` on every frame, counts the censored ones,
    and appends a ``"censor"`` :class:`FilterEvent` on each transition —
    the channel delivers it to ``ON_VIDEO_DETECTION`` hooks.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.censored = 0
        self.censoring = False

    @property
    def name(self) -> str:
        return "censor-probe"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        self.frames += 1
        if context.censoring:
            self.censored += 1
        if context.censoring != self.censoring:
            self.censoring = context.censoring
            context.events.append(
                FilterEvent(
                    kind="censor",
                    data=VideoDetectionEvent(
                        kind="censor",
                        labels=sorted(context.labels_detected),
                        metadata={"censoring": context.censoring},
                        frame_sequence=frame.sequence,
                    ),
                )
            )
        return frame


def _recording_dir() -> Path:
    return Path(
        os.environ.get("RECORDING_DIR") or Path(tempfile.gettempdir()) / "roomkit-webcam-censor"
    )


def _yolo_weights() -> str:
    """Absolute weights path, so ultralytics never downloads into the CWD."""
    default = Path(tempfile.gettempdir()) / "roomkit-yolo" / "yolo26n.pt"
    return str(Path(os.environ.get("YOLO_WEIGHTS") or default).resolve())


def _build_vision(args: argparse.Namespace) -> VisionProvider:
    if args.gemini:
        env = require_env("GEMINI_API_KEY")
        return GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=env["GEMINI_API_KEY"],
                model="gemini-3.8-flash",
                prompt=(
                    "List the objects you see. Include 'person' if any "
                    "human is visible. Respond with comma-separated labels only."
                ),
            )
        )
    return MockVisionProvider(
        descriptions=[
            "Empty room with a desk",
            "A person sitting at the desk",
            "A person waving at the camera",
            "Empty room, person has left",
            "Still empty",
            "A person walking into the room",
        ],
        labels=[
            ["desk", "room"],
            ["person", "desk"],
            ["person", "gesture"],
            ["desk", "room"],
            ["room"],
            ["person", "room"],
        ],
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam Censor Demo")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini vision")
    parser.add_argument("--yolo", action="store_true", help="Use YOLO object detection")
    parser.add_argument(
        "--effect",
        default=None,
        choices=["grayscale", "sepia", "invert", "blur", "cartoon", "edges", "sketch", "pixelate"],
        help="Apply a visual effect to the video",
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS")
    parser.add_argument("--interval", type=int, default=3000, help="Vision interval ms")
    args = parser.parse_args()

    kit = RoomKit()

    backend = LocalVideoBackend(device=args.device, fps=args.fps, width=640, height=480)

    vision = _build_vision(args)
    censor = CensorVideoFilter(blocked_labels={"person"}, grace_frames=30)
    probe = CensorStateProbe()
    recording_dir = _recording_dir()

    # Build transform chain (optional visual effect)
    transforms: list[VideoTransformProvider] = []
    if args.effect:
        transforms.append(VideoEffectTransform(effect=args.effect))

    # Build filter chain: YOLO (optional) → Censor → Probe → Watermark
    filters: list[VideoFilterProvider] = []
    if args.yolo:
        yolo = YOLODetectorFilter(model=_yolo_weights(), confidence=0.5, every_n_frames=1)
        filters.append(yolo)
    filters.append(censor)
    filters.append(probe)
    filters.append(
        WatermarkFilter(
            text="RoomKit {timestamp}",
            position="bottom-right",
        )
    )

    # --- Recorder: PyAV → MP4 (records post-filter frames) -----------------
    recorder = PyAVMediaRecorder()

    video = VideoChannel(
        "video-main",
        backend=backend,
        pipeline=VideoPipelineConfig(
            transforms=transforms,
            filters=filters,
            vision=vision,
        ),
        vision_interval_ms=args.interval,
    )
    kit.register_channel(video)

    await kit.create_room(
        room_id="censor-demo",
        recorders=[
            RoomRecorderBinding(
                recorder=recorder,
                config=MediaRecordingConfig(storage=str(recording_dir)),
            ),
        ],
    )
    await kit.attach_channel("censor-demo", "video-main")

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED)
    async def on_started(event: SessionStartedEvent, ctx: object) -> None:
        if event.session is not None:
            print(f"  Session started: {event.session.id[:8]}...")

    @kit.hook(HookTrigger.ON_VIDEO_DETECTION, execution=HookExecution.ASYNC)
    async def on_censor_change(event: VideoDetectionEvent, ctx: object) -> None:
        if event.kind != "censor":
            return
        if event.metadata.get("censoring"):
            print(f"  >>> Censoring ON (frame {event.frame_sequence}, labels={event.labels})")
        else:
            print(f"  <<< Censoring OFF (frame {event.frame_sequence})")

    @kit.on("video_vision_result")
    async def on_vision(event: FrameworkEvent) -> None:
        data = event.data
        labels = data.get("labels", [])
        desc = data.get("description", "")
        status = "CENSORED" if probe.censoring else "LIVE"
        print(f"  [{status}] Vision: {desc[:80]} | labels={labels}")
        print(f"    Frames: {probe.frames} total, {probe.censored} censored")

    session = await kit.join("censor-demo", "video-main", participant_id="local-user")

    print("Webcam Censor + Recording Demo")
    print("=" * 60)
    print(f"Mode    : {'Gemini' if args.gemini else 'Mock'} vision")
    print(f"YOLO    : {'enabled' if args.yolo else 'disabled'}")
    print(f"Effect  : {args.effect or 'none'}")
    print("Filter  : censor (blocked: person)")
    print(f"Camera  : device {args.device} at 640x480 @ {args.fps}fps")
    print(f"Vision  : every {args.interval}ms")
    print(f"Record  : {recording_dir}/ (post-filter — censored = black)")
    print("Press Ctrl+C to stop.\n")

    await backend.start_capture(session)

    async def cleanup() -> None:
        print(f"\nDone. {probe.frames} frames, {probe.censored} censored.")
        await backend.stop_capture(session)
        await kit.leave(session)
        await kit.close_room("censor-demo")

    await run_until_stopped(kit, cleanup=cleanup)
    print(f"Recording saved to {recording_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
