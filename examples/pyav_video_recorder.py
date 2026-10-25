"""RoomKit -- PyAV video recorder: H.264 webcam capture to compressed MP4.

Records webcam frames to a properly compressed H.264 MP4 file using
PyAVVideoRecorder (FFmpeg via PyAV). Produces 10-50x smaller files
than the OpenCV recorder. Supports NVIDIA hardware encoding (NVENC).

Prerequisites:
    pip install roomkit[local-video,video]

Run with:
    uv run python examples/pyav_video_recorder.py
    uv run python examples/pyav_video_recorder.py --duration 30
    uv run python examples/pyav_video_recorder.py --codec h264_nvenc
    uv run python examples/pyav_video_recorder.py --output ./my_recordings --fps 30
    uv run python examples/pyav_video_recorder.py --duration 5 --encrypt

The MP4 goes to ``<temp dir>/roomkit-recordings`` unless ``--output`` names
another directory; the path of the file is printed at the end.

By default the MP4 is written in the clear, so you can play it. ``--encrypt``
stores it encrypted at rest instead (RFC 17.6): ``VideoRecordingConfig`` takes
the same ``encryption=`` as the audio recorders, and the recorder hands the
finished file to it, which leaves only the ``.mp4.enc`` artifact. The cipher
below is a stand-in so the example needs no key management: it is NOT
encryption. A deployment passes its own ``RecordingEncryption``, or
``storage_encrypted_at_rest=True`` when the storage already encrypts.

Press Ctrl+C to stop early.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import asyncio
import contextlib
import logging
import signal
import tempfile

from shared import setup_logging

from roomkit import HookExecution, HookTrigger, RoomKit, SessionStartedEvent, VideoChannel
from roomkit.video import VideoFrame
from roomkit.video.backends import LocalVideoBackend
from roomkit.video.pipeline import VideoPipelineConfig
from roomkit.video.recorder import PyAVVideoRecorder, VideoRecordingConfig
from roomkit.voice.pipeline import RecordingEncryption

setup_logging("pyav_video_recorder", level=logging.WARNING)

DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "roomkit-recordings"


class ReversingEncryption(RecordingEncryption):
    """Stand-in cipher: reverses the bytes. Do not ship this.

    It shows the contract a real cipher honours: leave no plaintext behind,
    and return the path of the encrypted artifact.
    """

    @property
    def name(self) -> str:
        return "reversing-stand-in"

    def encrypt_file(self, path: str) -> str:
        source = Path(path)
        target = source.with_name(source.name + ".enc")
        target.write_bytes(source.read_bytes()[::-1])
        source.unlink()
        return str(target)


async def main() -> None:
    parser = argparse.ArgumentParser(description="PyAV Video Recorder Demo")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS")
    parser.add_argument("--duration", type=int, default=0, help="Record N seconds (0=Ctrl+C)")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--codec",
        default="auto",
        choices=["auto", "libx264", "h264_nvenc", "libx265"],
        help="Video codec (auto = NVENC if available, else libx264)",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Store the MP4 encrypted at rest (stand-in cipher, see the module docstring)",
    )
    args = parser.parse_args()

    kit = RoomKit()

    # --- Video backend: local webcam -----------------------------------------
    backend = LocalVideoBackend(device=args.device, fps=args.fps, width=640, height=480)

    # --- Recorder: PyAV (FFmpeg H.264) ---------------------------------------
    recorder = PyAVVideoRecorder()
    recording_config = VideoRecordingConfig(
        storage=args.output,
        format="mp4",
        codec=args.codec,
        fps=float(args.fps),
        encryption=ReversingEncryption() if args.encrypt else None,
    )

    # --- Video channel with pipeline recorder --------------------------------
    video = VideoChannel(
        "video-rec",
        backend=backend,
        pipeline=VideoPipelineConfig(
            recorder=recorder,
            recording_config=recording_config,
        ),
    )
    kit.register_channel(video)

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="recording-demo")
    await kit.attach_channel("recording-demo", "video-rec")

    # --- Hooks: log events ---------------------------------------------------
    frame_count = 0

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_started(event: SessionStartedEvent, ctx: object) -> None:
        print(f"  Session started: {event.participant_id} on {event.channel_id}")

    def on_frame(session: object, frame: VideoFrame) -> None:
        nonlocal frame_count
        frame_count += 1
        if frame_count % args.fps == 0:
            secs = frame_count // args.fps
            print(f"\r  Recording... {secs}s ({frame_count} frames)", end="", flush=True)

    video.add_media_tap(on_frame)

    # --- Connect and start ---------------------------------------------------
    session = await kit.join("recording-demo", "video-rec", participant_id="local-user")

    print("PyAV Video Recorder Demo")
    print("=" * 60)
    print(f"Codec: {args.codec}")
    print(f"Camera: device {args.device} at 640x480 @ {args.fps}fps")
    print(f"Output: {args.output}/{' (encrypted at rest)' if args.encrypt else ''}")
    if args.duration:
        print(f"Duration: {args.duration}s")
    else:
        print("Duration: until Ctrl+C")
    print()

    await backend.start_capture(session)

    # --- Wait for duration or Ctrl+C ----------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, stop.set)

    if args.duration:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=args.duration)
    else:
        await stop.wait()

    # --- Cleanup -------------------------------------------------------------
    print("\n\n  Stopping...")
    await backend.stop_capture(session)
    await kit.leave(session)
    await kit.close()

    # Show output info: the newest MP4 is this run's (file names start with
    # a random session id, so sorting by name would pick an older one).
    pattern = "*.mp4.enc" if args.encrypt else "*.mp4"
    files = sorted(Path(args.output).glob(pattern), key=lambda f: f.stat().st_mtime)
    if files:
        latest = files[-1]
        size_kb = latest.stat().st_size / 1024
        print(f"  Recorded {frame_count} frames → {latest} ({size_kb:.1f} KB)")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
