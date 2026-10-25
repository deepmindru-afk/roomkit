"""PyAV video recorder — H.264/H.265 encoding to MP4.

Produces properly compressed MP4 files using FFmpeg via PyAV.
Supports software encoding (libx264) and hardware encoding
(h264_nvenc on NVIDIA GPUs).

Requires the ``av`` package::

    pip install roomkit[video]
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.video.recorder.base import (
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
    build_recording_path,
    store_finished_recording,
)

if TYPE_CHECKING:
    import numpy as np

    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.recorder.pyav")


def _import_av() -> Any:
    """Import PyAV, raising a clear error if missing."""
    try:
        import av

        return av
    except ImportError as exc:
        raise ImportError(
            "av (PyAV) is required for PyAVVideoRecorder. Install with: pip install roomkit[video]"
        ) from exc


def _pick_codec(config: VideoRecordingConfig, av_mod: Any) -> str:
    """Pick the best available codec."""
    codec = config.codec
    if codec == "auto":
        try:
            av_mod.codec.Codec("h264_nvenc", "w")
            return "h264_nvenc"
        except Exception:
            return "libx264"
    return codec


def _to_ndarray(frame: VideoFrame) -> np.ndarray[Any, Any]:
    """Convert VideoFrame raw bytes to numpy ndarray."""
    import numpy as _np

    return _np.frombuffer(frame.data, dtype=_np.uint8).reshape(frame.height, frame.width, 3)


class _ActiveRecording:
    __slots__ = ("container", "stream", "frame_count", "start_time", "path", "av", "config")

    def __init__(
        self,
        container: Any,
        stream: Any,
        path: str,
        av_mod: Any,
        config: VideoRecordingConfig,
    ) -> None:
        self.container = container
        self.stream = stream
        self.path = path
        self.av = av_mod
        self.config = config
        self.frame_count = 0
        self.start_time = time.monotonic()


def _flush_and_close(active: _ActiveRecording) -> None:
    """Flush encoder (if frames were written) and close container safely."""
    try:
        if active.frame_count > 0:
            for packet in active.stream.encode():
                active.container.mux(packet)
    finally:
        active.container.close()


def _finish(active: _ActiveRecording) -> tuple[str, int, dict[str, object]]:
    """Close the file, then store it as configured, even when closing raised.

    A flush or close that fails must not keep the file from its encryption:
    it would stay on the storage in the clear (RFC §17.6).
    """
    try:
        _flush_and_close(active)
    finally:
        stored = store_finished_recording(active.config, active.path)
    return stored


class PyAVVideoRecorder(VideoRecorder):
    """Video recorder using PyAV (FFmpeg) for H.264/H.265 encoding.

    Produces compressed MP4 files — 10-50x smaller than raw OpenCV output.
    Supports NVIDIA hardware encoding (h264_nvenc) when available.
    """

    def __init__(self) -> None:
        self._av = _import_av()
        self._active: dict[str, _ActiveRecording] = {}

    @property
    def name(self) -> str:
        return "PyAVVideoRecorder"

    def start(self, session: VideoSession, config: VideoRecordingConfig) -> VideoRecordingHandle:
        rec_id = uuid4().hex[:12]
        fmt_override = "mp4" if config.format == "mp4v" else None
        path = build_recording_path(session, config, format_override=fmt_override)

        codec = _pick_codec(config, self._av)
        container = self._av.open(path, mode="w")
        try:
            stream = container.add_stream(codec, rate=int(config.fps))
            stream.pix_fmt = "yuv420p"
            # Dimensions set on first frame (lazy)
            stream.width = 0
            stream.height = 0
        except Exception:
            container.close()
            raise

        handle = VideoRecordingHandle(
            id=rec_id,
            session_id=session.id,
            path=path,
        )
        self._active[rec_id] = _ActiveRecording(
            container=container, stream=stream, path=path, av_mod=self._av, config=config
        )
        logger.info("Recording started: %s → %s (codec=%s)", rec_id, path, codec)
        return handle

    def stop(self, handle: VideoRecordingHandle) -> VideoRecordingResult:
        handle.state = "stopped"
        active = self._active.pop(handle.id, None)
        if active is None:
            return VideoRecordingResult(id=handle.id)

        elapsed = time.monotonic() - active.start_time
        url, size, metadata = _finish(active)

        logger.info(
            "Recording stopped: %s (%d frames, %.1fs, %d bytes)",
            handle.id,
            active.frame_count,
            elapsed,
            size,
        )
        return VideoRecordingResult(
            id=handle.id,
            url=url,
            frame_count=active.frame_count,
            duration_seconds=round(elapsed, 2),
            format=os.path.splitext(active.path)[1].lstrip("."),
            size_bytes=size,
            metadata=metadata,
        )

    def tap_frame(self, handle: VideoRecordingHandle, frame: VideoFrame) -> None:
        active = self._active.get(handle.id)
        if active is None:
            return

        av = active.av

        # Set dimensions on first frame
        if active.stream.width == 0:
            active.stream.width = frame.width
            active.stream.height = frame.height

        # Build av.VideoFrame from raw pixels
        if frame.codec == "raw_rgb24":
            av_frame = av.VideoFrame.from_ndarray(_to_ndarray(frame), format="rgb24")
        elif frame.codec == "raw_bgr24":
            av_frame = av.VideoFrame.from_ndarray(_to_ndarray(frame), format="bgr24")
        else:
            logger.debug("Skipping encoded frame (codec=%s)", frame.codec)
            return

        av_frame.pts = active.frame_count
        for packet in active.stream.encode(av_frame):
            active.container.mux(packet)
        active.frame_count += 1

    def close(self) -> None:
        # A recording still open is stored as configured too, or closing the
        # recorder would leave it in the clear; one failing stops no other.
        for rec_id, active in list(self._active.items()):
            try:
                _finish(active)
            except Exception:
                logger.exception("Recording %s failed to close", rec_id)
            else:
                logger.info("Recording closed: %s", rec_id)
        self._active.clear()
