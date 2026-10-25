"""Video recording providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.video.recorder.base import (
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
)
from roomkit.video.recorder.mock import MockVideoRecorder

if TYPE_CHECKING:
    from roomkit.video.recorder.opencv import OpenCVVideoRecorder
    from roomkit.video.recorder.pyav import PyAVVideoRecorder
else:
    __getattr__ = lazy_exports(
        __name__,
        {
            "OpenCVVideoRecorder": "roomkit.video.recorder.opencv",
            "PyAVVideoRecorder": "roomkit.video.recorder.pyav",
        },
    )


def get_opencv_recorder() -> type:
    """Lazy loader for OpenCVVideoRecorder (requires opencv-python-headless)."""
    from roomkit.video.recorder.opencv import OpenCVVideoRecorder

    return OpenCVVideoRecorder


def get_pyav_recorder() -> type:
    """Lazy loader for PyAVVideoRecorder (requires av)."""
    from roomkit.video.recorder.pyav import PyAVVideoRecorder

    return PyAVVideoRecorder


__all__ = [
    "MockVideoRecorder",
    "OpenCVVideoRecorder",
    "PyAVVideoRecorder",
    "VideoRecorder",
    "VideoRecordingConfig",
    "VideoRecordingHandle",
    "VideoRecordingResult",
    "get_opencv_recorder",
    "get_pyav_recorder",
]
