"""Room-level media recording package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecorder,
    MediaRecordingConfig,
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
    RoomRecorderBinding,
)
from roomkit.recorder.feed import RoomRecordingFeed
from roomkit.recorder.mock import MockMediaRecorder

if TYPE_CHECKING:
    from roomkit.recorder.pyav import PyAVMediaRecorder

__all__ = [
    "ChannelRecordingConfig",
    "MediaRecorder",
    "MediaRecordingConfig",
    "MediaRecordingHandle",
    "MediaRecordingResult",
    "MockMediaRecorder",
    "PyAVMediaRecorder",
    "RecordingTrack",
    "RoomRecorderBinding",
    "RoomRecordingFeed",
    "get_pyav_media_recorder",
]

__getattr__ = lazy_exports(__name__, {"PyAVMediaRecorder": "roomkit.recorder.pyav"})


def get_pyav_media_recorder() -> type:
    """Lazy-load ``PyAVMediaRecorder`` to avoid hard PyAV dependency."""
    from roomkit.recorder.pyav import PyAVMediaRecorder

    return PyAVMediaRecorder
