"""Video resizer pipeline stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.video.pipeline.resizer.base import VideoResizerProvider
from roomkit.video.pipeline.resizer.mock import MockVideoResizerProvider

if TYPE_CHECKING:
    from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer
else:
    __getattr__ = lazy_exports(
        __name__, {"PyAVVideoResizer": "roomkit.video.pipeline.resizer.pyav"}
    )

__all__ = [
    "MockVideoResizerProvider",
    "PyAVVideoResizer",
    "VideoResizerProvider",
]
