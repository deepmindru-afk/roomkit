"""Video decoder pipeline stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.video.pipeline.decoder.base import VideoDecoderProvider
from roomkit.video.pipeline.decoder.mock import MockVideoDecoderProvider

if TYPE_CHECKING:
    from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder

__all__ = [
    "MockVideoDecoderProvider",
    "PyAVVideoDecoder",
    "VideoDecoderProvider",
]

__getattr__ = lazy_exports(__name__, {"PyAVVideoDecoder": "roomkit.video.pipeline.decoder.pyav"})
