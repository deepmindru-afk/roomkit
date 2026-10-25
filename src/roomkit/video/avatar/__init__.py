"""Video avatar — lip-synced talking head generation from audio."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.video.avatar.base import AvatarProvider
from roomkit.video.avatar.mock import MockAvatarProvider
from roomkit.video.avatar.websocket import WebSocketAvatarProvider

if TYPE_CHECKING:
    from roomkit.video.avatar.musetalk import MuseTalkAvatarProvider
else:
    __getattr__, __dir__ = lazy_exports(
        __name__, {"MuseTalkAvatarProvider": "roomkit.video.avatar.musetalk"}
    )

__all__ = [
    "AvatarProvider",
    "MockAvatarProvider",
    "MuseTalkAvatarProvider",
    "WebSocketAvatarProvider",
]
