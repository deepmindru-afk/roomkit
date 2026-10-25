"""Video transport backends.

Each backend is imported from here. A backend's module loads on first use,
with the transport library it needs; ``import roomkit`` loads none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports

if TYPE_CHECKING:
    from roomkit.video.backends.fastrtc import FastRTCVideoBackend, mount_fastrtc_av
    from roomkit.video.backends.local import LocalVideoBackend
    from roomkit.video.backends.rtp import RTPVideoBackend
    from roomkit.video.backends.screen import ScreenCaptureBackend
    from roomkit.video.backends.sip import SIPVideoBackend
    from roomkit.video.backends.websocket import WebSocketVideoBackend, mount_websocket_video
else:
    __getattr__ = lazy_exports(
        __name__,
        {
            "FastRTCVideoBackend": "roomkit.video.backends.fastrtc",
            "LocalVideoBackend": "roomkit.video.backends.local",
            "RTPVideoBackend": "roomkit.video.backends.rtp",
            "SIPVideoBackend": "roomkit.video.backends.sip",
            "ScreenCaptureBackend": "roomkit.video.backends.screen",
            "WebSocketVideoBackend": "roomkit.video.backends.websocket",
            "mount_fastrtc_av": "roomkit.video.backends.fastrtc",
            "mount_websocket_video": "roomkit.video.backends.websocket",
        },
    )

__all__ = [
    "FastRTCVideoBackend",
    "LocalVideoBackend",
    "RTPVideoBackend",
    "SIPVideoBackend",
    "ScreenCaptureBackend",
    "WebSocketVideoBackend",
    "mount_fastrtc_av",
    "mount_websocket_video",
]


def get_screen_capture_backend() -> type:
    """Get ScreenCaptureBackend class (requires mss).

    Install with: ``pip install roomkit[screen-capture]``
    """
    from roomkit.video.backends.screen import ScreenCaptureBackend

    return ScreenCaptureBackend


def get_rtp_video_backend() -> type:
    """Get RTPVideoBackend class (requires aiortp).

    Install with: ``pip install roomkit[rtp]``
    """
    from roomkit.video.backends.rtp import RTPVideoBackend

    return RTPVideoBackend


def get_sip_video_backend() -> type:
    """Get SIPVideoBackend class (requires aiosipua[rtp]).

    Install with: ``pip install roomkit[sip]``
    """
    from roomkit.video.backends.sip import SIPVideoBackend

    return SIPVideoBackend


def get_fastrtc_video_backend() -> type:
    """Get FastRTCVideoBackend class (requires fastrtc).

    Install with: ``pip install roomkit[fastrtc]``
    """
    from roomkit.video.backends.fastrtc import FastRTCVideoBackend

    return FastRTCVideoBackend


def get_websocket_video_backend() -> type:
    """Get WebSocketVideoBackend class.

    No extra dependencies required beyond ``fastapi``.
    """
    from roomkit.video.backends.websocket import WebSocketVideoBackend

    return WebSocketVideoBackend
