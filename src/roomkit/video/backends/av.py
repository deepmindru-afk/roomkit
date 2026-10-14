"""Combined audio/video backends: a voice backend whose calls may carry video.

A transport that carries a call's audio and video together (SIP with an
``m=video`` line, RTP with a parallel video stream, WebRTC with both tracks)
is one call. Its lifecycle is the voice backend's: ``on_session_ready`` when a
call's audio path is live, ``on_client_disconnected`` when the call ends. The
video of its calls is served by the ``video`` facet, a :class:`VideoBackend`
whose sessions carry the call's session id (RFC §12.8.3).

One object does not implement both :class:`VoiceBackend` and
:class:`VideoBackend`: their lifecycle methods share names and differ in
meaning, so a channel registering its voice callbacks would be served the
video ones.
"""

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from roomkit.video.backends.base import VideoBackend
from roomkit.video.base import (
    VideoCapability,
    VideoChunk,
    VideoDisconnectCallback,
    VideoReceivedCallback,
    VideoSession,
    VideoSessionReadyCallback,
    VideoSessionState,
)
from roomkit.voice.backends.base import VoiceBackend

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.av")

__all__ = ["AudioVideoBackend", "CallVideoBackend"]


class AudioVideoBackend(VoiceBackend):
    """A voice backend whose calls may carry video.

    The call is the :class:`~roomkit.voice.base.VoiceSession`: the backend's
    own ``on_session_ready`` and ``on_client_disconnected`` report every call,
    with or without video. :attr:`video` serves the video of the calls that
    negotiated it, under the call's session id::

        backend.on_session_ready(on_call_ready)          # every call
        backend.video.on_session_ready(on_video_ready)   # calls with video
        backend.video.on_video_received(on_frame)
        VideoChannel("cam", backend=backend.video)
    """

    @property
    @abstractmethod
    def video(self) -> VideoBackend:
        """The video of this backend's calls (same session ids)."""
        ...

    # -- Deprecated: the video API moved to ``video`` -------------------------

    def _moved(self, old: str, new: str) -> None:
        warnings.warn(
            f"{type(self).__name__}.{old} is deprecated: use .video.{new} instead",
            DeprecationWarning,
            stacklevel=3,
        )

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        """Deprecated: use ``backend.video.on_video_received()``."""
        self._moved("on_video_received()", "on_video_received()")
        self.video.on_video_received(callback)

    def add_video_tap(self, callback: VideoReceivedCallback) -> None:
        """Deprecated: use ``backend.video.add_video_tap()``."""
        self._moved("add_video_tap()", "add_video_tap()")
        self.video.add_video_tap(callback)

    def get_video_session(self, session_id: str) -> VideoSession | None:
        """Deprecated: use ``backend.video.get_session()``."""
        self._moved("get_video_session()", "get_session()")
        return self.video.get_session(session_id)

    def list_video_sessions(self, room_id: str) -> list[VideoSession]:
        """Deprecated: use ``backend.video.list_sessions()``."""
        self._moved("list_video_sessions()", "list_sessions()")
        return self.video.list_sessions(room_id)

    async def send_video(
        self, session: VideoSession, video: bytes | AsyncIterator[VideoChunk]
    ) -> None:
        """Deprecated: use ``backend.video.send_video()``."""
        self._moved("send_video()", "send_video()")
        await self.video.send_video(session, video)

    def send_video_sync(self, session: VideoSession, frame: VideoFrame) -> None:
        """Deprecated: use ``backend.video.send_video_sync()``."""
        self._moved("send_video_sync()", "send_video_sync()")
        self.video.send_video_sync(session, frame)

    def request_keyframe(self, session: VideoSession) -> None:
        """Deprecated: use ``backend.video.request_keyframe()``."""
        self._moved("request_keyframe()", "request_keyframe()")
        self.video.request_keyframe(session)

    def set_video_passthrough(self, session_id: str, enabled: bool = True) -> None:
        """Deprecated: use ``backend.video.set_video_passthrough()``."""
        self._moved("set_video_passthrough()", "set_video_passthrough()")
        self.video.set_video_passthrough(session_id, enabled)

    @property
    def video_capabilities(self) -> VideoCapability:
        """Deprecated: use ``backend.video.capabilities``."""
        self._moved("video_capabilities", "capabilities")
        return self.video.capabilities


class CallVideoBackend(VideoBackend):
    """The video of an :class:`AudioVideoBackend`'s calls.

    Its sessions carry their call's id and live as long as their call: the
    owner registers one when its call negotiates video and ends it when the
    call ends. It creates no session of its own, and :meth:`disconnect` ends
    the call. Subclasses send video over their transport.
    """

    def __init__(self, owner: AudioVideoBackend) -> None:
        self._owner = owner
        self._sessions: dict[str, VideoSession] = {}
        self._received_callback: VideoReceivedCallback | None = None
        self._taps: list[VideoReceivedCallback] = []
        self._ready_callbacks: list[VideoSessionReadyCallback] = []
        self._ended_callbacks: list[VideoDisconnectCallback] = []

    @property
    def name(self) -> str:
        return f"{self._owner.name} video"

    # -- What the owner reports ------------------------------------------------

    def _add(self, session: VideoSession) -> None:
        """Hold the video session of a call, not yet reported live."""
        self._sessions[session.id] = session

    def _ready(self, session_id: str) -> None:
        """Report the call's video path live."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        for cb in list(self._ready_callbacks):
            try:
                cb(session)
            except Exception:
                logger.exception("Video session ready callback failed: %s", session_id[:8])

    def _started(self, session: VideoSession) -> None:
        """Hold the video session of a call and report it live."""
        self._add(session)
        self._ready(session.id)

    def _ended(self, session_id: str) -> VideoSession | None:
        """End the video of a call that ended, and report it."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.state = VideoSessionState.ENDED
        for cb in list(self._ended_callbacks):
            try:
                cb(session)
            except Exception:
                logger.exception("Video session ended callback failed: %s", session_id[:8])
        return session

    def _forget(self) -> None:
        """Drop what is left once the owner closed: its calls, ended first,
        reported their video ended."""
        self._sessions.clear()

    def _has_consumers(self) -> bool:
        return self._received_callback is not None or bool(self._taps)

    def _deliver(self, session_id: str, frame: VideoFrame) -> None:
        """Hand an inbound frame of a call's video to its consumers."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        if self._received_callback is not None:
            self._received_callback(session, frame)
        for tap in self._taps:
            tap(session, frame)

    # -- VideoBackend ----------------------------------------------------------

    async def disconnect(self, session: VideoSession) -> None:
        """End the call this video belongs to."""
        call = self._owner.get_session(session.id)
        if call is not None:
            await self._owner.disconnect(call)

    def get_session(self, session_id: str) -> VideoSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VideoSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._received_callback = callback

    def add_video_tap(self, callback: VideoReceivedCallback) -> None:
        self._taps.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:
        """Called with the video session when a call's video path is live."""
        self._ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:
        """Called with the video session when its call ends."""
        self._ended_callbacks.append(callback)

    @abstractmethod
    async def send_video(
        self, session: VideoSession, video: bytes | AsyncIterator[VideoChunk]
    ) -> None: ...
