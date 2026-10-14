"""FastRTC audio+video backend using WebRTC/WebSocket transport.

Extends :class:`FastRTCVoiceBackend` with video transport. Each
:meth:`connect` call creates both a voice session and a video session
(shared session ID), the video served by the backend's ``video`` facet.
Video frames flow through FastRTC's ``Stream`` with
``modality="audio-video"``.

Requires the ``fastrtc`` optional dependency::

    pip install roomkit[fastrtc]

Usage::

    from roomkit.video.backends.fastrtc import FastRTCVideoBackend, mount_fastrtc_av

    backend = FastRTCVideoBackend()
    channel = AudioVideoChannel("av", backend=backend, ...)
    kit.register_channel(channel)

    # Mount FastRTC A/V endpoints on FastAPI app (in lifespan). Pass `auth`
    # to authenticate /webrtc/offer, or allow_anonymous=True on a trusted net.
    mount_fastrtc_av(app, backend, path="/av", auth=my_auth)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from roomkit.video.backends.av import AudioVideoBackend, CallVideoBackend
from roomkit.video.base import VideoChunk, VideoSession, VideoSessionState
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.auth import AuthCallback
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend
from roomkit.voice.base import VoiceSession

if TYPE_CHECKING:
    import numpy as np
    from fastapi import FastAPI

logger = logging.getLogger("roomkit.video.fastrtc")

__all__ = ["FastRTCVideoBackend", "mount_fastrtc_av"]


class FastRTCVideoBackend(FastRTCVoiceBackend, AudioVideoBackend):
    """FastRTC backend for combined audio + video WebRTC transport.

    Extends :class:`FastRTCVoiceBackend` with a parallel video path, served
    by :attr:`video`. A session's video is live once its WebRTC connection
    is, with its audio. Audio-only users should use
    :class:`FastRTCVoiceBackend` directly.

    Args:
        video_width: Default video frame width.
        video_height: Default video frame height.
        video_queue_maxsize: Max pending frames per session video emit queue.
        **kwargs: Forwarded to :class:`FastRTCVoiceBackend`.
    """

    DEFAULT_VIDEO_QUEUE_MAXSIZE: int = 100

    def __init__(
        self,
        *,
        video_width: int = 640,
        video_height: int = 480,
        video_queue_maxsize: int = DEFAULT_VIDEO_QUEUE_MAXSIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._video_width = video_width
        self._video_height = video_height
        self._video_queue_maxsize = video_queue_maxsize

        # Video session tracking
        self._frame_sequences: dict[str, int] = {}
        self._video = _FastRTCVideo(self)

        # Outbound video emit queues: webrtc_id -> asyncio.Queue
        self._video_emit_queues: dict[str, asyncio.Queue[Any | None]] = {}

    @property
    def name(self) -> str:
        return "FastRTC-AV"

    @property
    def video(self) -> _FastRTCVideo:
        """The video of each session, carried by its WebRTC connection."""
        return self._video

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        # Audio session via parent
        voice_session = await super().connect(
            room_id, participant_id, channel_id, metadata=metadata
        )

        # Create matching video session with same ID
        video_session = VideoSession(
            id=voice_session.id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VideoSessionState.ACTIVE,
            metadata={
                "width": self._video_width,
                "height": self._video_height,
                "backend": "fastrtc-av",
            },
        )
        # Live once its WebRTC connection is (_register_webrtc), not yet.
        self._video._add(video_session)
        self._frame_sequences[voice_session.id] = 0

        logger.info(
            "A/V session created: session=%s, room=%s, participant=%s",
            voice_session.id,
            room_id,
            participant_id,
        )

        return voice_session

    async def disconnect(self, session: VoiceSession) -> None:
        sid = session.id

        # Clean up video state
        self._frame_sequences.pop(sid, None)

        # Clean up video emit queue
        ws_id = session.metadata.get("websocket_id")
        if ws_id:
            self._video_emit_queues.pop(ws_id, None)

        self._video._ended(sid)

        # Audio cleanup via parent
        await super().disconnect(session)

    async def close(self) -> None:
        # Ending each session (disconnect()) ends its video, reported.
        await super().close()
        self._video._forget()
        self._frame_sequences.clear()
        self._video_emit_queues.clear()

    # -------------------------------------------------------------------------
    # Inbound video
    # -------------------------------------------------------------------------

    def _handle_video_frame(
        self,
        websocket_id: str,
        video_data: np.ndarray[Any, Any],
        width: int,
        height: int,
    ) -> None:
        """Called by FastRTC handler with raw video frame data.

        Converts numpy HWC array to VideoFrame and fires callbacks.
        """
        if not self._video._has_consumers():
            return

        session = self._find_session_by_websocket_id(websocket_id)
        if session is None or self._video.get_session(session.id) is None:
            return

        # Convert numpy HWC (RGB) array to raw bytes
        frame_bytes = video_data.tobytes()

        seq = self._frame_sequences.get(session.id, 0)
        self._frame_sequences[session.id] = seq + 1

        frame = VideoFrame(
            data=frame_bytes,
            codec="raw_rgb24",
            width=width,
            height=height,
            timestamp_ms=time.monotonic() * 1000.0,
            keyframe=(seq == 0),
            sequence=seq,
        )

        self._video._deliver(session.id, frame)

    # -------------------------------------------------------------------------
    # WebRTC registration override (creates video emit queue too)
    # -------------------------------------------------------------------------

    def _register_webrtc(self, webrtc_id: str, session_id: str) -> None:
        """Register a WebRTC session and create audio + video emit queues.

        The connection carries both tracks: the session's video is live now,
        right after its audio (reported by the parent).
        """
        self._video_emit_queues[webrtc_id] = asyncio.Queue(maxsize=self._video_queue_maxsize)
        super()._register_webrtc(webrtc_id, session_id)
        self._video._ready(session_id)


class _FastRTCVideo(CallVideoBackend):
    """The video of a :class:`FastRTCVideoBackend`'s sessions, on their WebRTC."""

    _owner: FastRTCVideoBackend

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        """Send video frames to a session.

        For WebRTC sessions, frames are queued for the handler's
        ``video_emit()`` method. WebSocket video is not currently
        supported (use WebSocketVideoBackend instead).
        """
        voice_session = self._owner._sessions.get(session.id)
        if voice_session is None:
            logger.warning("send_video: no voice session for %s", session.id)
            return

        ws_id = voice_session.metadata.get("websocket_id")
        if not ws_id:
            logger.warning("send_video: no websocket_id for session %s", session.id)
            return

        queue = self._owner._video_emit_queues.get(ws_id)
        if queue is None:
            logger.warning("send_video: no video emit queue for %s", session.id)
            return

        import numpy as _np

        if isinstance(video, bytes):
            # Store as numpy array with default dimensions
            arr = _np.frombuffer(video, dtype=_np.uint8)
            arr = arr.reshape((self._owner._video_height, self._owner._video_width, 3))
            self._enqueue_video_frame(queue, arr)
        else:
            async for chunk in video:
                arr = _np.frombuffer(chunk.data, dtype=_np.uint8)
                h = chunk.height
                w = chunk.width
                arr = arr.reshape((h, w, 3))
                self._enqueue_video_frame(queue, arr)

    @staticmethod
    def _enqueue_video_frame(queue: asyncio.Queue[Any | None], data: Any) -> None:
        """Put a video frame into an emit queue, dropping oldest if full."""
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(data)


def mount_fastrtc_av(
    app: FastAPI,
    backend: FastRTCVideoBackend,
    *,
    path: str = "/av",
    session_factory: Any = None,
    auth: AuthCallback | None = None,
    allow_anonymous: bool = False,
    concurrency_limit: int | None = 1,
) -> None:
    """Mount FastRTC audio+video endpoints on a FastAPI app.

    Like :func:`mount_fastrtc_voice` but uses ``modality="audio-video"``
    so that the browser PeerConnection carries both audio and video tracks.

    Args:
        app: FastAPI application.
        backend: The FastRTCVideoBackend instance.
        path: Base path for A/V endpoints (default: /av).
        session_factory: Async callable(websocket_id) -> VoiceSession.
        auth: Optional async auth callback. For WebRTC it authenticates the
            HTTP ``/webrtc/offer`` request before any peer connection is
            created.
        allow_anonymous: Required ``True`` when *auth* is ``None`` to expose
            ``/webrtc/offer`` without authentication; otherwise mounting raises
            ``ValueError``. Defaults to ``False``.
        concurrency_limit: Max concurrent connections (None for unlimited).
    """
    from roomkit.voice.backends._webrtc_auth import register_webrtc_offer_auth
    from roomkit.webrtc import AsyncAudioVideoStreamHandler, Stream

    backend._session_factory = session_factory

    class AVPassthroughHandler(AsyncAudioVideoStreamHandler):
        """Passes raw audio + video frames to the backend's callbacks.

        Each connection gets its own handler instance via ``copy()``.
        """

        def __init__(self) -> None:
            super().__init__()
            self._rejected = False
            self._auth_meta: dict[str, Any] | None = None
            self._webrtc_id: str | None = None
            self._is_webrtc = False

        def copy(self) -> AVPassthroughHandler:
            return AVPassthroughHandler()

        async def shutdown(self) -> None:
            """Called by FastRTC when the connection ends."""
            if self._webrtc_id:
                backend._webrtc_auth_meta.pop(self._webrtc_id, None)
                session = backend._find_session_by_websocket_id(self._webrtc_id)
                if session:
                    logger.info(
                        "A/V disconnected: session=%s webrtc_id=%s",
                        session.id,
                        self._webrtc_id,
                    )
                    for cb in backend._client_disconnected_callbacks:
                        try:
                            result = cb(session)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            logger.exception("Client disconnected callback error")
                    # disconnect() ends the session's video and reports it
                    # to the video facet's listeners.
                    await backend.disconnect(session)

        async def start_up(self) -> None:
            """Called once per connection — run auth and detect transport."""
            from roomkit.webrtc.utils import current_context

            ctx = current_context.get()
            if not ctx:
                return
            self._webrtc_id = ctx.webrtc_id
            self._is_webrtc = ctx.websocket is None
            logger.debug(
                "start_up: webrtc_id=%s is_webrtc=%s",
                self._webrtc_id,
                self._is_webrtc,
            )

            if auth is not None and ctx.websocket is not None:
                try:
                    result = await auth(ctx.websocket)
                    if result is None:
                        self._rejected = True
                        logger.warning("Auth rejected for id=%s", self._webrtc_id)
                        return
                    self._auth_meta = result
                except Exception:
                    self._rejected = True
                    logger.exception("Auth error for id=%s", self._webrtc_id)
                    return

        async def receive(self, frame: tuple[int, Any]) -> None:
            from roomkit.webrtc.utils import current_context

            if self._rejected:
                return
            ctx = current_context.get()
            if ctx and ctx.webrtc_id:
                await backend._receive_audio(ctx.webrtc_id, ctx.websocket, frame, self._auth_meta)

        async def video_receive(self, frame: Any) -> None:
            """Handle inbound video frames from FastRTC.

            FastRTC delivers video as a numpy array in HWC (RGB) layout.
            """
            if self._rejected or not self._webrtc_id:
                return

            if frame is None:
                return

            height, width = frame.shape[:2]
            backend._handle_video_frame(self._webrtc_id, frame, width, height)

        async def emit(self) -> tuple[int, Any] | None:
            """Return next audio frame for WebRTC playback."""
            if self._is_webrtc and self._webrtc_id:
                queue = backend._emit_queues.get(self._webrtc_id)
                if queue:
                    try:
                        return await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        return None
                await asyncio.sleep(0.1)
                return None

            await asyncio.sleep(0.1)
            return None

        async def video_emit(self) -> Any:
            """Return next video frame for WebRTC playback.

            Returns a numpy array in HWC (RGB) format. Must never
            return None — aiortc's VP8 encoder crashes on NoneType.
            Returns a blank frame when no outbound video is queued.
            """
            import numpy as _np

            if not self._is_webrtc or not self._webrtc_id:
                await asyncio.sleep(0.1)
                return _np.zeros((backend._video_height, backend._video_width, 3), dtype=_np.uint8)

            queue = backend._video_emit_queues.get(self._webrtc_id)
            if not queue:
                await asyncio.sleep(0.1)
                return _np.zeros((backend._video_height, backend._video_width, 3), dtype=_np.uint8)

            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
                if frame is None:
                    return _np.zeros(
                        (backend._video_height, backend._video_width, 3),
                        dtype=_np.uint8,
                    )
                return frame
            except TimeoutError:
                return _np.zeros((backend._video_height, backend._video_width, 3), dtype=_np.uint8)

    # Create FastRTC stream with audio+video modality.
    effective_limit = concurrency_limit if concurrency_limit is not None else 2**31
    stream = Stream(
        handler=AVPassthroughHandler(),
        modality="audio-video",
        mode="send-receive",
        concurrency_limit=effective_limit,
    )

    backend._stream = stream

    # Authenticate /webrtc/offer before stream.mount() so unauthorized offers
    # are rejected before any RTCPeerConnection is allocated (see voice mount).
    register_webrtc_offer_auth(app, path, stream, backend, auth, allow_anonymous=allow_anonymous)
    stream.mount(app, path=path)
    logger.info("FastRTC A/V backend mounted at %s", path)
