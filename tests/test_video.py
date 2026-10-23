"""Tests for video subsystem — VideoFrame, VideoBackend, VisionProvider."""

from __future__ import annotations

import pytest

from roomkit.video import (
    MockVideoBackend,
    MockVisionProvider,
    VideoCapability,
    VideoChunk,
    VideoFrame,
    VideoSession,
    VideoSessionState,
)
from roomkit.video.backends.base import VideoBackend
from roomkit.video.vision.base import VisionProvider, VisionResult

# ---------------------------------------------------------------------------
# VideoFrame
# ---------------------------------------------------------------------------


class TestVideoFrame:
    @pytest.mark.parametrize(
        ("codec", "encoded"),
        [
            ("h264", True),
            ("vp8", True),
            ("vp9", True),
            ("av1", True),
            ("raw_rgb24", False),
            ("raw_bgr24", False),
            ("raw_yuv420p", False),
            ("raw_nv12", False),
        ],
    )
    def test_a_codec_is_encoded_or_raw(self, codec: str, encoded: bool) -> None:
        frame = VideoFrame(data=b"\x00", codec=codec)

        assert (frame.is_encoded, frame.is_raw) == (encoded, not encoded)

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"codec": "invalid"}, "codec must be one of"),
            ({"data": "not bytes"}, "must be bytes"),
            ({"width": 0, "height": 480}, "must be positive"),
            ({"width": 640, "height": -1}, "must be positive"),
            ({"sequence": -1}, "sequence must be >= 0"),
        ],
    )
    def test_an_invalid_frame_is_refused(self, fields: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            VideoFrame(**{"data": b"\x00", **fields})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# VideoChunk
# ---------------------------------------------------------------------------


class TestVideoChunk:
    def test_invalid_codec_raises(self) -> None:
        with pytest.raises(ValueError, match="codec must be one of"):
            VideoChunk(data=b"\x00", codec="invalid")


# ---------------------------------------------------------------------------
# VideoSession
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VideoCapability
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MockVideoBackend
# ---------------------------------------------------------------------------


class TestMockVideoBackend:
    async def test_connect_creates_session(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        assert session.state == VideoSessionState.ACTIVE
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert backend.calls[-1].method == "connect"

    async def test_disconnect(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.disconnect(session)
        stored = backend.get_session(session.id)
        assert stored is not None
        assert stored.state == VideoSessionState.ENDED
        assert backend.calls[-1].method == "disconnect"

    async def test_send_video_bytes(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.send_video(session, b"\x00" * 100)
        assert len(backend.sent_video) == 1
        assert backend.sent_video[0] == (session.id, b"\x00" * 100)

    async def test_send_video_stream(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")

        async def chunks():
            yield VideoChunk(data=b"\x01" * 50)
            yield VideoChunk(data=b"\x02" * 50)

        await backend.send_video(session, chunks())
        assert len(backend.sent_video) == 1
        assert backend.sent_video[0][1] == b"\x01" * 50 + b"\x02" * 50

    async def test_list_sessions(self) -> None:
        backend = MockVideoBackend()
        await backend.connect("room-1", "user-1", "video-1")
        await backend.connect("room-1", "user-2", "video-1")
        await backend.connect("room-2", "user-3", "video-1")

        room1_sessions = backend.list_sessions("room-1")
        assert len(room1_sessions) == 2

        room2_sessions = backend.list_sessions("room-2")
        assert len(room2_sessions) == 1

    async def test_simulate_video_received(self) -> None:
        backend = MockVideoBackend()
        received_frames: list[tuple[VideoSession, VideoFrame]] = []

        def on_frame(session: VideoSession, frame: VideoFrame) -> None:
            received_frames.append((session, frame))

        backend.on_video_received(on_frame)
        session = await backend.connect("room-1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264", width=640, height=480)

        await backend.simulate_video_received(session, frame)
        assert len(received_frames) == 1
        assert received_frames[0][0].id == session.id
        assert received_frames[0][1] is frame

    async def test_simulate_session_ready(self) -> None:
        backend = MockVideoBackend()
        ready_sessions: list[VideoSession] = []

        def on_ready(session: VideoSession) -> None:
            ready_sessions.append(session)

        backend.on_session_ready(on_ready)
        session = await backend.connect("room-1", "user-1", "video-1")

        # connect fires session_ready automatically, but let's test explicit
        ready_sessions.clear()
        await backend.simulate_session_ready(session)
        assert len(ready_sessions) == 1

    async def test_simulate_client_disconnected(self) -> None:
        backend = MockVideoBackend()
        disconnected: list[VideoSession] = []

        def on_disconnect(session: VideoSession) -> None:
            disconnected.append(session)

        backend.on_client_disconnected(on_disconnect)
        session = await backend.connect("room-1", "user-1", "video-1")

        await backend.simulate_client_disconnected(session)
        assert len(disconnected) == 1

    async def test_close(self) -> None:
        backend = MockVideoBackend()
        await backend.connect("room-1", "user-1", "video-1")
        await backend.close()
        assert backend.calls[-1].method == "close"
        assert backend.list_sessions("room-1") == []

    async def test_get_session_not_found(self) -> None:
        backend = MockVideoBackend()
        assert backend.get_session("nonexistent") is None

    async def test_connect_with_metadata(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1", metadata={"codec": "vp9"})
        assert session.metadata == {"codec": "vp9"}
        assert backend.calls[-1].args["metadata"] == {"codec": "vp9"}


# ---------------------------------------------------------------------------
# VideoBackend ABC
# ---------------------------------------------------------------------------


class _MinimalBackend(VideoBackend):
    """Implements only what the ABC requires."""

    @property
    def name(self) -> str:
        return "minimal"

    async def disconnect(self, session: VideoSession) -> None:
        pass

    async def send_video(self, session, video) -> None:
        pass


class TestVideoBackendABC:
    def test_default_capabilities(self) -> None:
        backend = _MinimalBackend()
        assert backend.capabilities == VideoCapability.NONE
        assert backend.get_session("x") is None
        assert backend.list_sessions("x") == []

    async def test_connect_not_implemented(self) -> None:
        backend = _MinimalBackend()
        with pytest.raises(NotImplementedError, match="minimal does not implement connect"):
            await backend.connect("r", "p", "c")

    async def test_accept_not_implemented(self) -> None:
        backend = _MinimalBackend()
        session = VideoSession(id="s", room_id="r", participant_id="p", channel_id="c")
        with pytest.raises(NotImplementedError, match="minimal does not implement accept"):
            await backend.accept(session, None)

    async def test_close_is_noop(self) -> None:
        backend = _MinimalBackend()
        await backend.close()  # should not raise


# ---------------------------------------------------------------------------
# VisionProvider ABC
# ---------------------------------------------------------------------------


class TestVisionProviderABC:
    def test_default_name(self) -> None:
        class MyVision(VisionProvider):
            async def analyze_frame(self, frame):
                return VisionResult(description="test")

        provider = MyVision()
        assert provider.name == "MyVision"
        assert provider.supports_streaming is False

    async def test_warmup_and_close_are_noop(self) -> None:
        class MyVision(VisionProvider):
            async def analyze_frame(self, frame):
                return VisionResult(description="test")

        provider = MyVision()
        await provider.warmup()  # should not raise
        await provider.close()  # should not raise


# ---------------------------------------------------------------------------
# VisionResult
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FaceDetection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MockVisionProvider
# ---------------------------------------------------------------------------


class TestMockVisionProvider:
    async def test_default_descriptions(self) -> None:
        provider = MockVisionProvider()
        frame = VideoFrame(data=b"\x00", codec="h264")

        r1 = await provider.analyze_frame(frame)
        assert r1.description == "A video frame"
        assert r1.labels == ["person"]

        r2 = await provider.analyze_frame(frame)
        assert r2.description == "A person in a room"
        assert r2.labels == ["room"]

    async def test_tracks_calls(self) -> None:
        provider = MockVisionProvider()
        frame1 = VideoFrame(data=b"\x01", codec="h264")
        frame2 = VideoFrame(data=b"\x02", codec="vp9")

        await provider.analyze_frame(frame1)
        await provider.analyze_frame(frame2)
        assert len(provider.calls) == 2
        assert provider.calls[0] is frame1
        assert provider.calls[1] is frame2

    async def test_round_robin(self) -> None:
        provider = MockVisionProvider(descriptions=["A", "B"])
        frame = VideoFrame(data=b"\x00", codec="h264")

        results = [await provider.analyze_frame(frame) for _ in range(4)]
        assert [r.description for r in results] == ["A", "B", "A", "B"]

    async def test_analyze_stream(self) -> None:
        provider = MockVisionProvider(descriptions=["Frame"])

        async def frames():
            for i in range(5):
                yield VideoFrame(data=b"\x00", codec="h264", timestamp_ms=float(i * 1000))

        results = []
        async for result in provider.analyze_stream(frames(), interval_ms=2000):
            results.append(result)

        # With interval_ms=2000, frames at 0, 2000, 4000 should be analyzed
        assert len(results) == 3

    async def test_analyze_stream_without_timestamps(self) -> None:
        """When frames lack timestamps, synthetic ~30fps timing is used."""
        provider = MockVisionProvider(descriptions=["Frame"])

        async def frames():
            # 100 frames without timestamps (~3.3s at synthetic 30fps)
            for _ in range(100):
                yield VideoFrame(data=b"\x00", codec="h264")

        results = []
        async for result in provider.analyze_stream(frames(), interval_ms=1000):
            results.append(result)

        # At ~33ms per frame synthetic, 1000ms interval means ~1 per 30 frames
        # 100 frames / 30 ≈ 3-4 analyses (first at 0, then ~30, ~60, ~90)
        assert 3 <= len(results) <= 5


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------
