"""Tests for ScreenCaptureBackend."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from roomkit.video.base import VideoCapability, VideoSession, VideoSessionState
from roomkit.video.video_frame import VideoFrame

# ---------------------------------------------------------------------------
# Mock mss module
# ---------------------------------------------------------------------------

# Small on purpose: every grab copies the whole screen, and a frame of a real
# screen's size costs the test its bytes for nothing it checks.
WIDTH, HEIGHT = 64, 36
PIXEL_COUNT = WIDTH * HEIGHT
RGB_BYTES = bytes([0xFF, 0x00, 0x00] * PIXEL_COUNT)  # All red


def _make_screenshot(rgb: bytes = RGB_BYTES, w: int = WIDTH, h: int = HEIGHT) -> MagicMock:
    shot = MagicMock()
    shot.rgb = rgb
    shot.width = w
    shot.height = h
    return shot


def _mock_mss_module(screenshot: MagicMock | None = None) -> MagicMock:
    """Build a mock mss module."""
    mock = MagicMock()
    sct = MagicMock()
    sct.monitors = [
        {"left": 0, "top": 0, "width": 3840, "height": 2160},  # 0 = all
        {"left": 0, "top": 0, "width": WIDTH, "height": HEIGHT},  # 1 = primary
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # 2 = secondary
    ]
    sct.grab.return_value = screenshot or _make_screenshot()
    mock.mss.return_value.__enter__ = MagicMock(return_value=sct)
    mock.mss.return_value.__exit__ = MagicMock(return_value=False)
    return mock


def _collect(backend, count: int) -> tuple[list[VideoFrame], asyncio.Event]:
    """Gather the frames a backend delivers; the event is set at ``count``."""
    received: list[VideoFrame] = []
    enough = asyncio.Event()

    def collect(_session, frame: VideoFrame) -> None:
        received.append(frame)
        if len(received) >= count:
            enough.set()

    backend.on_video_received(collect)
    return received, enough


async def _until_thread(predicate, *, timeout: float = 5.0) -> None:
    """Poll what the capture thread changes, which nothing on the loop can await."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "the capture thread did not get there in time"
        await asyncio.sleep(0.005)


@pytest.fixture
def mock_mss():
    return _mock_mss_module()


@pytest.fixture
def backend(mock_mss):
    with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
        from roomkit.video.backends.screen import ScreenCaptureBackend

        return ScreenCaptureBackend(monitor=1, fps=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScreenCaptureBackendProperties:
    def test_name(self, backend):
        assert backend.name == "ScreenCaptureBackend"

    def test_capabilities(self, backend):
        assert backend.capabilities == VideoCapability.SCREEN_SHARE


class TestScreenCaptureBackendConfig:
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"fps": 0}, "fps must be >= 1"),
            ({"scale": 0.0}, "scale must be in"),
            ({"scale": 1.5}, "scale must be in"),
        ],
    )
    def test_an_invalid_option_is_refused(self, mock_mss, options, message):
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            with pytest.raises(ValueError, match=message):
                ScreenCaptureBackend(**options)


class TestScreenCaptureBackendConnect:
    async def test_connect_creates_session(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")

        assert isinstance(session, VideoSession)
        assert session.state == VideoSessionState.ACTIVE
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.metadata["monitor"] == 1
        assert session.metadata["backend"] == "ScreenCaptureBackend"

    async def test_connect_with_metadata(self, backend):
        session = await backend.connect(
            "room-1", "user-1", "video-1", metadata={"custom": "value"}
        )
        assert session.metadata["custom"] == "value"
        assert session.metadata["monitor"] == 1

    async def test_get_session(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        assert backend.get_session(session.id) is session
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions(self, backend):
        await backend.connect("room-1", "user-1", "video-1")
        await backend.connect("room-2", "user-2", "video-2")

        assert len(backend.list_sessions("room-1")) == 1
        assert len(backend.list_sessions("room-2")) == 1
        assert len(backend.list_sessions("room-3")) == 0


class TestScreenCaptureBackendDisconnect:
    async def test_disconnect_updates_state(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.disconnect(session)

        stored = backend.get_session(session.id)
        assert stored is not None
        assert stored.state == VideoSessionState.ENDED

    async def test_disconnect_fires_callback(self, backend):
        disconnected: list[VideoSession] = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.disconnect(session)

        assert len(disconnected) == 1

    async def test_close_cleans_all(self, backend):
        await backend.connect("room-1", "user-1", "video-1")
        await backend.connect("room-2", "user-2", "video-2")

        await backend.close()

        assert len(backend.list_sessions("room-1")) == 0
        assert len(backend.list_sessions("room-2")) == 0


class TestScreenCaptureBackendSendVideo:
    async def test_send_video_noop(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.send_video(session, b"\x00")  # no error


class TestScreenCaptureBackendCallbacks:
    async def test_session_ready_callback(self, backend):
        ready: list[VideoSession] = []
        backend.on_session_ready(lambda s: ready.append(s))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await backend.stop_capture(session)

        assert len(ready) == 1
        assert ready[0].id == session.id

    async def test_video_received_callback(self, backend):
        """Capture thread delivers frames via on_video_received."""
        received, enough = _collect(backend, 1)

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.wait_for(enough.wait(), timeout=5.0)
        await backend.stop_capture(session)

        frame = received[0]
        assert frame.codec == "raw_rgb24"
        assert frame.width == WIDTH
        assert frame.height == HEIGHT
        assert frame.keyframe is True
        assert frame.sequence == 0

    async def test_duplicate_start_capture_warns(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await backend.start_capture(session)  # should warn, not crash
        await backend.stop_capture(session)


class TestScreenCaptureBackendRegion:
    async def test_region_override(self, mock_mss):
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(region=(100, 200, 800, 600), fps=5)

        _, enough = _collect(backend, 1)

        # The grabbed region is a screenshot of its own size.
        sct = mock_mss.mss.return_value.__enter__.return_value
        sct.grab.return_value = _make_screenshot(bytes([0x00, 0xFF, 0x00] * 8 * 6), 8, 6)

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.wait_for(enough.wait(), timeout=5.0)
        await backend.stop_capture(session)

        # _resolve_monitor should have built the region dict
        grab_arg = sct.grab.call_args[0][0]
        assert grab_arg == {"left": 100, "top": 200, "width": 800, "height": 600}


class TestScreenCaptureBackendDiffSkipping:
    async def test_identical_frames_skipped(self, mock_mss):
        """With diff_threshold > 0, identical frames after the first are skipped."""
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(fps=30, diff_threshold=0.01)

        received, _ = _collect(backend, 1)
        sct = mock_mss.mss.return_value.__enter__.return_value

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await _until_thread(lambda: sct.grab.call_count >= 3)
        await backend.stop_capture(session)

        # Three identical screenshots, one frame delivered
        assert len(received) == 1

    async def test_different_frames_not_skipped(self, mock_mss):
        """When screenshots change significantly, frames are not skipped."""
        call_count = 0

        def alternating_grab(_monitor):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return _make_screenshot(bytes([0x00, 0x00, 0xFF] * PIXEL_COUNT))
            return _make_screenshot(RGB_BYTES)

        sct = mock_mss.mss.return_value.__enter__.return_value
        sct.grab.side_effect = alternating_grab

        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(fps=30, diff_threshold=0.01)

        _, enough = _collect(backend, 2)

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.wait_for(enough.wait(), timeout=5.0)
        await backend.stop_capture(session)

    async def test_no_diff_threshold_delivers_all(self, mock_mss):
        """With diff_threshold=0 (default), all frames are delivered."""
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(fps=30, diff_threshold=0.0)

        # Identical screenshots, every one delivered
        _, enough = _collect(backend, 2)

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.wait_for(enough.wait(), timeout=5.0)
        await backend.stop_capture(session)


class TestScreenCaptureBackendMonitorValidation:
    async def test_invalid_monitor_index_delivers_no_frames(self, mock_mss):
        """Invalid monitor index logs error and delivers no frames."""
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(monitor=99, fps=5)

        received, _ = _collect(backend, 1)
        context = mock_mss.mss.return_value

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await _until_thread(lambda: context.__exit__.called)
        await backend.stop_capture(session)

        # The thread left without grabbing anything
        assert received == []
        assert context.__enter__.return_value.grab.call_count == 0
