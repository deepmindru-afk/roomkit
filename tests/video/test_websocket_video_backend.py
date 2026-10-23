"""Tests for WebSocketVideoBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from roomkit.video.backends.websocket import (
    _HEADER_SIZE,
    _HEADER_STRUCT,
    WebSocketVideoBackend,
)
from roomkit.video.base import VideoSessionState


class TestWebSocketVideoBackendLifecycle:
    """Session lifecycle: connect, disconnect, close."""

    async def test_connect_creates_session(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")

        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.state == VideoSessionState.ACTIVE
        assert backend.get_session(session.id) is session

    async def test_disconnect_ends_session(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")
        await backend.disconnect(session)

        assert session.state == VideoSessionState.ENDED
        assert backend.get_session(session.id) is None

    async def test_list_sessions(self):
        backend = WebSocketVideoBackend()
        await backend.connect("room-1", "user-1", "video")
        await backend.connect("room-1", "user-2", "video")
        await backend.connect("room-2", "user-3", "video")

        assert len(backend.list_sessions("room-1")) == 2
        assert len(backend.list_sessions("room-2")) == 1

    async def test_close_clears_all(self):
        backend = WebSocketVideoBackend()
        await backend.connect("room-1", "user-1", "video")
        await backend.connect("room-1", "user-2", "video")

        await backend.close()

        assert len(backend.list_sessions("room-1")) == 0

    async def test_name_property(self):
        backend = WebSocketVideoBackend()
        assert backend.name == "WebSocketVideo"


class TestWebSocketVideoBackendBinaryProtocol:
    """Binary frame parsing and serialization."""

    # flags: bit 0 keyframe, bits 1-3 codec id (0 h264, 1 vp8, 2 raw_rgb24).
    @pytest.mark.parametrize(
        ("flags", "codec", "keyframe"),
        [(0x01, "h264", True), (0x02, "vp8", False), (0x05, "raw_rgb24", True)],
    )
    async def test_a_binary_frame_is_parsed(self, flags: int, codec: str, keyframe: bool):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")
        backend._connection_sessions["conn-1"] = session.id

        received = []
        backend.on_video_received(lambda s, f: received.append((s, f)))

        payload = b"\x00\x00\x00\x01\x67"
        backend._handle_binary_frame("conn-1", _HEADER_STRUCT.pack(flags, 42) + payload)

        [(_, frame)] = received
        assert (frame.codec, frame.keyframe, frame.sequence) == (codec, keyframe, 42)
        assert frame.data == payload

    async def test_short_message_ignored(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")
        backend._connection_sessions["conn-1"] = session.id

        received = []
        backend.on_video_received(lambda s, f: received.append(f))

        # Too short — only header, no payload
        backend._handle_binary_frame("conn-1", b"\x00" * _HEADER_SIZE)
        assert len(received) == 0

    async def test_unknown_connection_ignored(self):
        backend = WebSocketVideoBackend()

        received = []
        backend.on_video_received(lambda s, f: received.append(f))

        header = _HEADER_STRUCT.pack(0, 0)
        backend._handle_binary_frame("unknown-conn", header + b"\x00" * 10)
        assert len(received) == 0


class TestWebSocketVideoBackendJsonConfig:
    """JSON control messages."""

    async def test_config_updates_dimensions(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")
        backend._connection_sessions["conn-1"] = session.id

        backend._handle_json_message(
            "conn-1",
            {
                "type": "config",
                "codec": "vp8",
                "width": 1280,
                "height": 720,
            },
        )

        config = backend._connection_config["conn-1"]
        assert config["codec"] == "vp8"
        assert config["width"] == 1280
        assert config["height"] == 720

    async def test_config_used_for_frame_dimensions(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")
        backend._connection_sessions["conn-1"] = session.id

        backend._handle_json_message(
            "conn-1",
            {
                "type": "config",
                "width": 1920,
                "height": 1080,
            },
        )

        received = []
        backend.on_video_received(lambda s, f: received.append(f))

        header = _HEADER_STRUCT.pack(0x01, 0)
        backend._handle_binary_frame("conn-1", header + b"\x00" * 10)

        assert received[0].width == 1920
        assert received[0].height == 1080


class TestWebSocketVideoBackendCallbacks:
    """Callback registration."""

    async def test_video_tap(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")
        backend._connection_sessions["conn-1"] = session.id

        tapped = []
        backend.add_video_tap(lambda s, f: tapped.append(f))

        header = _HEADER_STRUCT.pack(0, 0)
        backend._handle_binary_frame("conn-1", header + b"\x00" * 10)

        assert len(tapped) == 1

    async def test_session_ready_callback(self):
        backend = WebSocketVideoBackend()
        ready = []
        backend.on_session_ready(lambda s: ready.append(s))

        ws_mock = AsyncMock()
        await backend._on_client_connect("conn-1", ws_mock)

        assert len(ready) == 1

    async def test_disconnect_callback(self):
        backend = WebSocketVideoBackend()
        disconnected = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        ws_mock = AsyncMock()
        await backend._on_client_connect("conn-1", ws_mock)
        await backend._on_client_disconnect("conn-1")

        assert len(disconnected) == 1


class TestWebSocketVideoBackendSessionFactory:
    """Session factory for auto-creation on connect."""

    async def test_session_factory_called_on_connect(self):
        backend = WebSocketVideoBackend()

        custom_session = await backend.connect("custom-room", "custom-user", "video")

        async def factory(conn_id):
            return custom_session

        backend.set_session_factory(factory)

        ws_mock = AsyncMock()
        session = await backend._on_client_connect("conn-1", ws_mock)

        assert session is custom_session
        assert backend._websockets[custom_session.id] is ws_mock


class TestWebSocketVideoBackendSendVideo:
    """Outbound video sending."""

    async def test_send_video_bytes(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")

        ws_mock = AsyncMock()
        backend._websockets[session.id] = ws_mock

        await backend.send_video(session, b"\x00" * 100)

        ws_mock.send_bytes.assert_called_once()
        sent_data = ws_mock.send_bytes.call_args[0][0]
        assert len(sent_data) == _HEADER_SIZE + 100

        # Parse header
        flags, seq = _HEADER_STRUCT.unpack_from(sent_data, 0)
        assert seq == 0
        assert flags & 0x01 == 0  # not keyframe

    async def test_send_video_no_websocket(self):
        backend = WebSocketVideoBackend()
        session = await backend.connect("room-1", "user-1", "video")
        # No websocket registered — should not crash
        await backend.send_video(session, b"\x00" * 100)
