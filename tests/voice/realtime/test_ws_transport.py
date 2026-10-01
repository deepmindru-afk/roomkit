"""Tests for WebSocketRealtimeTransport."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

from roomkit.voice.base import VoiceSession


def _make_session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(
        id=sid,
        room_id="room1",
        participant_id="p1",
        channel_id="ch1",
    )


def _load_module():
    """Import the transport module with websockets mocked."""
    fake_ws = SimpleNamespace(
        connect=lambda *a, **kw: None,
    )
    mods = {"websockets": fake_ws}
    with patch.dict(sys.modules, mods):
        import roomkit.voice.realtime.ws_transport as mod

        importlib.reload(mod)
        return mod


class TestWebSocketRealtimeTransport:
    def test_constructor_and_name(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        assert transport.name == "WebSocketRealtimeTransport"

    def test_default_audio_format(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        assert transport._audio_format == "binary"

    def test_binary_audio_format(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport(audio_format="binary")
        assert transport._audio_format == "binary"

    def test_callback_registration_audio(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()

        cb = lambda session, audio: None  # noqa: E731
        transport.on_audio_received(cb)
        assert cb in transport._audio_callbacks

    def test_callback_registration_disconnect(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()

        cb = lambda session: None  # noqa: E731
        transport.on_client_disconnected(cb)
        assert cb in transport._disconnect_callbacks

    async def test_disconnect_unknown_session_is_noop(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        session = _make_session("unknown")
        # Should not raise
        await transport.disconnect(session)

    async def test_close_empty_transport(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        await transport.close()

    async def test_send_audio_no_connection(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        session = _make_session("s1")
        # No websocket accepted, should be a no-op
        await transport.send_audio(session, b"\x00\x01\x02")

    async def test_constructor_with_auth(self):
        mod = _load_module()

        async def fake_auth(connection):
            return {"user_id": "u1"}

        transport = mod.WebSocketRealtimeTransport(authenticate=fake_auth)
        assert transport._authenticate is fake_auth


async def test_default_sends_pcm_binary_and_legacy_json_is_explicit():
    import base64
    import json
    from unittest.mock import AsyncMock

    for mode in (None, "base64_json"):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport(**({"audio_format": mode} if mode else {}))
        connection = AsyncMock()
        session = _make_session()
        transport._websockets[session.id] = connection
        audio = b"\x00\x01" * 80
        await transport.send_audio(session, audio)
        sent = connection.send.call_args.args[0]
        if mode is None:
            assert sent == audio
        else:
            assert base64.b64decode(json.loads(sent)["data"]) == audio


class _ClosedSocket:
    """A client socket the peer already closed, the way websockets reports it."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def send(self, _message: object) -> None:
        raise self._error


def _real_module():
    """The transport module loaded with the real websockets package."""
    import roomkit.voice.realtime.ws_transport as mod

    return importlib.reload(mod)


class TestSendingToAClosedConnection:
    """RMK-353: a normal hang-up no longer logs an ERROR traceback.

    The channel tells the client ``session_ended`` after the client closed
    its socket; the send fails with websockets' ConnectionClosedOK, which is
    the expected end of a session, not an error.
    """

    async def test_a_closed_connection_is_logged_at_debug(self, caplog):
        from websockets.exceptions import ConnectionClosedOK
        from websockets.frames import Close

        mod = _real_module()
        transport = mod.WebSocketRealtimeTransport()
        session = _make_session()
        closed = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), rcvd_then_sent=True)
        transport._websockets[session.id] = _ClosedSocket(closed)

        with caplog.at_level("DEBUG", logger="roomkit.voice.realtime.ws_transport"):
            await transport.send_message(session, {"type": "session_ended"})
            await transport.send_audio(session, b"\x00\x00")

        assert not [r for r in caplog.records if r.levelname == "ERROR"]
        assert "connection is closed" in caplog.text

    async def test_another_send_failure_still_logs_an_error(self, caplog):
        mod = _real_module()
        transport = mod.WebSocketRealtimeTransport()
        session = _make_session()
        transport._websockets[session.id] = _ClosedSocket(RuntimeError("boom"))

        with caplog.at_level("ERROR", logger="roomkit.voice.realtime.ws_transport"):
            await transport.send_message(session, {"type": "x"})

        assert [r for r in caplog.records if r.levelname == "ERROR"]
