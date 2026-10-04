"""A websocket client that ``auth`` refuses is closed, not left open (RMK-408).

Cleaning the stream's record alone forgot the socket without closing it: the
client kept it, its emit loops running, while it no longer counted against
``concurrency_limit``. Both FastRTC entry points close it: the realtime
transport (through ``reject_connection``) and the voice backend.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

pytest.importorskip("roomkit.webrtc", reason="roomkit[fastrtc] transport deps not installed")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, mount_fastrtc_voice  # noqa: E402
from roomkit.voice.realtime.fastrtc_transport import (  # noqa: E402
    FastRTCRealtimeTransport,
    mount_fastrtc_realtime,
)


async def _refuse(_ctx: Any) -> None:
    return None


def _closed_by_the_server(client: TestClient, path: str) -> bool:
    """Open a websocket client, start it, and say whether the server closes it
    within 3 s. Read on a thread of its own: a socket left open may send
    nothing, and the test client's read would never return."""
    closed = threading.Event()

    def read_until_close(ws: Any) -> None:
        try:
            while ws.receive()["type"] != "websocket.close":
                pass
        except WebSocketDisconnect:
            pass
        closed.set()

    with client.websocket_connect(f"{path}/websocket/offer") as ws:
        ws.send_json({"event": "start", "websocket_id": "ws-1"})
        threading.Thread(target=read_until_close, args=(ws,), daemon=True).start()
        return closed.wait(timeout=3.0)


def test_the_realtime_transport_closes_a_refused_websocket_client() -> None:
    transport = FastRTCRealtimeTransport()
    app = FastAPI()
    mount_fastrtc_realtime(app, transport, auth=_refuse, concurrency_limit=1)
    stream = transport._stream
    assert stream is not None

    with TestClient(app) as client:
        assert _closed_by_the_server(client, "/rtc-realtime")
        time.sleep(0.2)

    assert stream.connections == {}
    assert transport._handlers == {}


def test_the_voice_backend_closes_a_refused_websocket_client() -> None:
    backend = FastRTCVoiceBackend()
    app = FastAPI()
    mount_fastrtc_voice(app, backend, path="/voice", auth=_refuse)

    with TestClient(app) as client:
        assert _closed_by_the_server(client, "/voice")
        time.sleep(0.2)

    assert backend._rejections == set()
