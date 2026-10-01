"""``mount_websocket_video`` accepts connections (RMK-353).

The endpoint imported ``WebSocket`` inside ``mount_websocket_video`` while the
module postpones annotations, so FastAPI could not resolve the parameter's
type, took ``websocket`` for a query parameter and refused every connection
with a 403 before the handler ran.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from roomkit.video.backends.websocket import (  # noqa: E402
    WebSocketVideoBackend,
    mount_websocket_video,
)


def test_a_client_connects_and_gets_a_session() -> None:
    app = FastAPI()
    backend = WebSocketVideoBackend()
    sessions: list[object] = []
    backend.on_session_ready(sessions.append)
    mount_websocket_video(app, backend, path="/video")

    with TestClient(app).websocket_connect("/video") as websocket:
        websocket.send_text('{"type": "ping"}')

    assert len(sessions) == 1
