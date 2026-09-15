"""Server ICE credentials are owned by each connection, never by route mounting."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("roomkit.webrtc")
pytest.importorskip("fastapi")

from aiortc import RTCConfiguration  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from roomkit.voice.realtime.fastrtc_transport import (  # noqa: E402
    FastRTCRealtimeTransport,
    mount_fastrtc_realtime,
)
from roomkit.webrtc import Stream  # noqa: E402
from roomkit.webrtc import webrtc_connection_mixin as connections  # noqa: E402


def _offer(connection_id: str) -> dict[str, str]:
    return {"type": "offer", "sdp": "v=0\r\n", "webrtc_id": connection_id}


@pytest.fixture
def peer_configurations(monkeypatch: pytest.MonkeyPatch) -> list[RTCConfiguration | None]:
    """Record aiortc's input while leaving route admission and resolution real."""
    configurations: list[RTCConfiguration | None] = []

    def peer(*, configuration: RTCConfiguration | None) -> SimpleNamespace:
        configurations.append(configuration)
        return SimpleNamespace(
            on=lambda _event: lambda callback: callback,
            setRemoteDescription=AsyncMock(),
            createAnswer=AsyncMock(),
            setLocalDescription=AsyncMock(),
            localDescription=SimpleNamespace(sdp="answer", type="answer"),
        )

    monkeypatch.setattr(connections, "RTCPeerConnection", peer)
    monkeypatch.setattr(Stream, "connection_timeout", AsyncMock())
    return configurations


@pytest.mark.parametrize("configuration", [None, {"iceServers": []}])
async def test_static_configuration_preserves_defaults(
    configuration: dict[str, Any] | None,
) -> None:
    stream = Stream(lambda: None, server_rtc_configuration=configuration)
    result = await stream.resolve_server_rtc_configuration()
    if configuration is None:
        assert result is None
    else:
        assert isinstance(result, RTCConfiguration)
        assert result.iceServers == []


@pytest.mark.parametrize("kind", ["sync", "async", "bound", "async_object"])
async def test_callable_forms_resolve_per_connection(
    kind: str,
    peer_configurations: list[RTCConfiguration],
) -> None:
    calls: list[int] = []

    def resolve() -> dict[str, Any]:
        calls.append(len(calls) + 1)
        return {"iceServers": [{"urls": "turn:relay.example", "credential": str(calls[-1])}]}

    async def async_resolve() -> dict[str, Any]:
        return resolve()

    class Provider:
        def resolve(self) -> dict[str, Any]:
            return resolve()

        async def __call__(self) -> dict[str, Any]:
            return resolve()

    provider = Provider()
    callbacks = {
        "sync": resolve,
        "async": async_resolve,
        "bound": provider.resolve,
        "async_object": provider,
    }
    stream = Stream(lambda: None, server_rtc_configuration=callbacks[kind], concurrency_limit=2)
    assert calls == []
    for connection_id in ("first", "second"):
        answer = await stream.handle_offer(_offer(connection_id), lambda _: None)
        assert answer["type"] == "answer"
    assert [config.iceServers[0].credential for config in peer_configurations] == ["1", "2"]
    assert calls == [1, 2]


async def test_failed_callback_leaves_mounted_endpoint_available(
    peer_configurations: list[RTCConfiguration],
) -> None:
    callback = AsyncMock(side_effect=TimeoutError("private credential detail"))
    app = FastAPI()
    transport = FastRTCRealtimeTransport()
    mount_fastrtc_realtime(app, transport, rtc_configuration=callback)
    callback.assert_not_awaited()
    stream = transport._stream
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/rtc-realtime/webrtc/offer", json=_offer("retry"))
        assert response.status_code == 200
        assert response.json() == {
            "status": "failed",
            "meta": {"error": "rtc_configuration_failed"},
        }
        assert not stream.pcs
        assert not stream.handlers
        assert not stream.connections
        assert peer_configurations == []
        callback.side_effect = None
        callback.return_value = {"iceServers": []}
        response = await client.post("/rtc-realtime/webrtc/offer", json=_offer("retry"))
        assert response.json()["type"] == "answer"
        assert callback.await_count == 2
    await transport.close()


@pytest.mark.parametrize("duplicate", [False, True])
async def test_admission_is_atomic_after_awaiting_credentials(
    duplicate: bool,
    peer_configurations: list[RTCConfiguration],
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def resolve() -> dict[str, Any]:
        entered.set()
        await release.wait()
        return {"iceServers": []}

    stream = Stream(lambda: None, server_rtc_configuration=resolve, concurrency_limit=1)
    first = asyncio.create_task(stream.handle_offer(_offer("first"), lambda _: None))
    await entered.wait()
    second = asyncio.create_task(
        stream.handle_offer(_offer("first" if duplicate else "second"), lambda _: None)
    )
    release.set()
    results = await asyncio.gather(first, second)
    answers = [result for result in results if isinstance(result, dict)]
    failures = [result for result in results if not isinstance(result, dict)]
    assert len(answers) == len(failures) == 1
    assert answers[0]["type"] == "answer"
    assert json.loads(failures[0].body)["meta"]["error"] == (
        "connection_already_exists" if duplicate else "concurrency_limit_reached"
    )
    assert len(peer_configurations) == len(stream.pcs) == 1


async def test_client_configuration_uses_the_same_callable_resolution() -> None:
    callback = AsyncMock(return_value={"iceServers": []})
    stream = Stream(lambda: None, rtc_configuration=callback)
    assert await stream.get_rtc_configuration() == {"iceServers": []}
    assert await stream.resolve_rtc_configuration() == {"iceServers": []}
    assert callback.await_count == 2
