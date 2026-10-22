"""The delivery queue, the status bus and the realtime backend across a
connection a NAT or a load balancer forgot (half-open): each gives up the dead
connection within its bound and goes on with a new one (RMK-868). Set
REDIS_URL to run against Redis."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

import roomkit._redis as redis_bounds
import roomkit.delivery.redis as redis_delivery
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.redis import RedisDeliveryBackend
from roomkit.orchestration.status_bus import StatusEntry, StatusLevel
from roomkit.orchestration.status_redis import RedisStatusBackend
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.realtime.redis import RedisRealtimeBackend
from tests.redis_silencing import Silencing

pytestmark = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")


@pytest.fixture
def short_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bounds short enough for a test: a call gives up after 0.3 s, a read
    for new work waits 0.2 s, a pub/sub silent for 0.3 s is pinged and
    dropped 0.3 s after."""
    monkeypatch.setattr(redis_bounds, "CALL_BOUND_S", 0.3)
    monkeypatch.setattr(redis_bounds, "PING_AFTER_S", 0.3)
    monkeypatch.setattr(redis_bounds, "PONG_WAIT_S", 0.3)
    monkeypatch.setattr(redis_delivery, "_READ_SLICE_S", 0.2)


@pytest.fixture
async def proxy() -> Any:
    silencing = Silencing()
    yield silencing
    await silencing.close()


async def test_a_live_worker_keeps_its_lease_across_a_silenced_connection(
    short_bounds: None, proxy: Silencing
) -> None:
    """The worker's renewal gives the silenced connection up within its bound
    and renews on a new one: no other worker takes its item."""
    redis = pytest.importorskip("redis.asyncio")
    direct = redis.from_url(os.environ["REDIS_URL"])
    prefix = f"roomkit:test:{uuid4().hex}"
    # Renewed every 1.5 s; a renewal cut after 0.5 s is tried again 1.5 s on.
    held = RedisDeliveryBackend(await proxy.url(), stream_prefix=prefix, claim_idle_seconds=4.5)
    other = RedisDeliveryBackend(client=direct, stream_prefix=prefix, claim_idle_seconds=4.5)
    await direct.xgroup_create(held._pending_key, held._group, id="0", mkstream=True)
    try:
        await held.enqueue(DeliveryItem(room_id="room-1", content="hello"))
        [taken] = await held.dequeue("held", timeout=1.0)
        proxy.silence()

        assert await other.dequeue("other", timeout=9.0) == []
        await held.ack(taken.id)
        assert (await direct.xpending(held._pending_key, held._group))["pending"] == 0
    finally:
        await held.close()
        await other.close()
        await direct.delete(held._pending_key, held._dl_key, held._delayed_key)
        await direct.aclose()


def _status_listener(url: str, prefix: str) -> tuple[Any, Any, Callable[[str], Awaitable[None]]]:
    """A status bus listening through *url*, one publishing directly, and how
    to publish a detail."""
    listening = RedisStatusBackend(url, key_prefix=prefix)
    publishing = RedisStatusBackend(os.environ["REDIS_URL"], key_prefix=prefix)

    async def publish(detail: str) -> None:
        entry = StatusEntry(
            ts=datetime.now(UTC).isoformat(),
            agent_id="worker",
            action="task",
            status=StatusLevel.INFO,
            detail=detail,
        )
        await publishing.publish(entry)

    return listening, publishing, publish


def _realtime_listener(url: str, prefix: str) -> tuple[Any, Any, Callable[[str], Awaitable[None]]]:
    """A realtime backend listening through *url*, one publishing directly,
    and how to publish a detail."""
    listening = RedisRealtimeBackend(url, channel_prefix=prefix)
    publishing = RedisRealtimeBackend(os.environ["REDIS_URL"], channel_prefix=prefix)

    async def publish(detail: str) -> None:
        event = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_START,
            user_id="user-1",
            data={"detail": detail},
        )
        await publishing.publish("room-1", event)

    return listening, publishing, publish


async def _subscribe(backend: Any, received: list[str]) -> None:
    if isinstance(backend, RedisStatusBackend):
        await backend.subscribe(lambda entry: received.append(entry.detail))
    else:
        await backend.subscribe("room-1", lambda event: received.append(event.data["detail"]))


@pytest.mark.parametrize("listener", [_status_listener, _realtime_listener])
async def test_a_reader_hears_again_once_its_connection_is_silenced(
    short_bounds: None, proxy: Silencing, listener: Any
) -> None:
    """A reader whose connection goes silent pings it, drops it unanswered,
    and subscribes again on a new one: what is published then reaches it."""
    prefix = f"roomkit:test:{uuid4().hex}"
    listening, publishing, publish = listener(await proxy.url(), prefix)
    received: list[str] = []
    try:
        await _subscribe(listening, received)
        await publish("before")
        async with asyncio.timeout(2.0):
            while "before" not in received:
                await asyncio.sleep(0.05)
        proxy.silence()

        async with asyncio.timeout(8.0):
            sent = 0
            while not any(detail.startswith("after") for detail in received):
                await publish(f"after-{sent}")
                sent += 1
                await asyncio.sleep(0.2)
    finally:
        await listening.close()
        await publishing.close()
        await _delete_keys(prefix)


async def _delete_keys(prefix: str) -> None:
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    keys = [key async for key in client.scan_iter(f"{prefix}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


async def _status_call(url: str, prefix: str) -> tuple[Any, Callable[[], Awaitable[object]]]:
    backend = RedisStatusBackend(url, key_prefix=prefix)
    entry = StatusEntry(
        ts=datetime.now(UTC).isoformat(), agent_id="w", action="task", status=StatusLevel.INFO
    )
    return backend, lambda: backend.publish(entry)


async def _realtime_call(url: str, prefix: str) -> tuple[Any, Callable[[], Awaitable[object]]]:
    backend = RedisRealtimeBackend(url, channel_prefix=prefix)
    event = EphemeralEvent(room_id="r", type=EphemeralEventType.TYPING_START, user_id="u")
    return backend, lambda: backend.publish("r", event)


async def _delivery_call(url: str, prefix: str) -> tuple[Any, Callable[[], Awaitable[object]]]:
    backend = RedisDeliveryBackend(url, stream_prefix=prefix)
    return backend, lambda: backend.enqueue(DeliveryItem(room_id="r", content="hello"))


@pytest.mark.parametrize("call", [_status_call, _realtime_call, _delivery_call])
async def test_a_call_on_a_silenced_connection_gives_up_within_its_bound(
    short_bounds: None, proxy: Silencing, call: Any
) -> None:
    """A call whose pooled connection went silent gives it up within the
    client's bound (0.3 s here, 0.5 s with the delivery's read), however
    redis-py then reports it: never the 5 s of redis-py 8's default, nor
    TCP's 15 minutes before 8.0."""
    prefix = f"roomkit:test:{uuid4().hex}"
    backend, once = await call(await proxy.url(), prefix)
    try:
        await once()  # the pool now holds the connection about to be silenced
        proxy.silence()
        started = asyncio.get_running_loop().time()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(4.0):
                await once()
        assert asyncio.get_running_loop().time() - started < 2.0
    finally:
        await backend.close()
        await _delete_keys(prefix)
