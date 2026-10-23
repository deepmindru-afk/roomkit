"""How a pub/sub reader tells a dead connection from a quiet one
(roomkit._redis.PubSubWatch; RMK-868): a quiet connection that answers its
ping is kept, one that does not is dropped once, and a ping's answer never
reaches the reader."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

import roomkit._redis as redis_bounds
from roomkit._redis import PubSubWatch


class _Connection:
    def __init__(self) -> None:
        self.dropped = 0

    async def disconnect(self) -> None:
        self.dropped += 1


class _PubSub:
    """A pub/sub whose connection answers its pings, or stays silent."""

    def __init__(self, *, answers: bool, subscribed: bool = True) -> None:
        self.connection = _Connection()
        self.subscribed = subscribed
        self.pings = 0
        self._answers = answers
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def ping(self) -> None:
        self.pings += 1
        if self._answers:
            self.arrive({"type": "pong", "pattern": None, "channel": None, "data": b""})

    def arrive(self, message: dict[str, Any]) -> None:
        self._queue.put_nowait(message)

    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float = 0.0
    ) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None


@pytest.fixture(autouse=True)
def short_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redis_bounds, "PING_AFTER_S", 0.05)
    monkeypatch.setattr(redis_bounds, "PONG_WAIT_S", 0.05)


async def _read_for(watch: PubSubWatch, pubsub: _PubSub, seconds: float) -> list[Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    read = []
    while loop.time() < deadline:
        read.append(await watch.read(pubsub, timeout=0.02))
    return read


async def test_a_quiet_connection_that_answers_its_ping_is_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pubsub = _PubSub(answers=True)

    with caplog.at_level(logging.WARNING, logger="roomkit.redis"):
        read = await _read_for(PubSubWatch("test"), pubsub, 0.2)  # four silences

    assert pubsub.pings >= 2
    assert pubsub.connection.dropped == 0
    assert caplog.records == []
    assert all(message is None for message in read)  # a pong never reaches the reader


async def test_a_connection_that_never_answers_is_dropped_once_per_silence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pubsub = _PubSub(answers=False)

    with caplog.at_level(logging.WARNING, logger="roomkit.redis"):
        await _read_for(PubSubWatch("test"), pubsub, 0.15)

    assert pubsub.connection.dropped >= 1
    # One ping per silence, each unanswered one dropping the connection.
    assert pubsub.pings - pubsub.connection.dropped in (0, 1)
    assert "connection silent" in caplog.text


async def test_nothing_is_pinged_while_nothing_is_subscribed() -> None:
    pubsub = _PubSub(answers=False, subscribed=False)

    await _read_for(PubSubWatch("test"), pubsub, 0.12)  # past two silences

    assert (pubsub.pings, pubsub.connection.dropped) == (0, 0)


async def test_a_message_reaches_the_reader_and_counts_as_heard() -> None:
    pubsub = _PubSub(answers=False)
    watch = PubSubWatch("test")
    message = {"type": "message", "pattern": None, "channel": b"c", "data": b"x"}

    for _ in range(8):
        await asyncio.sleep(0.02)
        pubsub.arrive(message)
        assert await watch.read(pubsub, timeout=0.02) == message

    assert (pubsub.pings, pubsub.connection.dropped) == (0, 0)
