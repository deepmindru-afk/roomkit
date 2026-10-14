"""Every delivery backend lists its dead letters in the same order.

Redis listed the most recent first and the in-memory backend the oldest
first, so ``limit=50`` returned different items from each.
Set REDIS_URL to run the Redis case against Redis 6.2+.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from roomkit.delivery.base import DeliveryBackend, DeliveryItem
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.redis import RedisDeliveryBackend


@pytest.fixture(params=["memory", "redis"])
async def backend(request: pytest.FixtureRequest) -> AsyncIterator[DeliveryBackend]:
    if request.param == "memory":
        yield InMemoryDeliveryBackend()
        return
    if not os.environ.get("REDIS_URL"):
        pytest.skip("REDIS_URL not set")
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    backend = RedisDeliveryBackend(client=client, stream_prefix=f"roomkit:test:{uuid4().hex}")
    await client.xgroup_create(backend._pending_key, backend._group, id="0", mkstream=True)
    try:
        yield backend
    finally:
        await backend.close()
        await client.delete(backend._pending_key, backend._dl_key, backend._delayed_key)
        await client.aclose()


async def test_the_most_recent_dead_letters_come_first(backend: DeliveryBackend) -> None:
    for i in range(5):
        await backend.enqueue(DeliveryItem(room_id="r1", content=f"msg-{i}"))
        [item] = await backend.dequeue("w1", timeout=1.0)
        await backend.dead_letter(item.id, f"error-{i}")

    newest = await backend.get_dead_letter_items(limit=3)

    assert [item.content for item in newest] == ["msg-4", "msg-3", "msg-2"]
