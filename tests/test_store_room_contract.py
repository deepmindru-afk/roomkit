"""``update_room`` contract shared by all conversation stores."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.enums import RoomStatus
from roomkit.models.room import Room
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def contract_store(request: pytest.FixtureRequest) -> AsyncIterator[ConversationStore]:
    store: ConversationStore
    if request.param == "memory":
        store = InMemoryStore()
    elif request.param == "sqlite":
        store = SQLiteStore(":memory:")
    else:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        from roomkit.store.postgres import PostgresStore

        store = PostgresStore(dsn=dsn)
        await store.init(min_size=1, max_size=2)
    try:
        yield store
    finally:
        await store.close()


async def test_update_room_refuses_a_room_the_store_lacks(
    contract_store: ConversationStore,
) -> None:
    with pytest.raises(RoomNotFoundError):
        await contract_store.update_room(Room(id=uuid4().hex))


async def test_a_stale_copy_neither_rewinds_nor_reports_an_old_delivered_index(
    contract_store: ConversationStore,
) -> None:
    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        stale = await contract_store.get_room(room.id)
        assert stale is not None
        assert await contract_store.advance_delivered_index(room.id, 0)
        assert await contract_store.advance_delivered_index(room.id, 1)

        updated = await contract_store.update_room(
            stale.model_copy(update={"status": RoomStatus.PAUSED})
        )

        assert updated.status == RoomStatus.PAUSED
        assert updated.delivered_index == 1
        assert await contract_store.get_delivered_index(room.id) == 1
    finally:
        await contract_store.delete_room(room.id)
