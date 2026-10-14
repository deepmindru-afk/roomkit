"""Room contracts shared by all conversation stores: ``update_room``, the
order of room pages, and the read markers (a channel's read high-water mark
is an event ``index``)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.enums import RoomStatus
from roomkit.models.room import Room
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from tests.conftest import make_event


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


async def test_a_stale_copy_does_not_rewind_the_event_counters(
    contract_store: ConversationStore,
) -> None:
    """event_count and latest_index move when an event is committed; a copy
    read before two commits must not take them back to where they were."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        stale = await contract_store.get_room(room.id)
        assert stale is not None
        for i in range(2):
            await contract_store.commit_event(room.id, make_event(room_id=room.id, body=str(i)))

        updated = await contract_store.update_room(
            stale.model_copy(update={"status": RoomStatus.PAUSED})
        )
        stored = await contract_store.get_room(room.id)

        assert updated.status == RoomStatus.PAUSED
        assert stored is not None and stored.status == RoomStatus.PAUSED
        assert (updated.event_count, updated.latest_index) == (2, 1)
        assert (stored.event_count, stored.latest_index) == (2, 1)
    finally:
        await contract_store.delete_room(room.id)


async def _rooms_created_out_of_order(store: ConversationStore) -> tuple[str, list[str]]:
    """Five rooms of one organization, stored in an order their creation
    times contradict, two of them created at the same instant."""
    org = uuid4().hex
    created = datetime(2026, 1, 1, tzinfo=UTC)
    rooms = (("c", 2, "x"), ("a", 0, "x"), ("e", 3, "y"), ("d", 1, "x"), ("b", 1, "x"))
    for name, minutes, team in rooms:
        await store.create_room(
            Room(
                id=f"{org}-{name}",
                organization_id=org,
                created_at=created + timedelta(minutes=minutes),
                metadata={"team": team},
            )
        )
    return org, [f"{org}-{name}" for name in "abcde"]


async def test_a_page_of_rooms_is_the_same_in_every_store(
    contract_store: ConversationStore,
) -> None:
    """Rooms come oldest first, by created_at then id, whatever order they
    were stored in: a page boundary falls on the same room in every store."""
    org, ids = await _rooms_created_out_of_order(contract_store)
    a, b, c, d, e = ids
    try:
        page = await contract_store.find_rooms(organization_id=org, limit=2, offset=1)
        teams = await contract_store.find_rooms(
            organization_id=org, metadata_filter={"team": "x"}, limit=2, offset=2
        )
        listed = [
            room.id for room in await contract_store.list_rooms(limit=10_000) if room.id in ids
        ]

        assert [room.id for room in page] == [b, d]
        assert [room.id for room in teams] == [d, c]
        assert listed == [a, b, d, c, e]
    finally:
        for room_id in ids:
            await contract_store.delete_room(room_id)


async def test_rooms_created_with_and_without_a_time_zone_still_page() -> None:
    """A naive created_at beside an aware one is a room like any other."""
    store = InMemoryStore()
    await store.create_room(Room(id="aware", created_at=datetime(2026, 1, 1, 12, tzinfo=UTC)))
    await store.create_room(Room(id="naive", created_at=datetime(2026, 1, 1, 12)))

    assert {room.id for room in await store.list_rooms()} == {"aware", "naive"}


async def _room_with_events(store: ConversationStore, *indexes: int) -> tuple[str, list[str]]:
    room = await store.create_room(Room(id=uuid4().hex))
    ids = []
    for i in indexes:
        event = await store.add_event(make_event(room_id=room.id, index=i, body=str(i)))
        ids.append(event.id)
    return room.id, ids


async def test_marking_an_empty_room_read_leaves_its_first_event_unread(
    contract_store: ConversationStore,
) -> None:
    room_id, _ = await _room_with_events(contract_store)
    try:
        await contract_store.mark_all_read(room_id, "sms")
        assert await contract_store.list_read_markers(room_id) == {}
        await contract_store.add_event(make_event(room_id=room_id, index=0))
        assert await contract_store.get_unread_count(room_id, "sms") == 1
    finally:
        await contract_store.delete_room(room_id)


async def test_the_marker_is_the_read_events_index_whatever_was_deleted_or_written_first(
    contract_store: ConversationStore,
) -> None:
    # Written 1, 0, 2, 3: write order is not index order (a backfill, a
    # concurrent commit), and deleting index 0 must not shift the marker.
    room_id, ids = await _room_with_events(contract_store, 1, 0, 2, 3)
    try:
        await contract_store.mark_read(room_id, "sms", ids[2])  # index 2
        assert await contract_store.list_read_markers(room_id) == {"sms": 2}
        assert await contract_store.get_unread_count(room_id, "sms") == 1

        await contract_store.delete_event(room_id, ids[1])  # index 0
        assert await contract_store.list_read_markers(room_id) == {"sms": 2}
        assert await contract_store.get_unread_count(room_id, "sms") == 1

        await contract_store.mark_all_read(room_id, "sms")
        assert await contract_store.list_read_markers(room_id) == {"sms": 3}
        assert await contract_store.get_unread_count(room_id, "sms") == 0
    finally:
        await contract_store.delete_room(room_id)


async def test_the_marker_never_moves_back(contract_store: ConversationStore) -> None:
    room_id, ids = await _room_with_events(contract_store, 0, 1, 2)
    try:
        await contract_store.mark_read(room_id, "sms", ids[2])
        await contract_store.mark_read(room_id, "sms", ids[0])
        assert await contract_store.list_read_markers(room_id) == {"sms": 2}
        assert await contract_store.get_unread_count(room_id, "sms") == 0
    finally:
        await contract_store.delete_room(room_id)


async def test_an_event_of_another_room_marks_nothing(contract_store: ConversationStore) -> None:
    room_id, _ = await _room_with_events(contract_store, 0, 1)
    other_id, other_ids = await _room_with_events(contract_store, 0, 1, 2, 3)
    try:
        await contract_store.mark_read(room_id, "sms", other_ids[3])
        await contract_store.mark_read(room_id, "sms", "no-such-event")
        assert await contract_store.list_read_markers(room_id) == {}
        assert await contract_store.get_unread_count(room_id, "sms") == 2
    finally:
        await contract_store.delete_room(room_id)
        await contract_store.delete_room(other_id)
