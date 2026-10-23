"""Tests for SQLiteStore.

The behavioural contract is the one ``InMemoryStore`` defines, so the whole
memory suite runs again here against SQLite — the module-level ``store``
fixture shadows the conftest one for every inherited test class. SQLite-only
behaviour (persistence across connections, FTS5 search) gets its own tests.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

import pytest

from roomkit import RoomKit
from roomkit.core.mixins.verification import HELD_KEY
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, EventStatus, EventType, RoomStatus
from roomkit.models.event import DeleteContent
from roomkit.models.room import Room
from roomkit.store.sqlite import _SCHEMA_VERSION, SQLiteSchemaError, SQLiteStore
from tests.conftest import make_event
from tests.test_store_memory import (
    TestBindingOperations,
    TestConnectionTenure,
    TestCursorPagination,
    TestDeleteEvent,
    TestDeleteRoomCleanup,
    TestEventOperations,
    TestEventOwnership,
    TestFindLatestRoomIndex,
    TestFindRooms,
    TestIdentityOperations,
    TestListEventsVisibilityFilter,
    TestObservationOperations,
    TestParticipantOperations,
    TestReadTracking,
    TestRoomOperations,
    TestTaskOperations,
)

__all__ = [
    "TestBindingOperations",
    "TestConnectionTenure",
    "TestCursorPagination",
    "TestDeleteEvent",
    "TestDeleteRoomCleanup",
    "TestEventOperations",
    "TestEventOwnership",
    "TestFindLatestRoomIndex",
    "TestFindRooms",
    "TestIdentityOperations",
    "TestListEventsVisibilityFilter",
    "TestObservationOperations",
    "TestParticipantOperations",
    "TestReadTracking",
    "TestRoomOperations",
    "TestTaskOperations",
]


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    await s.close()


class TestPersistence:
    async def test_data_survives_a_close_and_reopen(self, tmp_path) -> None:
        path = tmp_path / "persist.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="premier message"))
        await store.close()

        reopened = SQLiteStore(path)
        try:
            room = await reopened.get_room("r1")
            assert room is not None
            assert room.event_count == 1
            events = await reopened.get_conversation("r1")
            assert [e.content.body for e in events] == ["premier message"]
        finally:
            await reopened.close()

    async def test_close_is_idempotent(self, tmp_path) -> None:
        store = SQLiteStore(tmp_path / "x.db")
        await store.create_room(Room(id="r1"))
        await store.close()
        await store.close()

    async def test_v1_database_migrates_sequence_and_unique_index(self, tmp_path) -> None:
        """A development-era v1 file upgrades without reusing deleted history."""
        path = tmp_path / "v1.db"
        room = Room(id="r1", event_count=2, latest_index=1)
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE rooms(
                id TEXT PRIMARY KEY,
                created_ts REAL NOT NULL,
                organization_id TEXT,
                status TEXT NOT NULL,
                delivered_index INTEGER NOT NULL DEFAULT -1,
                data TEXT NOT NULL
            );
            CREATE TABLE events(
                id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                type TEXT NOT NULL,
                visibility TEXT NOT NULL,
                source_channel_id TEXT,
                source_channel_type TEXT,
                participant_id TEXT,
                correlation_id TEXT,
                parent_event_id TEXT,
                idempotency_key TEXT,
                created_ts REAL NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX idx_events_room_idx ON events(room_id, idx);
            PRAGMA user_version=1;
            """
        )
        conn.execute(
            "INSERT INTO rooms(id, created_ts, status, data) VALUES(?, ?, ?, ?)",
            (room.id, room.created_at.timestamp(), str(room.status), room.model_dump_json()),
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            committed = await store.commit_event(
                "r1", make_event(room_id="r1", body="after deleted history")
            )
            assert committed.index == 2
        finally:
            await store.close()

        conn = sqlite3.connect(path)
        try:
            # A v1 file is brought all the way to the current schema, not
            # to the version that happened to follow it.
            assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
            indexes = conn.execute("PRAGMA index_list(events)").fetchall()
            room_index = next(row for row in indexes if row[1] == "idx_events_room_idx")
            assert room_index[2] == 1  # unique
        finally:
            conn.close()

    async def test_newer_schema_is_rejected_without_relabelling(self, tmp_path) -> None:
        path = tmp_path / "future.db"
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version=999")
        conn.close()

        store = SQLiteStore(path)
        with pytest.raises(SQLiteSchemaError, match="newer"):
            await store.get_room("r1")
        await store.close()

        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 999
        finally:
            conn.close()

    async def test_a_v5_file_keeps_its_read_markers_as_event_indexes(self, tmp_path) -> None:
        """v5 kept the read event's id; v6 keeps its index. A marker on an
        event that is gone is dropped: it read as all unread, and still does."""
        path = tmp_path / "v5.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        events = [
            await store.add_event(make_event(room_id="r1", index=i, body=str(i))) for i in range(3)
        ]
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            DROP TABLE read_markers;
            CREATE TABLE read_markers(
                room_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(room_id, channel_id)
            );
            PRAGMA user_version=5;
            """
        )
        conn.executemany(
            "INSERT INTO read_markers(room_id, channel_id, event_id) VALUES(?, ?, ?)",
            [("r1", "sms", events[1].id), ("r1", "email", "gone")],
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            assert await store.list_read_markers("r1") == {"sms": 1}
            assert await store.get_unread_count("r1", "sms") == 1
            assert await store.get_unread_count("r1", "email") == 3
        finally:
            await store.close()
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
        finally:
            conn.close()

    async def test_a_v6_file_stamps_its_events_visibility(self, tmp_path) -> None:
        """v7 stores an event's visibility stamped with its source binding's
        (RFC §7.3). An event stored ``all`` under a narrowed binding takes the
        binding's scope once; an event with a scope of its own, or whose
        source is no longer bound, keeps its own."""
        path = tmp_path / "v6.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.add_binding(
            ChannelBinding(
                room_id="r1", channel_id="sms", channel_type=ChannelType.SMS, visibility="ws"
            )
        )
        bodies = {"under the binding": "all", "its own scope": "ai", "gone": "all"}
        for index, (body, scope) in enumerate(bodies.items()):
            source = "gone" if body == "gone" else "sms"
            await store.add_event(
                make_event(
                    room_id="r1", channel_id=source, body=body, index=index, visibility=scope
                )
            )
        held = {HELD_KEY: {"visibility": "all", "identity_id": "i1", "organization_id": None}}
        await store.add_event(
            make_event(
                room_id="r1",
                channel_id="sms",
                body="held",
                index=3,
                visibility="transport",
                metadata=held,
            )
        )
        await store.close()
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version=6")
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            events = await store.list_events("r1")
            assert {e.content.body: e.visibility for e in events} == {
                "under the binding": "ws",
                "its own scope": "ai",
                "gone": "all",
                "held": "transport",
            }
            held_event = next(e for e in events if e.content.body == "held")
            assert held_event.metadata[HELD_KEY]["visibility"] == "ws"
        finally:
            await store.close()

    async def test_v7_database_scopes_its_keys_to_their_source_channel(self, tmp_path) -> None:
        """A v7 file kept one key per room: the keys it recorded stay found
        under the channel that sent them, and a second channel may reuse one."""
        path = tmp_path / "v7.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.add_event(
            make_event(room_id="r1", channel_id="sms", body="first", index=0, idempotency_key="k")
        )
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            DROP INDEX idx_events_idempotency_channel;
            CREATE UNIQUE INDEX idx_events_idempotency
                ON events(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
            DROP TABLE idempotency;
            CREATE TABLE idempotency(
                room_id TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY(room_id, key)
            );
            INSERT INTO idempotency(room_id, key) VALUES('r1', 'k');
            PRAGMA user_version=7;
            """
        )
        conn.close()

        store = SQLiteStore(path)
        try:
            assert await store.check_idempotency("r1", "k", channel_id="sms")
            assert not await store.check_idempotency("r1", "k", channel_id="email")
            await store.add_event(
                make_event(
                    room_id="r1", channel_id="email", body="second", index=1, idempotency_key="k"
                )
            )
            found = await store.get_event_by_idempotency_key("r1", "k", channel_id="email")
            assert found is not None and found.content.body == "second"
        finally:
            await store.close()

    async def test_v8_database_records_when_its_paused_rooms_paused(self, tmp_path) -> None:
        """A room a v8 file stored paused takes its last modification as the
        instant it paused (RFC §5.1), which its close timer counts from."""
        path = tmp_path / "v8.db"
        store = SQLiteStore(path)
        paused = await store.create_room(Room(id="r1", status=RoomStatus.PAUSED))
        await store.create_room(Room(id="r2"))
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            UPDATE rooms SET data = json_remove(data, '$.paused_at');
            PRAGMA user_version=8;
            """
        )
        conn.close()

        store = SQLiteStore(path)
        try:
            upgraded = await store.get_room("r1")
            active = await store.get_room("r2")
            assert upgraded is not None and active is not None
            assert upgraded.paused_at == paused.updated_at
            assert active.paused_at is None
        finally:
            await store.close()

    async def test_two_connections_allocate_one_monotonic_sequence(self, tmp_path) -> None:
        path = tmp_path / "shared.db"
        first = SQLiteStore(path)
        second = SQLiteStore(path)
        await first.create_room(Room(id="r1"))
        try:
            committed = await asyncio.gather(
                *(
                    (first if i % 2 == 0 else second).commit_event(
                        "r1", make_event(room_id="r1", body=f"message {i}")
                    )
                    for i in range(20)
                )
            )
            assert sorted(event.index for event in committed) == list(range(20))
        finally:
            await first.close()
            await second.close()

    async def test_duplicate_index_is_rejected_by_database(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.add_event(make_event(room_id="r1", body="first", index=0))

        with pytest.raises(sqlite3.IntegrityError):
            await store.add_event(make_event(room_id="r1", body="duplicate", index=0))

    async def test_duplicate_idempotency_key_is_rejected_across_connections(
        self, tmp_path
    ) -> None:
        path = tmp_path / "idempotency.db"
        first = SQLiteStore(path)
        second = SQLiteStore(path)
        await first.create_room(Room(id="r1"))
        one = make_event(room_id="r1", body="one", idempotency_key="same")
        two = make_event(room_id="r1", body="two", idempotency_key="same")
        try:
            results = await asyncio.gather(
                first.commit_event("r1", one),
                second.commit_event("r1", two),
                return_exceptions=True,
            )
            assert sum(isinstance(result, sqlite3.IntegrityError) for result in results) == 1
            assert len(await first.list_events("r1")) == 1
        finally:
            await first.close()
            await second.close()

    async def test_framework_warns_that_shared_sqlite_needs_distributed_lock(
        self, tmp_path, caplog
    ) -> None:
        store = SQLiteStore(tmp_path / "warning.db")
        with caplog.at_level(logging.WARNING, logger="roomkit"):
            RoomKit(store=store)
        assert "safe only in a single process" in caplog.text
        await store.close()


class TestFullTextSearch:
    async def test_search_finds_by_content_words(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="on a parlé du projet AEC"))
        await store.commit_event(
            "r1", make_event(room_id="r1", body="recette de tarte aux pommes")
        )

        hits = await store.search_events("projet AEC")
        assert [e.content.body for e in hits] == ["on a parlé du projet AEC"]

    async def test_search_scopes_to_a_room(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.create_room(Room(id="r2"))
        await store.commit_event("r1", make_event(room_id="r1", body="meeting notes alpha"))
        await store.commit_event("r2", make_event(room_id="r2", body="meeting notes beta"))

        hits = await store.search_events("meeting", room_id="r2")
        assert [e.room_id for e in hits] == ["r2"]

    async def test_search_does_not_find_the_rows_the_room_refused(
        self, store: SQLiteStore
    ) -> None:
        """RFC §14.1 reaches the index too: a body a hook refused is stored
        BLOCKED, out of the timeline by default, and not findable by search."""
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="the secret plan"))
        await store.commit_event(
            "r1",
            make_event(
                room_id="r1", body="the secret leak", status=EventStatus.BLOCKED, blocked_by="dlp"
            ),
        )

        hits = await store.search_events("secret")
        assert [e.content.body for e in hits] == ["the secret plan"]

    async def test_search_survives_hostile_query_syntax(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="hello world"))

        assert await store.search_events('"hello* (world -') != []
        assert await store.search_events("???") == []
        assert await store.search_events("") == []

    async def test_deleted_events_leave_the_index(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="ephemeral thing"))
        assert await store.search_events("ephemeral") != []

        await store.delete_event("r1", event.id)
        assert await store.search_events("ephemeral") == []

    async def test_a_soft_deleted_event_leaves_the_index(self, store: SQLiteStore) -> None:
        """A deleted message's text is withheld (RFC §10.3): the search
        finds it no more, while the stored record keeps it."""
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="ephemeral thing"))
        delete = make_event(room_id="r1", status=EventStatus.DELIVERED).model_copy(
            update={"type": EventType.DELETE, "content": DeleteContent(target_event_id=event.id)}
        )
        deleted = event.model_copy(update={"metadata": {"deleted": True}})

        await store.commit_event("r1", delete, target=deleted)

        assert await store.search_events("ephemeral") == []
        stored = await store.get_event(event.id)
        assert stored is not None and stored.content == event.content

    async def test_updated_events_reindex(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="old wording"))
        updated = event.model_copy(deep=True)
        updated.content.body = "new phrasing"
        await store.update_event(updated)

        assert await store.search_events("wording") == []
        assert [e.id for e in await store.search_events("phrasing")] == [event.id]
