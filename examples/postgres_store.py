"""PostgreSQL storage backend.

Demonstrates how to set up PostgresStore for production persistence
instead of the default InMemoryStore. Shows:
- Configuring PostgresStore with a connection URL
- Room, event, and participant CRUD operations
- Querying conversation history with pagination
- Falling back to InMemoryStore when DATABASE_URL is not set

Requires the ``postgres`` extra (asyncpg)::

    pip install roomkit[postgres]

Run with:
    DATABASE_URL=postgresql://user:pass@localhost/roomkit uv run python examples/postgres_store.py

Environment variables:
    DATABASE_URL  PostgreSQL DSN (optional). The tables are created on first
                  use; running the example again reuses the room and appends
                  to its history, which shows the data survived the restart.

Without DATABASE_URL, this example falls back to InMemoryStore as a demo.

With PostgreSQL the per-room lock is a PostgresAdvisoryLockManager on its own
connection pool, so several processes sharing the database serialise the same
room (RFC §13.5); the default InMemoryLockManager only covers one process.
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit

from roomkit import (
    InboundMessage,
    RoomEvent,
    RoomKit,
    RoomNotFoundError,
    TextContent,
    WebSocketChannel,
)
from roomkit.store import (
    ConversationStore,
    InMemoryStore,
    PostgresAdvisoryLockManager,
    PostgresStore,
)

ROOM_ID = "persistent-room"


def _redact_dsn(dsn: str) -> str:
    """Return *dsn* with its password replaced, safe to print."""
    parts = urlsplit(dsn)
    if parts.password is None:
        return dsn
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
    return urlunsplit(parts._replace(netloc=netloc))


async def main() -> None:
    database_url = os.environ.get("DATABASE_URL", "")

    store: ConversationStore
    lock_manager: PostgresAdvisoryLockManager | None = None
    if database_url:
        # Production: use PostgreSQL
        pg_store = PostgresStore(database_url)
        await pg_store.init()  # Creates the pool and the tables if needed
        store = pg_store
        # Cross-process room locks, on a pool separate from the store's
        lock_manager = PostgresAdvisoryLockManager(database_url)
        await lock_manager.init()
        print(f"Using PostgresStore + advisory locks ({_redact_dsn(database_url)})")
    else:
        # Fallback: in-memory (for demo purposes)
        store = InMemoryStore()
        print("Using InMemoryStore (set DATABASE_URL for PostgreSQL)")

    # --- RoomKit with custom store (kit.close() closes both) ---
    kit = RoomKit(store=store, lock_manager=lock_manager)

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv, room_id=ROOM_ID)

    # --- Create the room, or reuse it when an earlier run persisted it ---
    try:
        room = await kit.get_room(ROOM_ID)
        print(f"\nRoom found from an earlier run: {room.id} (event_count={room.event_count})")
    except RoomNotFoundError:
        room = await kit.create_room(room_id=ROOM_ID, metadata={"topic": "Support"})
        print(f"\nRoom created: {room.id} (status={room.status})")

    # Attaching again over an existing binding is an update, not an error.
    await kit.attach_channel(ROOM_ID, "ws-user")

    # Send several messages
    messages = [
        "Hello, I need help with my account",
        "I can't log in since yesterday",
        "I've tried resetting my password",
        "The reset email never arrives",
        "Can someone help me?",
    ]

    for text in messages:
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user",
                content=TextContent(body=text),
            ),
            room_id=ROOM_ID,
        )

    print(f"Sent {len(messages)} messages")

    # --- Query with pagination ---
    print("\n--- Paginated History ---")

    # Page 1: first 3 events
    page1 = await kit.store.list_events(ROOM_ID, offset=0, limit=3)
    print(f"\nPage 1 ({len(page1)} events):")
    for ev in page1:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] {ev.content.body}")

    # Page 2: next 3 events
    page2 = await kit.store.list_events(ROOM_ID, offset=3, limit=3)
    print(f"\nPage 2 ({len(page2)} events):")
    for ev in page2:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] {ev.content.body}")

    # --- Get timeline via convenience method ---
    print("\n--- Full Timeline ---")
    timeline = await kit.get_timeline(ROOM_ID, offset=0, limit=50)
    msg_events = [e for e in timeline if e.type.value == "message"]
    print(f"Total messages: {len(msg_events)}")

    # --- Room metadata ---
    room = await kit.get_room(ROOM_ID)
    print(f"\nRoom metadata: {room.metadata}")
    print(f"Room event_count: {room.event_count}")

    # --- Participants ---
    participants = await kit.store.list_participants(ROOM_ID)
    print(f"\nParticipants ({len(participants)}):")
    for p in participants:
        print(f"  {p.id}: role={p.role}, status={p.status}")

    # --- Bindings ---
    bindings = await kit.store.list_bindings(ROOM_ID)
    print(f"\nBindings ({len(bindings)}):")
    for b in bindings:
        print(f"  {b.channel_id}: type={b.channel_type}, muted={b.muted}")

    # --- Cleanup ---
    if database_url:
        print("\nData is persisted in PostgreSQL and will survive restarts.")
    else:
        print("\nData is in-memory only (will be lost when process exits).")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
