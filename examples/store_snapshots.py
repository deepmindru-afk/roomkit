"""Change a stored room the way every store keeps: a copy, then a write.

A room, binding, participant or event a store returns is its snapshot (RFC
§14.4). ``InMemoryStore`` shares it with every reader instead of copying it
per read, and SQL stores build a new one per read; the same update code reads
and writes alike on both. The example pauses a room and tags it on each store,
then checks that what the caller kept of the object it wrote never reaches
the store.

Run with: uv run python examples/store_snapshots.py
No credentials or external services required.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from shared import setup_logging

from roomkit import ConversationStore, InMemoryStore, RoomKit, RoomStatus, SQLiteStore

logger = setup_logging("store_snapshots")


async def pause_and_tag(kit: RoomKit, room_id: str) -> None:
    """Pause *room_id* and tag it, through a copy of the stored room."""
    room = await kit.store.get_room(room_id)
    assert room is not None
    changed = room.model_copy(
        update={"status": RoomStatus.PAUSED, "metadata": {**room.metadata, "tag": "vip"}}
    )
    await kit.store.update_room(changed)
    # What the caller still holds is its own: changing it touches nothing.
    changed.metadata["tag"] = "scratch"


async def run(store: ConversationStore) -> tuple[RoomStatus, dict[str, object]]:
    async with RoomKit(store=store) as kit:
        await kit.create_room(room_id="support", metadata={"topic": "billing"})
        await pause_and_tag(kit, "support")
        room = await kit.store.get_room("support")
        assert room is not None
        logger.info("%s: %s %s", type(store).__name__, room.status, room.metadata)
        return room.status, room.metadata


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        stores = [InMemoryStore(), SQLiteStore(Path(directory) / "rooms.db")]
        results = [await run(store) for store in stores]
    expected = (RoomStatus.PAUSED, {"topic": "billing", "tag": "vip"})
    assert results == [expected, expected], results
    logger.info("Both stores read the same room")


if __name__ == "__main__":
    asyncio.run(main())
