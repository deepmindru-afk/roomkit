"""InMemoryStore object-ownership contract (RFC §14.4), and store close().

The store copies ON WRITE and shares the stored snapshot on read, for events,
rooms, bindings and participants alike (RMK-984): a read object is treated as
frozen (its isolation is not promised).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore
from tests.conftest import make_event


async def _first(page: Awaitable[list[Any]]) -> Any:
    return (await page)[0]


_READS: dict[str, Callable[[InMemoryStore], Awaitable[Any]]] = {
    "get_room": lambda store: store.get_room("r1"),
    "list_rooms": lambda store: _first(store.list_rooms()),
    "find_rooms": lambda store: _first(store.find_rooms()),
    "find_latest_room": lambda store: store.find_latest_room("p1"),
    "get_binding": lambda store: store.get_binding("r1", "sms"),
    "list_bindings": lambda store: _first(store.list_bindings("r1")),
    "get_participant": lambda store: store.get_participant("r1", "p1"),
    "list_participants": lambda store: _first(store.list_participants("r1")),
    "get_event": lambda store: store.get_event("e1"),
    "list_events": lambda store: _first(store.list_events("r1")),
}


@pytest.mark.parametrize("read", list(_READS))
async def test_reads_share_the_stored_snapshot(read: str) -> None:
    """No copy per read: every reader holds the store's snapshot, frozen to
    it, and a change goes through a write (RFC §14.4, RMK-984). That a write
    copies its record in is the stores' contract
    (``test_store_room_contract.py::test_a_written_record_is_copied_in``)."""
    store = InMemoryStore()
    await store.create_room(Room(id="r1"))
    await store.add_binding(
        ChannelBinding(room_id="r1", channel_id="sms", channel_type=ChannelType.SMS)
    )
    await store.add_participant(Participant(id="p1", room_id="r1", channel_id="sms"))
    await store.add_event(make_event(room_id="r1", id="e1"))

    assert await _READS[read](store) is await _READS[read](store)


async def test_store_close_is_idempotent_noop() -> None:
    store = InMemoryStore()
    await store.close()
    await store.close()  # idempotent — must not raise


async def test_roomkit_close_closes_store() -> None:
    from roomkit import RoomKit

    closed: list[bool] = []

    class SpyStore(InMemoryStore):
        async def close(self) -> None:
            closed.append(True)
            await super().close()

    kit = RoomKit(store=SpyStore())
    await kit.close()
    assert closed == [True]
