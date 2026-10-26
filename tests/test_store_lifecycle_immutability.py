"""InMemoryStore object-ownership contract (RFC §14.4), and store close().

The store copies ON WRITE and shares the stored snapshot on read, for events,
rooms, bindings and participants alike (RMK-984): a caller's mutation of the
object it *wrote* must not reach the store, and a read object is treated as
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


def _write(store: InMemoryStore, kind: str, op: str) -> Callable[[Any], Awaitable[Any]]:
    return {
        ("room", "add"): store.create_room,
        ("room", "update"): store.update_room,
        ("binding", "add"): store.add_binding,
        ("binding", "update"): store.update_binding,
        ("participant", "add"): store.add_participant,
        ("participant", "update"): store.update_participant,
        ("event", "add"): store.add_event,
        ("event", "update"): store.update_event,
    }[(kind, op)]


async def _read(store: InMemoryStore, kind: str) -> Any:
    if kind == "room":
        return await store.get_room("r1")
    if kind == "binding":
        return await store.get_binding("r1", "sms")
    if kind == "participant":
        return await store.get_participant("r1", "p1")
    return await store.get_event("e1")


def _record(kind: str) -> Any:
    nested = {"nested": {"a": 1}}
    if kind == "room":
        return Room(id="r1", metadata=nested)
    if kind == "binding":
        return ChannelBinding(
            room_id="r1", channel_id="sms", channel_type=ChannelType.SMS, metadata=nested
        )
    if kind == "participant":
        return Participant(id="p1", room_id="r1", channel_id="sms", metadata=nested)
    return make_event(room_id="r1", id="e1", metadata=nested)


async def _stored(store: InMemoryStore, kind: str, op: str) -> Any:
    """*kind*'s record written by *op*: for an update, over one added first."""
    if kind != "room":
        await store.create_room(Room(id="r1"))
    record = _record(kind)
    if op == "update":
        await _write(store, kind, "add")(record.model_copy(deep=True))
    await _write(store, kind, op)(record)
    return record


KINDS = pytest.mark.parametrize("kind", ["room", "binding", "participant", "event"])


@KINDS
@pytest.mark.parametrize("op", ["add", "update"])
async def test_a_written_record_is_copied_in(kind: str, op: str) -> None:
    """The store owns its stored representation from the moment the write
    returns (RFC §14.4): the writer's retained reference is not the store's."""
    store = InMemoryStore()
    record = await _stored(store, kind, op)

    record.metadata["nested"]["a"] = 999  # the caller mutates ITS object

    assert (await _read(store, kind)).metadata["nested"]["a"] == 1


@KINDS
async def test_reads_share_the_stored_snapshot(kind: str) -> None:
    """No copy per read: every reader holds the store's snapshot, frozen to
    it, and a change goes through a write (RFC §14.4)."""
    store = InMemoryStore()
    await _stored(store, kind, "add")

    assert await _read(store, kind) is await _read(store, kind)


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
