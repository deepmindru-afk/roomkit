"""A record no member receives is committed outside the pipeline (RFC §10.5).

``commit_event`` reads the room scoped to its tenant, refuses a room whose
status refuses new events, and commits the record with the room's next index,
counted delivered at once: the room's next event never waits on it. The
record is stored as given, the source write rule (§7.5 rule 2) aside, and a
hook already holding the room's lock calls it without deadlocking.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.core.exceptions import RoomClosedError, RoomNotFoundError
from roomkit.models.enums import EventStatus
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from tests.conftest import make_event
from tests.test_framework import SimpleChannel


async def _room(kit: RoomKit, *, organization_id: str = "tenant-a") -> None:
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1", organization_id=organization_id)
    await kit.attach_channel("r1", "sms1")


def _record(body: str = "trace") -> RoomEvent:
    return make_event(room_id="r1", channel_id="tracer", body=body)


async def test_a_record_takes_the_next_index_and_reaches_no_one() -> None:
    kit = RoomKit()
    await _room(kit)
    sms = kit.channels["sms1"]
    assert isinstance(sms, SimpleChannel)
    before = await kit.store.list_events("r1")

    committed = await kit.commit_event("r1", _record(), organization_id="tenant-a")

    assert committed.index == before[-1].index + 1
    assert [e.id for e in await kit.store.list_events("r1")][-1] == committed.id
    assert sms.delivered == []
    await kit.close()


async def test_the_next_event_does_not_wait_on_the_records_index() -> None:
    """Counted delivered at once: a send_event right after never waits the
    delivery gap timeout on a hole."""
    kit = RoomKit(delivery_gap_timeout=5.0)
    await _room(kit)
    committed = await kit.commit_event("r1", _record(), organization_id="tenant-a")

    sent = await asyncio.wait_for(
        kit.send_event("r1", "sms1", TextContent(body="hello")), timeout=2.0
    )

    assert sent.index == committed.index + 1
    await kit.close()


async def test_another_tenants_room_is_not_found_and_nothing_is_written() -> None:
    kit = RoomKit()
    await _room(kit)
    before = await kit.store.list_events("r1")

    with pytest.raises(RoomNotFoundError):
        await kit.commit_event("r1", _record(), organization_id="tenant-b")

    assert await kit.store.list_events("r1") == before
    await kit.close()


@pytest.mark.parametrize("operation", ["close_room", "archive_room"])
async def test_a_room_that_refuses_events_refuses_the_record(operation: str) -> None:
    kit = RoomKit()
    await _room(kit)
    await getattr(kit, operation)("r1")
    before = await kit.store.list_events("r1")

    with pytest.raises(RoomClosedError):
        await kit.commit_event("r1", _record())

    assert await kit.store.list_events("r1") == before
    await kit.close()


async def test_a_muted_sources_record_is_stored_as_given() -> None:
    """A record is not an injected event: the source write rule does not
    turn it BLOCKED."""
    kit = RoomKit()
    await _room(kit)
    await kit.mute("r1", "sms1")
    record = make_event(room_id="r1", channel_id="sms1", body="snapshot")

    committed = await kit.commit_event("r1", record)

    assert committed.status == record.status != EventStatus.BLOCKED
    await kit.close()


async def test_a_hook_holding_the_rooms_lock_commits_a_record() -> None:
    kit = RoomKit()
    await _room(kit)
    traced: list[RoomEvent] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC, name="trace")
    async def trace(event: RoomEvent, ctx: object) -> HookResult:
        if event.source.channel_id == "sms1":
            traced.append(await kit.commit_event("r1", _record("traced")))
        return HookResult.allow()

    await asyncio.wait_for(kit.send_event("r1", "sms1", TextContent(body="hello")), timeout=2.0)

    assert [e.content.body for e in traced] == ["traced"]
    await kit.close()


async def test_an_unscoped_call_reads_the_room_as_get_room_does() -> None:
    kit = RoomKit()
    await _room(kit)

    committed = await kit.commit_event("r1", _record())

    assert isinstance(await kit.get_room("r1"), Room)
    assert committed.room_id == "r1"
    await kit.close()


async def test_a_record_of_another_room_is_refused_before_anything_is_read() -> None:
    """The stores file a row under its event's room: the room checked must be
    the room written (RFC §10.5)."""
    kit = RoomKit()
    await _room(kit)
    before = await kit.store.list_events("r1")

    with pytest.raises(ValueError, match="of room elsewhere"):
        await kit.commit_event("r1", make_event(room_id="elsewhere", channel_id="tracer"))

    assert await kit.store.list_events("r1") == before
    await kit.close()
