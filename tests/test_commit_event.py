"""A record no member receives is committed outside the pipeline (RFC §10.5).

``commit_event`` reads the room scoped to its tenant, refuses a room whose
status refuses new events, and commits the record with the room's next index,
which the delivery lane passes at its turn with nothing to run: the room's
next event never waits on it longer than on the events before it. The record
is no activity, is stored as given, the source write rule (§7.5 rule 2)
aside, and a hook already holding the room's lock calls it without
deadlocking.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.core.exceptions import IdempotencyKeyTakenError, RoomClosedError, RoomNotFoundError
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
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
    """Passed at once with nothing before it in flight: a send_event right
    after never waits the delivery gap timeout on a hole."""
    kit = RoomKit(delivery_gap_timeout=5.0)
    await _room(kit)
    committed = await kit.commit_event("r1", _record(), organization_id="tenant-a")

    sent = await asyncio.wait_for(
        kit.send_event("r1", "sms1", TextContent(body="hello")), timeout=2.0
    )

    assert sent.index == committed.index + 1
    await kit.close()


class _Held(SimpleChannel):
    """A transport whose delivery waits until it is released."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.entered.set()
        await self.release.wait()
        return await super().deliver(event, binding, context)


async def test_a_record_under_a_key_its_channel_holds_is_refused() -> None:
    """RFC §10.5, §14.3: a record is never a replay. One under a key its
    source channel holds in the room raises ``IdempotencyKeyTakenError``,
    nothing written, and takes no index the room's next event would wait on."""
    kit = RoomKit(delivery_gap_timeout=5.0)
    await _room(kit)
    first = await kit.commit_event("r1", _record().model_copy(update={"idempotency_key": "k"}))
    again = _record("again").model_copy(update={"idempotency_key": "k"})

    with pytest.raises(IdempotencyKeyTakenError):
        await kit.commit_event("r1", again)

    assert await kit.store.get_event(again.id) is None
    sent = await asyncio.wait_for(
        kit.send_event("r1", "sms1", TextContent(body="hello")), timeout=2.0
    )
    assert sent.index == first.index + 1
    await kit.close()


async def test_a_record_behind_a_delivery_in_flight_is_passed_at_its_turn() -> None:
    """RFC §10.5, §13.5: the record's index waits for the delivery before it,
    then the lane passes it with nothing to run, and the next event's
    delivery waits on it no longer than on the events before it."""
    kit = RoomKit(delivery_gap_timeout=5.0)
    await _room(kit)
    held = _Held("out")
    kit.register_channel(held)
    await kit.attach_channel("r1", "out")
    first = asyncio.create_task(kit.send_event("r1", "sms1", TextContent(body="first")))
    await asyncio.wait_for(held.entered.wait(), timeout=1.0)

    committed = await kit.commit_event("r1", _record(), organization_id="tenant-a")

    assert await kit.store.get_delivered_index("r1") < committed.index - 1
    held.release.set()
    await asyncio.wait_for(first, timeout=1.0)
    sent = await asyncio.wait_for(
        kit.send_event("r1", "sms1", TextContent(body="next")), timeout=2.0
    )
    assert sent.index == committed.index + 1
    assert await kit.store.get_delivered_index("r1") == sent.index
    assert [e.content.body for e in held.delivered] == ["first", "next"]
    await kit.close()


async def test_a_record_is_no_activity() -> None:
    """RFC §5.1, §10.5: a trace nobody receives does not keep the room
    alive; ``last_activity_at`` stays where it was."""
    kit = RoomKit()
    await _room(kit)
    room = await kit.get_room("r1")
    long_ago = datetime(2026, 1, 1, tzinfo=UTC)
    timers = room.timers.model_copy(update={"last_activity_at": long_ago})
    await kit.store.update_room(room.model_copy(update={"timers": timers}))

    await kit.commit_event("r1", _record(), organization_id="tenant-a")

    assert (await kit.get_room("r1")).timers.last_activity_at == long_ago
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


async def test_a_record_is_stored_as_given_but_for_its_author_record() -> None:
    """RFC §10.5: a record is stored as its caller gives it, but for the
    record of its author, which the room fixes: one the caller wrote in its
    metadata is dropped, the rest of its metadata kept."""
    kit = RoomKit()
    await _room(kit)
    forged = {"name": "Someone else", "rank": 9}
    record = make_event(room_id="r1", channel_id="sms1", body="snapshot").model_copy(
        update={"metadata": {"author": forged, "keep": 1}}
    )

    committed = await kit.commit_event("r1", record)

    stored = await kit.store.get_event(committed.id)
    assert stored is not None
    assert stored.metadata.get("author") != forged
    assert stored.metadata.get("keep") == 1
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
