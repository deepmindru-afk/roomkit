"""A room's activity (RFC §5.1).

The events a room stores DELIVERED through the pipeline are its activity:
each such commit moves ``timers.last_activity_at``. A lifecycle record, a
record stored BLOCKED and a record committed outside the pipeline are not.
"""

from __future__ import annotations

from datetime import UTC, datetime

from roomkit import HookResult, HookTrigger, RoomContext, RoomKit, RoomTimers
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventStatus, EventType, Visibility
from roomkit.models.event import RoomEvent, SystemContent, TextContent
from tests.test_framework import SimpleChannel

_LONG_AGO = datetime(2026, 1, 1, tzinfo=UTC)


async def _kit() -> RoomKit:
    """A kit with room ``r1``, its channel ``sms1`` bound, whose last
    activity is long ago."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1", timers=RoomTimers(inactive_after_seconds=3600))
    await kit.attach_channel("r1", "sms1")
    room = await kit.get_room("r1")
    timers = room.timers.model_copy(update={"last_activity_at": _LONG_AGO})
    await kit.store.update_room(room.model_copy(update={"timers": timers}))
    return kit


async def test_a_lifecycle_record_is_no_activity() -> None:
    """A mute does not restart the room's inactivity timer: its record takes
    the next index and counts in ``event_count``, in the shape §5.1 gives
    it, and leaves ``last_activity_at``."""
    kit = await _kit()
    count = (await kit.get_room("r1")).event_count

    await kit.mute("r1", "sms1")

    room = await kit.get_room("r1")
    [record] = await kit.store.list_events("r1", after_index=count - 1)
    assert (record.index, room.event_count) == (count, count + 1)
    assert record.type is EventType.CHANNEL_MUTED
    assert isinstance(record.content, SystemContent)
    assert (record.source.channel_id, record.visibility) == ("system", Visibility.INTERNAL)
    assert room.timers.last_activity_at == _LONG_AGO
    await kit.close()


async def test_a_record_stored_blocked_is_no_activity() -> None:
    """RFC §10.1 step 10: a message a hook blocks is stored BLOCKED, an
    audit record, and leaves ``last_activity_at``."""
    kit = await _kit()

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("refused")

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi")),
        room_id="r1",
    )

    assert result.blocked and result.event is not None
    assert result.event.status is EventStatus.BLOCKED
    assert (await kit.get_room("r1")).timers.last_activity_at == _LONG_AGO
    await kit.close()
