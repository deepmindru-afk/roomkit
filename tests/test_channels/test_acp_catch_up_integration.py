"""The ACP catch-up when the loaded tail stops short of the gap (RFC §19.3.2).

With no hook registered, the framework loads exactly the window the room's
channels declare (RMK-103), so the tail the catch-up reads can stop short of
what the agent missed. The first tests go through the framework rather than a
hand-built ``RoomContext``: that loading is the behaviour under test. The last
ones pin the header's wording on hand-built tails.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import RoomKit
from roomkit.channels._acp_context import room_context_block
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import RoomEvent, TextContent
from tests.conftest import make_event
from tests.test_channels.test_acp import _channel, _context, _sent
from tests.test_framework import SimpleChannel

ROOM = "room-1"


async def _room(channel: Any) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, channel.channel_id, category=ChannelCategory.INTELLIGENCE)
    assert not kit.hook_engine.has_hooks()
    return kit


async def _say(kit: RoomKit, body: str, *, to: list[str]) -> None:
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms", sender_id="marie", content=TextContent(body=body), addressed_to=to
        ),
        room_id=ROOM,
    )
    await asyncio.sleep(0)


async def test_a_tail_short_of_the_gap_is_reported_as_partial(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False, room_history=5)
    kit = await _room(channel)
    for number in range(12):
        await _say(kit, f"said without you {number}", to=[])  # solicits nobody

    await _say(kit, "what did I miss?", to=[channel.channel_id])

    sent = _sent(connection)
    assert "the 5 most recent messages you did not receive; up to" in sent
    assert "earlier room events were not loaded" in sent
    assert "said without you 11" in sent and "said without you 6" not in sent
    assert sent.endswith("what did I miss?")
    await kit.close()


async def test_a_contributor_sees_the_frameworks_tail_not_a_floor(tmp_path: Any) -> None:
    """With ``room_history=0`` and no hook nothing bound reads history, so the
    framework loads none: the tail holds the triggering event alone, and a
    contributor that needs the room reads the store."""
    seen: list[list[RoomEvent]] = []

    async def contribute(context: RoomContext, trigger: RoomEvent) -> list[str]:
        seen.append(list(context.recent_events))
        return ["Policy: be brief"]

    channel, connection, _ = _channel(
        tmp_path, emit_updates=False, room_history=0, context_contributor=contribute
    )
    kit = await _room(channel)
    await _say(kit, "earlier line", to=[])

    await _say(kit, "go", to=[channel.channel_id])

    assert [[event.content.body for event in tail] for tail in seen] == [["go"]]
    assert _sent(connection) == "Policy: be brief\n\ngo"
    await kit.close()


def _header(*tail: RoomEvent, after_index: int, limit: int = 5) -> str:
    """The first line of the block for a tail whose last event is the trigger."""
    block = room_context_block(
        _context(*tail), "acp-agent", after_index=after_index, trigger=tail[-1], limit=limit
    )
    return block.split("\n")[0]


def _message(index: int, **fields: Any) -> RoomEvent:
    return make_event(room_id=ROOM, body=f"line {index}", index=index, **fields)


def test_a_tail_of_nothing_new_still_says_the_gap_was_not_loaded() -> None:
    """The agent's own turn can fill the tail with tool calls: silence would
    read as a complete catch-up."""
    tools = [_message(i, type=EventType.TOOL_CALL_START) for i in range(20, 25)]

    header = _header(*tools, _message(25), after_index=3)

    assert header == (
        "[Room context — none of the loaded messages are new to you; up to 16 earlier "
        "room events were not loaded. Context only; the request follows.]"
    )


def test_both_cuts_are_named_when_both_apply() -> None:
    loaded = [_message(i) for i in range(20, 30)]

    header = _header(*loaded, _message(30), after_index=3)

    assert header == (
        "[Room context — the 5 most recent of 10 loaded messages you did not receive; "
        "up to 16 earlier room events were not loaded. Context only; the request follows.]"
    )


def test_one_of_each_reads_in_the_singular() -> None:
    header = _header(_message(5), _message(6), after_index=3)

    assert header == (
        "[Room context — the 1 most recent message you did not receive; up to 1 earlier "
        "room event was not loaded. Context only; the request follows.]"
    )
