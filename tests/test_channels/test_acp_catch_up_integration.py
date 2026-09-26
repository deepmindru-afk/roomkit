"""The ACP catch-up on a real RoomKit room with no hook (RFC §19.3.2).

With no hook registered, the framework loads exactly the window the room's
channels declare (RMK-103), so the tail the catch-up reads can stop short of
what the agent missed. These tests go through the framework rather than a
hand-built ``RoomContext``: that loading is the behaviour under test.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import RoomEvent, TextContent
from tests.test_channels.test_acp import _channel, _sent
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
