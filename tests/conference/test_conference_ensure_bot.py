"""``ConferenceChannel.ensure_bot``: a host's join, awaited (RFC §12.10.4).

The lazy join's triggers start the join without waiting on it; a host that
must know the bot is in awaits it here. One join for concurrent calls, the
live session returned as it is, a lost session joined again, and a room the
channel is not attached to refused without a record left behind.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, MockConferenceBackend, RoomKit
from roomkit.channels.conference import ConferenceChannel
from roomkit.core.exceptions import RoomNotAttachedError
from roomkit.voice.stt.mock import MockSTTProvider

ROOM = "room-1"


async def _kit() -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider())
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel, backend


async def test_the_join_is_done_and_announced_when_it_returns() -> None:
    kit, channel, backend = await _kit()
    started: list[Any] = []

    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC, name="started")
    async def on_started(event: Any, ctx: Any) -> None:
        started.append(event)

    assert backend.bots == []  # lazy until something needs it

    bot = await channel.ensure_bot(ROOM)

    assert backend.bots == [bot]
    assert len(started) == 1
    await kit.close()


async def test_concurrent_calls_join_once_and_a_live_session_is_returned_as_is() -> None:
    kit, channel, backend = await _kit()

    first, second = await asyncio.gather(channel.ensure_bot(ROOM), channel.ensure_bot(ROOM))

    assert first is second
    assert await channel.ensure_bot(ROOM) is first
    assert backend.bots == [first]
    await kit.close()


async def test_a_lost_session_is_joined_again() -> None:
    kit, channel, backend = await _kit()
    lost = await channel.ensure_bot(ROOM)
    await backend.simulate_bot_disconnected(lost)

    again = await channel.ensure_bot(ROOM)

    assert again is not lost
    assert backend.bots == [again]
    await kit.close()


async def test_a_room_the_channel_is_not_attached_to_is_refused() -> None:
    kit, channel, backend = await _kit()

    with pytest.raises(RoomNotAttachedError):
        await channel.ensure_bot("elsewhere")

    assert backend.bots == []
    assert "elsewhere" not in channel._rooms  # no record left for an unknown room
    await kit.close()
