"""What ON_THOUGHT reports when the agent speaks while its thinker is still
thinking (RMK-627, RFC §6.4): the thought a late call replaces is the one the
agent spoke with, emptied, not the one the call started from."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import MockSpeakPolicy, Thinker, Thought, ThoughtEvent
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext
from roomkit.providers.ai.mock import MockAIProvider
from tests.bounds import CUT, expire, opened_with_cut
from tests.conference.test_conference_realtime import until
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_BINDING = ChannelBinding(
    channel_id="ai1",
    room_id="r1",
    channel_type=ChannelType.AI,
    category=ChannelCategory.INTELLIGENCE,
)
_SMS = ChannelBinding(
    channel_id="sms1",
    room_id="r1",
    channel_type=ChannelType.SMS,
    category=ChannelCategory.TRANSPORT,
)
PRICE = Thought("They look for the price; I know it.", ("It costs 1,200 $ a year.",))
LATER = Thought("They settled on ten licences.", ("Ten gets the discount.",))


class _SlowSecondCall(Thinker):
    """The first call brings PRICE at once; the second lasts until released,
    then fails, keeps the thought (``same``), or brings LATER."""

    def __init__(self, second: str) -> None:
        self.second = second
        self.calls = 0
        self.release = asyncio.Event()

    async def think(self, previous: Thought, context: AIContext) -> Thought:
        self.calls += 1
        if self.calls == 1:
            return PRICE
        await self.release.wait()
        if self.second == "fails":
            raise RuntimeError("model down")
        return previous if self.second == "same" else LATER


def _context(*events: Any) -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[_BINDING, _SMS], recent_events=list(events))


@pytest.mark.parametrize(
    ("second", "late_report"),
    [
        ("fails", []),
        ("same", []),
        ("new", [(LATER.said(), PRICE.said())]),
    ],
)
async def test_a_call_that_ends_after_the_agent_spoke_reports_against_what_it_spoke_with(
    second: str, late_report: list[tuple[Thought, Thought]], monkeypatch: pytest.MonkeyPatch
) -> None:
    waits = opened_with_cut(monkeypatch)
    thinker = _SlowSecondCall(second)
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["Yes?"]),
        speak_policy=MockSpeakPolicy(["silent", "silent", "silent", "speak"]),
        thinker=thinker,
        think_wait=CUT,
    )
    seen: list[ThoughtEvent] = []

    async def hook(event: ThoughtEvent) -> None:
        seen.append(event)

    channel._thought_hook = hook
    said = [make_event(body=b, channel_id="sms1", room_id="r1") for b in ("a", "b", "Nova?")]
    await channel.on_event(said[0], _BINDING, _context(*said[:2]))  # the first call: PRICE
    second = asyncio.create_task(channel.on_event(said[1], _BINDING, _context(*said[:2])))
    await until(lambda: len(waits) == 2)  # the second call started, its event waits on it
    expire(waits)
    await second
    await respond(channel, said[2], _BINDING, _context(*said))  # speaks during the call
    thinker.release.set()
    await asyncio.wait_for(channel._minds["r1"]._task, 5)  # the call ended, reported
    await channel.close()

    assert thinker.calls == 2
    assert [(e.thought, e.previous) for e in seen] == [
        (PRICE, Thought()),
        (PRICE.said(), PRICE),
        *late_report,
    ]
    assert seen[1].duration_ms is None
