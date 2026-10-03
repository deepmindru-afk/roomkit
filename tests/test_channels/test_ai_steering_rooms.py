"""A steering directive addressed to a room reaches that room's loops only (RFC §21.3).

One AI channel object serves every room it is bound to, so its running loops
can belong to several rooms' turns at once. Addressed to a room, a ``Cancel``
reaches every loop of the room and any other directive the room's most recent
one; ``steer`` says how many loops it reached.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.steering import Cancel, InjectMessage
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import _ToolLoopContext
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond


def _channel_with_loops(*rooms: str) -> tuple[AIChannel, list[_ToolLoopContext]]:
    """A channel running one loop per entry of *rooms*, started in that order."""
    ch = AIChannel("ai1", provider=MockAIProvider())
    loops = [_ToolLoopContext(loop_id=f"loop-{i}", room_id=room) for i, room in enumerate(rooms)]
    for ctx in loops:
        ch._active_loops[ctx.loop_id] = ctx
    return ch, loops


def _reached(ctx: _ToolLoopContext) -> bool:
    return not ctx.steering_queue.empty()


def test_a_cancel_addressed_to_a_room_reaches_every_loop_of_it_and_no_other() -> None:
    ch, (a1, b1, a2) = _channel_with_loops("A", "B", "A")

    assert ch.steer(Cancel(reason="stop"), room_id="A") == 2

    assert a1.cancel_event.is_set() and a2.cancel_event.is_set()
    assert not b1.cancel_event.is_set() and not _reached(b1)


def test_another_directive_reaches_the_rooms_most_recent_loop() -> None:
    ch, (a1, a2, b1) = _channel_with_loops("A", "A", "B")

    assert ch.steer(InjectMessage(content="also check the logs"), room_id="A") == 1

    assert _reached(a2)
    assert not _reached(a1) and not _reached(b1)


def test_a_room_with_no_running_loop_is_reached_by_nothing() -> None:
    ch, (b1,) = _channel_with_loops("B")

    assert ch.steer(Cancel(reason="stop"), room_id="A") == 0

    assert not b1.cancel_event.is_set()


def test_unaddressed_the_most_recent_loop_is_reached_whatever_its_room() -> None:
    ch, (a1, b1) = _channel_with_loops("A", "B")

    assert ch.steer(InjectMessage(content="hi")) == 1

    assert _reached(b1) and not _reached(a1)


def test_a_loop_and_a_room_are_not_addressed_together() -> None:
    ch, _loops = _channel_with_loops("A")

    with pytest.raises(ValueError, match="not both"):
        ch.steer(Cancel(reason="stop"), loop_id="loop-0", room_id="A")


class _HeldProvider(MockAIProvider):
    """Answers once *release* is set, so two turns run side by side."""

    def __init__(self, release: asyncio.Event, *, streaming: bool) -> None:
        super().__init__(ai_responses=[AIResponse(content="done")], streaming=streaming)
        self._release = release

    async def generate(self, context: AIContext) -> AIResponse:
        await self._release.wait()
        return await super().generate(context)


async def _turn(ch: AIChannel, room_id: str) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    return await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id=room_id))
    )


async def test_a_stop_in_one_room_leaves_the_other_rooms_turn_running(streaming: bool) -> None:
    """Two members' turns on one shared channel: a Stop pressed in room A
    cancels A's turn, and B's turn answers."""
    release = asyncio.Event()
    ch = AIChannel("ai1", provider=_HeldProvider(release, streaming=streaming))
    turn_a = asyncio.create_task(_turn(ch, "A"))
    turn_b = asyncio.create_task(_turn(ch, "B"))
    for _ in range(100):
        if len(ch._active_loops) == 2:
            break
        await asyncio.sleep(0)

    reached = ch.steer(Cancel(reason="stop pressed"), room_id="A")
    release.set()
    run_a, run_b = await asyncio.gather(turn_a, turn_b)

    assert reached == 1
    assert run_a.reason == "cancelled"
    assert (run_b.reason, run_b.text) == ("completed", "done")
