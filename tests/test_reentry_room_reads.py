"""A response event's commit reads the room once (RMK-331, RFC §10.1 steps 6 and 12).

Each response event of a non-streamed turn is committed on its own, under a
fresh room lock: in a reentry pass after a turn, one commit per event after a
regenerate. The commit reads what the lock protects once, the room's
context, and its status gate and its source's right to write read that
context rather than the store again. A close landing just before the lock
is taken is then seen by the gate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from roomkit.channels.agent import Agent
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, RoomStatus
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.buffered_agent import BufferedAgent
from tests.test_framework import SimpleChannel

# The reads of the commit running, by store method.
_COMMIT: ContextVar[dict[str, int] | None] = ContextVar("_COMMIT", default=None)


class _CountingStore(InMemoryStore):
    """Counts the room and binding reads made inside a counted commit."""

    async def get_room(self, room_id: str) -> Any:
        _count("get_room")
        return await super().get_room(room_id)

    async def get_binding(self, room_id: str, channel_id: str) -> Any:
        _count("get_binding")
        return await super().get_binding(room_id, channel_id)


def _count(method: str) -> None:
    reads = _COMMIT.get()
    if reads is not None:
        reads[method] = reads.get(method, 0) + 1


def _count_commits(kit: RoomKit, method: str) -> list[dict[str, int]]:
    """The reads of each call to *method* on *kit*, one dict per call."""
    commits: list[dict[str, int]] = []
    commit: Callable[..., Any] = getattr(kit, method)

    async def counting(*args: Any, **kwargs: Any) -> Any:
        reads: dict[str, int] = {}
        commits.append(reads)
        token = _COMMIT.set(reads)
        try:
            return await commit(*args, **kwargs)
        finally:
            _COMMIT.reset(token)

    setattr(kit, method, counting)
    return commits


async def _say(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )


async def _buffered_room() -> RoomKit:
    """An agent that answers at once, ten rows a turn: the buffered path."""
    kit = RoomKit(store=_CountingStore())
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(BufferedAgent("ai1", *(f"Row {i}." for i in range(10)), rows=10))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit


async def test_each_reentry_pass_reads_the_room_once_and_no_binding() -> None:
    kit = await _buffered_room()
    passes = _count_commits(kit, "_run_reentry_pass")

    await _say(kit)

    # A pass per row of the buffered answer.
    assert len(passes) == 10
    assert all(reads == {"get_room": 1} for reads in passes), passes
    await kit.close()


async def test_each_regenerated_answer_reads_the_room_once_and_no_binding() -> None:
    kit = await _buffered_room()
    await _say(kit)
    passes = _count_commits(kit, "_run_reentry_pass")

    await kit.regenerate_response("r1")

    # A regenerated answer re-enters like a first-time one: a pass per row.
    assert len(passes) == 10
    assert all(reads == {"get_room": 1} for reads in passes), passes
    await kit.close()


def _close_before_the_lock(kit: RoomKit) -> None:
    """Close the room once, just before the next room lock is taken."""
    locked = kit._lock_manager.locked
    closed: list[str] = []

    @asynccontextmanager
    async def closing_first(room_id: str) -> AsyncIterator[None]:
        if not closed:
            closed.append(room_id)
            await kit.close_room(room_id)
        async with locked(room_id):
            yield

    kit._lock_manager.locked = closing_first  # type: ignore[method-assign]


async def test_a_close_landing_before_the_lock_refuses_the_greeting() -> None:
    """RFC §5.1 / §10.1 step 6: the status gate reads the room under the
    lock, so a close landing between the call and the lock is seen."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("t1"))
    kit.register_channel(
        Agent("ai1", provider=MockAIProvider(responses=["x"]), greeting="Hi!", auto_greet=False)
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "t1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    _close_before_the_lock(kit)

    await kit.send_greeting("r1")

    room = await kit.get_room("r1")
    events = await kit.store.list_events("r1")
    assert room.status is RoomStatus.CLOSED
    assert not [e for e in events if (e.metadata or {}).get("auto_greeting")]
    await kit.close()
