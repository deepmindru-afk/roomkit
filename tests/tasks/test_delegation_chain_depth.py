"""A delegation's result continues the chain of the turn that delegated (RMK-287).

RFC §23.3 step 8: the result delivered back to the room carries the depth of
the response whose turn delegated, so a cycle of delegation, result and
delegation again ends at ``max_chain_depth`` (§8.3) instead of running
without end.
"""

from __future__ import annotations

import asyncio

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.worker import execute_delivery
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventStatus, EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import DelegateHandler, setup_delegation
from tests.test_framework import SimpleChannel

# A model that delegates again whenever it is handed a result.
_DELEGATES_AGAIN = [
    AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            AIToolCall(id="c1", name="delegate_task", arguments={"agent": "worker", "task": "X"})
        ],
    ),
    AIResponse(content="delegated"),
]


async def _delegating_room(
    streaming: bool, max_chain_depth: int
) -> tuple[RoomKit, MockAIProvider]:
    kit = RoomKit(max_chain_depth=max_chain_depth)
    front = MockAIProvider(streaming=streaming, ai_responses=list(_DELEGATES_AGAIN))
    agent = AIChannel("front", provider=front, tool_search=False)
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(agent)
    kit.register_channel(
        AIChannel("worker", provider=MockAIProvider(streaming=streaming, responses=["found"]))
    )
    setup_delegation(agent, DelegateHandler(kit))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "front", category=ChannelCategory.INTELLIGENCE)
    return kit, front


async def _timeline(kit: RoomKit) -> list[RoomEvent]:
    return await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))


def _answers(events: list[RoomEvent]) -> list[int]:
    """The chain depth of each answer the front agent delivered."""
    return [
        e.chain_depth
        for e in events
        if e.source.channel_id == "front"
        and e.type == EventType.MESSAGE
        and e.status == EventStatus.DELIVERED
    ]


async def _until_blocked(kit: RoomKit, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not any(e.status == EventStatus.BLOCKED for e in await _timeline(kit)):
            await asyncio.sleep(0.01)


async def test_a_delegation_cycle_ends_at_max_chain_depth(streaming: bool) -> None:
    kit, front = await _delegating_room(streaming, max_chain_depth=3)

    await asyncio.wait_for(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        ),
        timeout=5.0,
    )
    await _until_blocked(kit)
    await asyncio.sleep(0.1)

    events = await _timeline(kit)
    # Each result reaches the agent as an instruction one deeper than the
    # answer that delegated (never stored, RMK-310): the agent answers the
    # human at 1 and the first result at 2, until the answer to the last one
    # would reach the limit.
    assert _answers(events) == [1, 2]
    [record] = [e for e in events if e.status == EventStatus.BLOCKED]
    assert (record.source.channel_id, record.chain_depth) == ("front", 3)
    # Three turns (two model calls each): the human's, then one per result
    # below the limit. Nothing is generated for the result at depth 2.
    assert len(front.calls) == 4
    children = [r for r in await kit.store.list_rooms() if r.id.startswith("r1::task-")]
    assert len(children) == 2
    await kit.close()


async def test_a_delegation_outside_a_tool_call_opens_a_chain() -> None:
    """``kit.delegate`` called by host code has no turn above it: its result is
    delivered at depth 0, as a person's message is, and the cycle the agent
    starts from there still ends at the limit."""
    kit, _ = await _delegating_room(streaming=False, max_chain_depth=3)

    handle = await kit.delegate("r1", "worker", "look it up", notify="front")
    await handle.wait()
    await _until_blocked(kit)
    await asyncio.sleep(0.1)

    events = await _timeline(kit)
    # The first result at 0: the agent answers it at 1, the next result at 2,
    # and the answer to the last one is the refused one at the limit.
    assert _answers(events) == [1, 2]
    [record] = [e for e in events if e.status == EventStatus.BLOCKED]
    assert (record.source.channel_id, record.chain_depth) == ("front", 3)
    await kit.close()


async def _answering_room(streaming: bool) -> RoomKit:
    kit = RoomKit(max_chain_depth=5)
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel("ai1", provider=MockAIProvider(streaming=streaming, responses=["noted"]))
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit


async def _depths(kit: RoomKit) -> dict[str, int]:
    return {
        e.source.channel_id: e.chain_depth
        for e in await _timeline(kit)
        if e.source.channel_id in ("sms1", "ai1")
    }


async def test_delivered_content_continues_the_chain_it_names(streaming: bool) -> None:
    kit = await _answering_room(streaming)

    await kit.deliver("r1", "a result", channel_id="sms1", chain_depth=2)

    assert await _depths(kit) == {"sms1": 2, "ai1": 3}
    await kit.close()


async def test_a_queued_delivery_keeps_the_chain_it_names() -> None:
    """The depth survives the queue: a backend stores the item as JSON."""
    kit = await _answering_room(streaming=False)
    queued = DeliveryItem(room_id="r1", content="a result", channel_id="sms1", chain_depth=2)

    await execute_delivery(kit, DeliveryItem.model_validate_json(queued.model_dump_json()))

    assert await _depths(kit) == {"sms1": 2, "ai1": 3}
    await kit.close()


async def test_a_child_room_keeps_what_its_broadcast_blocked() -> None:
    """A delegated turn the limit stops leaves its record in the child room,
    as any broadcast does (RFC §8.3)."""
    kit = RoomKit(max_chain_depth=1)
    worker = MockAIProvider(responses=["never"])
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("worker", provider=worker))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    announced: list[str] = []

    @kit.on("chain_depth_exceeded")
    async def on_exceeded(event) -> None:  # type: ignore[no-untyped-def]
        announced.append(event.room_id)

    handle = await kit.delegate("r1", "worker", "look it up", wait=True)

    events = await kit.store.list_events(
        handle.child_room_id, event_filter=EventFilter(include_blocked=True)
    )
    [record] = [e for e in events if e.source.channel_id == "worker"]
    assert (record.status, record.blocked_by) == (EventStatus.BLOCKED, "event_chain_depth_limit")
    assert announced == [handle.child_room_id]
    assert worker.calls == []
    await kit.close()
