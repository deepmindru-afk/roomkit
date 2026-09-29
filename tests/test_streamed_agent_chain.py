"""A started response is read, whichever pass started it (RMK-287).

RFC §8.3 and §10.1 step 14: an agent's output is an event like any other.
A streamed segment, a greeting and a regenerated answer each solicit the
other agents; what they answer re-enters, and a stream one of them starts is
read, never discarded. The chain stops at ``max_chain_depth`` in every mode.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.core.event_router import StreamingResponse
from roomkit.core.framework import RoomKit
from roomkit.core.lanes import DeliveryCascade
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventStatus, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

_MODES = [(False, False), (True, True), (False, True), (True, False)]
_MODE_IDS = ["buffered", "streamed", "a-buffered-b-streamed", "a-streamed-b-buffered"]


def _answers(tag: str) -> list[AIResponse]:
    return [AIResponse(content=f"{tag}{i}") for i in range(20)]


async def _room(kit: RoomKit, *agents: AIChannel) -> SimpleChannel:
    sms = SimpleChannel("sms1")
    kit.register_channel(sms)
    for agent in agents:
        kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    for agent in agents:
        await kit.attach_channel("r1", agent.channel_id, category=ChannelCategory.INTELLIGENCE)
    return sms


async def _say(kit: RoomKit, body: str = "hi") -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))
    )


async def _agent_rows(kit: RoomKit, *agent_ids: str) -> list[RoomEvent]:
    events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
    return [e for e in events if e.source.channel_id in agent_ids]


@pytest.mark.parametrize(("a_streams", "b_streams"), _MODES, ids=_MODE_IDS)
async def test_two_agents_chain_to_the_limit_in_every_mode(
    a_streams: bool, b_streams: bool
) -> None:
    pa = MockAIProvider(streaming=a_streams, ai_responses=_answers("A"))
    pb = MockAIProvider(streaming=b_streams, ai_responses=_answers("B"))
    kit = RoomKit(max_chain_depth=3)
    await _room(kit, AIChannel("a", provider=pa), AIChannel("b", provider=pb))

    await _say(kit)

    rows = await _agent_rows(kit, "a", "b")
    delivered = sorted(
        (e.source.channel_id, e.chain_depth) for e in rows if e.status == EventStatus.DELIVERED
    )
    blocked = sorted(
        (e.source.channel_id, e.chain_depth, e.blocked_by)
        for e in rows
        if e.status == EventStatus.BLOCKED
    )
    # Each answers the human (depth 1) and the other's answer (depth 2); the
    # next answer would reach the limit, so neither is asked a third time.
    assert delivered == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    assert blocked == [("a", 3, "event_chain_depth_limit"), ("b", 3, "event_chain_depth_limit")]
    assert (len(pa.calls), len(pb.calls)) == (2, 2)
    await kit.close()


async def test_a_buffered_answer_to_a_streamed_segment_is_stored_with_its_tools() -> None:
    """The turn another agent runs on a streamed segment is read, tools included."""
    ran: list[str] = []

    async def charge(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    streamer = MockAIProvider(streaming=True, ai_responses=[AIResponse(content="A0")] * 4)
    worker = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="charge", arguments={})],
            ),
            AIResponse(content="charged"),
        ]
    )
    kit = RoomKit(max_chain_depth=3)
    await _room(
        kit,
        AIChannel("a", provider=streamer),
        AIChannel(
            "b",
            provider=worker,
            tool_handler=charge,
            tools=[AITool(name="charge", description="d")],
        ),
    )

    await _say(kit)

    answer_to_a = [
        (e.type, e.status)
        for e in await _agent_rows(kit, "b")
        if e.chain_depth == 2 and e.status == EventStatus.DELIVERED
    ]
    assert answer_to_a == [
        (EventType.TOOL_CALL_START, EventStatus.DELIVERED),
        (EventType.TOOL_CALL_END, EventStatus.DELIVERED),
        (EventType.MESSAGE, EventStatus.DELIVERED),
    ]
    assert ran.count("charge") >= 1
    await kit.close()


async def test_a_greeting_is_answered_by_the_other_agent(streaming: bool) -> None:
    other = MockAIProvider(streaming=streaming, ai_responses=[AIResponse(content="hello back")])
    kit = RoomKit(max_chain_depth=2)
    await _room(
        kit,
        Agent("greeter", provider=MockAIProvider(), greeting="welcome"),
        AIChannel("other", provider=other),
    )

    await kit.send_greeting("r1", agent_id="greeter")

    answers = [(e.content.body, e.chain_depth) for e in await _agent_rows(kit, "other")]
    assert answers == [("hello back", 1)]
    await kit.close()


async def test_a_regenerated_answer_is_answered_by_the_other_agent(streaming: bool) -> None:
    first = MockAIProvider(streaming=streaming, ai_responses=_answers("A"))
    second = MockAIProvider(streaming=streaming, ai_responses=_answers("B"))
    kit = RoomKit(max_chain_depth=3)
    await _room(kit, AIChannel("a", provider=first), AIChannel("b", provider=second))
    await _say(kit)
    before = {e.id for e in await _agent_rows(kit, "a", "b")}

    await kit.regenerate_response("r1")

    new = [e for e in await _agent_rows(kit, "a", "b") if e.id not in before]
    delivered_depths = sorted(
        (e.source.channel_id, e.chain_depth) for e in new if e.status == EventStatus.DELIVERED
    )
    # Both regenerate (depth 1) and each answers the other's new answer (2).
    assert delivered_depths == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    await kit.close()


async def test_the_answer_scope_holds_along_a_streamed_chain(streaming: bool) -> None:
    """A trigger's response scope binds what an agent answers to a segment too."""
    pa = MockAIProvider(streaming=streaming, ai_responses=_answers("A"))
    pb = MockAIProvider(streaming=streaming, ai_responses=_answers("B"))
    kit = RoomKit(max_chain_depth=3)
    await _room(kit, AIChannel("a", provider=pa), AIChannel("b", provider=pb))
    outsider = SimpleChannel("sms2")
    kit.register_channel(outsider)
    await kit.attach_channel("r1", "sms2")

    await kit.process_inbound(
        InboundMessage(
            channel_id="sms1",
            sender_id="u1",
            content=TextContent(body="hi"),
            response_visibility="sms1,a,b",
        )
    )

    answers = [e for e in await _agent_rows(kit, "a", "b") if e.status == EventStatus.DELIVERED]
    assert {e.chain_depth for e in answers} == {1, 2}
    assert {e.visibility for e in answers} == {"sms1,a,b"}
    assert [e for e in outsider.delivered if e.source.channel_id in ("a", "b")] == []
    await kit.close()


async def test_a_chained_stream_past_the_reentry_budget_is_closed_unread() -> None:
    kit = RoomKit()
    await _room(kit, AIChannel("a", provider=MockAIProvider()))
    started: list[str] = []

    async def stream() -> Any:
        started.append("read")
        yield "never"

    trigger = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS),
        content=TextContent(body="x"),
        chain_depth=1,
    )
    cascade = DeliveryCascade("r1", reentry_budget=0)
    cascade.add_streams(
        [
            StreamingResponse(
                stream=stream(),
                source_channel_id="a",
                source_channel_type=ChannelType.AI,
                trigger_event=trigger,
            )
        ],
        chained=True,
    )

    error, record = await kit._process_streaming_responses(cascade, "r1")

    assert (error, dict(record), started) == (None, {}, [])
    [row] = await _agent_rows(kit, "a")
    assert (row.status, row.blocked_by, row.chain_depth) == (
        EventStatus.BLOCKED,
        "reentry_loop_cap",
        2,
    )
    await kit.close()
