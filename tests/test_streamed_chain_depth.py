"""A response at the chain-depth limit is blocked, streamed or not (RMK-283).

RFC §8.3: when ``chain_depth >= max_chain_depth`` the response is stored
BLOCKED with ``blocked_by = "event_chain_depth_limit"`` and announced as
``chain_depth_exceeded``. A streamed response is generated and read to its
end like a buffered one; each of its segments is stored BLOCKED and
delivered to no channel.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventStatus, EventType
from roomkit.models.event import TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

_ANSWERS = [
    AIResponse(
        content="Calling.",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="t1", name="lookup", arguments={})],
    ),
    AIResponse(content="Answer.", finish_reason="stop"),
]


class _Voice(SimpleChannel):
    """A streaming transport that records every chunk it is handed live."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.live: list[Any] = []

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(self, text_stream, event, binding, context):  # type: ignore[no-untyped-def]
        async for chunk in text_stream:
            self.live.append(chunk)
        return ChannelOutput.empty()


async def _turn(
    streaming: bool, max_chain_depth: int, then: Any = None
) -> tuple[RoomKit, SimpleChannel, _Voice, list[str]]:
    kit = RoomKit(max_chain_depth=max_chain_depth)
    sms, voice = SimpleChannel("sms1"), _Voice("voice1")
    ran: list[str] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        if then is not None:
            await then(kit)
        return "ok"

    kit.register_channel(sms)
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=MockAIProvider(streaming=streaming, ai_responses=list(_ANSWERS)),
            tool_handler=lookup,
            tools=[AITool(name="lookup", description="d")],
        )
    )
    await kit.create_room(room_id="r1")
    for channel_id in ("sms1", "voice1"):
        await kit.attach_channel("r1", channel_id)
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit, sms, voice, ran


async def _say(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )


async def _ai_rows(kit: RoomKit) -> list[Any]:
    events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
    return [e for e in events if e.source.channel_id == "ai1"]


async def test_at_the_limit_every_segment_is_blocked(streaming: bool) -> None:
    kit, sms, voice, ran = await _turn(streaming, max_chain_depth=1)
    announced: list[Any] = []

    @kit.on("chain_depth_exceeded")
    async def on_exceeded(event: Any) -> None:
        announced.append(event)

    await _say(kit)

    rows = await _ai_rows(kit)
    assert [row.type for row in rows] == [
        EventType.MESSAGE,
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_END,
        EventType.MESSAGE,
    ]
    assert {(row.status, row.blocked_by, row.chain_depth) for row in rows} == {
        (EventStatus.BLOCKED, "event_chain_depth_limit", 1)
    }
    assert [e.data for e in announced] == [{"chain_depth": 1, "max_chain_depth": 1}] * len(rows)
    # One observation per blocked row, as a buffered response's blocked events get.
    observations = await kit.store.list_observations("r1")
    assert sorted(o.id for o in observations) == sorted(f"obs_{row.id}" for row in rows)
    # Generated and run, as a buffered response is; delivered to nobody.
    assert ran == ["lookup"]
    assert sms.delivered == []
    assert voice.live == []


async def test_the_limit_holds_at_its_default(streaming: bool) -> None:
    """A trigger at depth 3 is answered at 4 and delivered; one at 4 would be
    answered at 5, the default limit, and is blocked."""
    for trigger_depth, blocked in ((3, False), (4, True)):
        kit, _, _, _ = await _turn(streaming, max_chain_depth=5)

        await kit.send_event("r1", "sms1", TextContent(body="go"), chain_depth=trigger_depth)

        rows = await _ai_rows(kit)
        assert rows
        assert {row.chain_depth for row in rows} == {trigger_depth + 1}
        assert all((row.status == EventStatus.BLOCKED) is blocked for row in rows)


async def test_a_room_closed_while_blocked_rows_are_written_takes_no_more(
    streaming: bool,
) -> None:
    async def close_the_room(kit: RoomKit) -> None:
        await kit.close_room("r1")

    kit, _, _, _ = await _turn(streaming, max_chain_depth=1, then=close_the_room)

    await _say(kit)

    rows = await _ai_rows(kit)
    assert EventType.TOOL_CALL_END not in [row.type for row in rows]


async def test_below_the_limit_nothing_is_blocked(streaming: bool) -> None:
    kit, sms, _, _ = await _turn(streaming, max_chain_depth=5)

    await _say(kit)

    rows = await _ai_rows(kit)
    assert rows
    assert all(row.status != EventStatus.BLOCKED for row in rows)
    assert all(row.chain_depth == 1 for row in rows)
    assert sms.delivered
