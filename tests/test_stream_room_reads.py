"""A stream reads the room's status once, not once per row (RMK-302).

The status gate holds at every row a stream writes (RFC §5.1): a room closed
mid-stream takes no further row. The run holds the room as its context read
it and reads it again only when the kit closed or archived a room since.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel
from tests.test_streamed_chain_depth import _Voice


def _answers(rounds: int) -> list[AIResponse]:
    steps = [
        AIResponse(
            content=f"Step {i}.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=f"t{i}", name="lookup", arguments={})],
        )
        for i in range(rounds)
    ]
    return [*steps, AIResponse(content="Answer.", finish_reason="stop")]


async def _streamed_turn(rounds: int, during_tool: Any = None) -> tuple[RoomKit, list[str]]:
    """A streamed turn of *rounds* tool rounds; returns the kit and its status reads."""
    kit = RoomKit()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        if during_tool is not None:
            await during_tool(kit)
        return "ok"

    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(_Voice("voice1"))
    provider = MockAIProvider(streaming=True, ai_responses=_answers(rounds))
    tools = [AITool(name="lookup", description="d")]
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=lookup, tools=tools))
    for room_id in ("r1", "r2"):
        await kit.create_room(room_id=room_id)
    for channel_id in ("sms1", "voice1"):
        await kit.attach_channel("r1", channel_id)
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    status_reads: list[str] = []
    read_status = kit._room_refuses_writes

    async def counting(room_id: str) -> bool:
        status_reads.append(room_id)
        return await read_status(room_id)

    kit._room_refuses_writes = counting  # type: ignore[method-assign]
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go")),
        room_id="r1",
    )
    return kit, status_reads


async def _ai_rows(kit: RoomKit) -> list[EventType]:
    return [e.type for e in await kit.store.list_events("r1") if e.source.channel_id == "ai1"]


async def test_a_streamed_turn_reads_the_room_status_no_more_as_it_grows() -> None:
    kit, reads = await _streamed_turn(rounds=4)

    assert reads == []
    assert (await _ai_rows(kit)).count(EventType.TOOL_CALL_END) == 4
    await kit.close()


async def test_another_room_closing_costs_one_read_and_blocks_nothing() -> None:
    async def close_the_other_room(kit: RoomKit) -> None:
        await kit.close_room("r2")

    kit, reads = await _streamed_turn(rounds=1, during_tool=close_the_other_room)

    assert reads == ["r1"]
    assert (await _ai_rows(kit))[-1] == EventType.MESSAGE
    await kit.close()


async def test_this_room_closing_refuses_the_rows_after_it() -> None:
    async def close_this_room(kit: RoomKit) -> None:
        await kit.close_room("r1")

    kit, _ = await _streamed_turn(rounds=1, during_tool=close_this_room)

    assert EventType.TOOL_CALL_END not in await _ai_rows(kit)
    await kit.close()
