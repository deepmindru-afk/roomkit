"""Act between two rounds of a tool loop with AFTER_TOOL_ROUND.

A model calls two mailbox tools; the mail connection is down, so the first
round fails. An AFTER_TOOL_ROUND hook reads that round whole, withdraws the
mailbox tools for the rest of the turn and tells the model why. The model's
next attempt at a mailbox tool is refused, and it answers with what it has.
Shows:
- ToolRoundEvent: the round's calls and results, read together
- event.withdraw(): tools gone for the rest of the turn, refused if called
- event.add_message(): what the next round reads after the results
- A BEFORE_TOOL_USE hook counting the calls the turn started, read back on
  InboundResult.response_metadata

Run with:
    uv run python examples/hook_after_tool_round.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    EventType,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomKit,
    TextContent,
    ToolRoundEvent,
    WebSocketChannel,
)
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import current_response_metadata

logger = setup_logging("hook_after_tool_round")

MAIL_TOOLS = ("mail_search", "mail_read")
TOOLS = [
    AITool(name=name, description=f"{name} in the mailbox", parameters={"type": "object"})
    for name in MAIL_TOOLS
]


def _call(call_id: str, name: str) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments={})],
    )


async def mailbox(name: str, arguments: dict[str, Any]) -> str:
    """The mail connection is down: every call fails."""
    return json.dumps({"success": False, "error": "mail connection expired"})


async def count_calls(event: Any, ctx: RoomContext) -> HookResult:
    """BEFORE_TOOL_USE: count each call the turn starts into its record."""
    record = current_response_metadata()
    if record is not None:
        record["calls_started"] = record.get("calls_started", 0) + 1
    return HookResult.allow()


async def close_mailbox(event: ToolRoundEvent, ctx: RoomContext) -> HookResult:
    """AFTER_TOOL_ROUND: a round whose every call failed closes the mailbox."""
    failed = [r for r in event.results if '"success": false' in str(r.result)]
    if failed and len(failed) == len(event.results):
        event.withdraw(*MAIL_TOOLS)
        event.add_message("The mailbox is unavailable for the rest of this turn.")
        logger.info("round %d: mailbox tools withdrawn", event.round_index)
    return HookResult.allow()


async def main() -> None:
    model = MockAIProvider(
        ai_responses=[
            _call("c1", "mail_search"),
            _call("c2", "mail_read"),
            AIResponse(content="Your mailbox is unreachable right now; reconnect it."),
        ]
    )
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("member"))
    kit.register_channel(AIChannel("agent", provider=model, tools=TOOLS, tool_handler=mailbox))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "member")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)
    kit.hook(HookTrigger.BEFORE_TOOL_USE, name="count_calls")(count_calls)
    kit.hook(HookTrigger.AFTER_TOOL_ROUND, name="close_mailbox")(close_mailbox)

    result = await kit.process_inbound(
        InboundMessage(channel_id="member", sender_id="u1", content=TextContent(body="Any mail?"))
    )

    for event in await kit.store.list_events("room"):
        if event.type == EventType.TOOL_CALL_END:
            logger.info("%s: %s", event.content.tool_name, event.content.status)
    logger.info("calls started: %s", result.response_metadata.get("calls_started"))
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
