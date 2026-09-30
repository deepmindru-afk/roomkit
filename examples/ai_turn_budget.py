"""AI Turn Budget — cap what one agent turn may spend.

An agent working through its tools bills every round. ``turn_budget_usd``
(the cost at the model's catalogue price) and ``turn_budget_tokens`` (every
billed token, cache included) stop the tool loop at the first round boundary
where the turn has reached its budget: the calls that round asked for do not
run, no further generation is asked for, and ``ON_AI_RESPONSE`` reports
``loop_end_reason="budget_exceeded"`` with the turn's usage. A budget can also
be set per room (binding metadata) or per turn (``AIChannelTurnConfig``).

The agent here is asked to look up thirty orders one by one on a budget
of 2 cents, which it cannot finish.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/ai_turn_budget.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, setup_logging

from roomkit import (
    AIResponseEvent,
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AITool
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

logger = setup_logging("ai_turn_budget")

LOOKUP = AITool(
    name="lookup_order",
    description="Look up one order by id (A-1 to A-30): its status and total.",
    parameters={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
)


async def lookup_order(_name: str, arguments: dict[str, Any]) -> str:
    order = str(arguments.get("id"))
    return f'{{"id": "{order}", "status": "shipped", "total": 42.0}}'


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")
    kit = RoomKit()

    user = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(
            AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-sonnet-5")
        ),
        tools=[LOOKUP],
        tool_handler=lookup_order,
        turn_budget_usd=0.02,  # this turn may cost at most 2 cents
    )
    kit.register_channel(user)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="budget")
    async def on_response(event: AIResponseEvent, ctx: RoomContext) -> None:
        logger.info(
            "Turn ended: %s after %d tool call(s), usage %s",
            event.loop_end_reason,
            event.tool_calls_count,
            event.usage,
        )

    await kit.create_room(room_id="budget-room")
    await kit.attach_channel("budget-room", "ws-user")
    await kit.attach_channel("budget-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(
                body="Look up orders A-1 to A-30, one lookup_order call per message, "
                "then give me the sum of their totals."
            ),
        )
    )
    await asyncio.sleep(0.5)  # ON_AI_RESPONSE runs async
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
