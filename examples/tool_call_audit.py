"""Tool call audit — every call, the refused ones included.

``ON_TOOL_CALL`` fires for the outcome of each tool call. Registered **async**
it observes them all: the call that ran, the one the tool policy denied, the
name the agent does not have, the handler that raised. Each states its outcome
with ``is_error`` instead of leaving it to be read out of the result text.

Registered **sync** it is a servant instead — it can block a call or replace
its result — and a refused call never reaches it, so a denial prevents the side
effect rather than hiding it. The ledger below prints both counts.

Run with:
    uv run python examples/tool_call_audit.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomKit,
    TextContent,
    ToolCallEvent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.policy import ToolPolicy

logger = setup_logging("examples.tool_call_audit")

TOOLS = [
    {"name": "get_weather", "description": "Weather for a city", "parameters": {}},
    {"name": "wire_money", "description": "Move money", "parameters": {}},
    {"name": "flaky_lookup", "description": "Calls an integration gateway", "parameters": {}},
]


@dataclass
class Entry:
    """One audited call, as an audit trail would store it."""

    tool: str
    status: str
    body: str


async def main() -> None:
    ledger: list[Entry] = []
    served: list[str] = []

    async def tool_handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "get_weather":
            return '{"temp_c": 22}'
        if name == "flaky_lookup":
            raise RuntimeError("integration gateway unreachable")
        return '{"ok": true}'

    # One call per round: served, denied by policy, undeclared, raising.
    calls = ["get_weather", "wire_money", "unknown_tool", "flaky_lookup"]
    responses = [
        AIResponse(
            content=f"Calling {name}.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=f"tc{i}", name=name, arguments={"city": "Paris"})],
        )
        for i, name in enumerate(calls)
    ]
    responses.append(AIResponse(content="Here is what I could do.", finish_reason="stop"))

    kit = RoomKit()
    ai = AIChannel(
        "ai-agent",
        provider=MockAIProvider(ai_responses=responses, streaming=False),
        tool_handler=tool_handler,
        # wire_money is declared to the model and denied to this agent — the
        # refusal an audit trail must be able to show.
        tool_policy=ToolPolicy(deny=["wire_money"]),
    )
    ws = WebSocketChannel("ws-user")
    kit.register_channel(ai)
    kit.register_channel(ws)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, _ctx: RoomContext) -> HookResult:
        if event.result is None:
            # A dispatch, not an outcome: a realtime channel with no handler
            # asking its hooks to serve the call.
            return HookResult.allow()
        body = event.result if isinstance(event.result, str) else str(event.result)
        ledger.append(
            Entry(
                tool=event.name,
                status="failed" if event.is_error else "ok",
                body=body[:60].replace("\n", " "),
            )
        )
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="servant")
    async def servant(event: ToolCallEvent, _ctx: RoomContext) -> HookResult:
        served.append(event.name)
        return HookResult.allow()

    await kit.create_room(room_id="audit-demo")
    await kit.attach_channel("audit-demo", "ws-user")
    await kit.attach_channel(
        "audit-demo",
        "ai-agent",
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": TOOLS},
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Do everything you can."),
        )
    )

    print("\n  Audit trail")
    print(f"  {'tool':<14} {'status':<8} result")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 40}")
    for entry in ledger:
        print(f"  {entry.tool:<14} {entry.status:<8} {entry.body}")

    refused = sum(1 for e in ledger if e.status == "failed")
    print(f"\n  friction: {refused}/{len(ledger)} calls refused or failed")
    print(f"  reached a serving (sync) hook: {served}")
    print("  a refused call is audited, and never served\n")


if __name__ == "__main__":
    asyncio.run(main())
