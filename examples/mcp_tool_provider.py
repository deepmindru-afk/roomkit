"""compose_tool_handlers — local tools and an MCP server's tools behind one handler.

Demonstrates how to combine MCP-discovered tools with custom local tools
using compose_tool_handlers, and wire them into an AIChannel.

No MCP server is needed: the MCP side is a plain handler with the same
contract as ``MCPToolProvider.as_tool_handler()``, and its tool definition
stands in for ``MCPToolProvider.get_tools()``. ``examples/mcp_stdio_tools.py``
plugs a real MCP server. The AI is a MockAIProvider scripted to call one tool
of each kind, so both go through the composed handler.

Run with:
    uv run python examples/mcp_tool_provider.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import json
from typing import Any

from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.compose import compose_tool_handlers

setup_logging("mcp_tool_provider")


# -- Custom local tool handler ------------------------------------------------

CLOCK_TOOL = AITool(
    name="get_time",
    description="Get the current time",
    parameters={
        "type": "object",
        "properties": {"timezone": {"type": "string", "default": "UTC"}},
    },
)


async def local_tool_handler(name: str, arguments: dict[str, Any]) -> str:
    """Handle locally-defined tools."""
    if name == "get_time":
        tz = arguments.get("timezone", "UTC")
        return json.dumps({"time": "2026-02-16T12:00:00", "timezone": tz})
    return json.dumps({"error": f"Unknown tool: {name}"})


# -- Simulated MCP tools (in a real app these come from MCPToolProvider) ------

MCP_SEARCH_TOOL = AITool(
    name="web_search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


async def mcp_tool_handler(name: str, arguments: dict[str, Any]) -> str:
    """Simulates an MCP server's tool handler."""
    if name == "web_search":
        query = arguments.get("query", "")
        return json.dumps({"results": [f"Result for: {query}"]})
    return json.dumps({"error": f"Unknown tool: {name}"})


# -- Main ---------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    # Compose handlers: local tools first, then MCP tools
    combined = compose_tool_handlers(local_tool_handler, mcp_tool_handler)

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        # Scripted model turns: a local tool call, an MCP tool call, then the answer.
        provider=MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        AIToolCall(id="call-1", name="get_time", arguments={"timezone": "UTC"})
                    ],
                ),
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        AIToolCall(id="call-2", name="web_search", arguments={"query": "RoomKit"})
                    ],
                ),
                AIResponse(content="It is 12:00 UTC, and the search found a result for RoomKit."),
            ]
        ),
        system_prompt="You are a helpful assistant with access to local and MCP tools.",
        tools=[CLOCK_TOOL, MCP_SEARCH_TOOL],
        tool_handler=combined,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv, room_id="demo-room")

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-user")
    await kit.attach_channel(
        "demo-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
    )

    # -- Demo: tools are dispatched through composed handler --
    print("=== Composed Tool Handler Demo ===\n")

    # Local tool
    result = await combined("get_time", {"timezone": "America/Montreal"})
    print(f"  get_time → {result}")

    # MCP tool
    result = await combined("web_search", {"query": "RoomKit framework"})
    print(f"  web_search → {result}")

    # Unknown tool (falls through both)
    result = await combined("nonexistent", {})
    print(f"  nonexistent → {result}")

    # -- Demo: end-to-end with AIChannel --
    print("\n=== AIChannel Integration ===\n")

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What time is it, and what is RoomKit?"),
        )
    )

    for ev in inbox:
        # The tool calls reach the room too, as ToolCallContent.
        if ev.source.channel_id == "ai-assistant" and isinstance(ev.content, TextContent):
            print(f"  AI: {ev.content.body}")

    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
