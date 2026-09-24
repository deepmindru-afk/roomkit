"""RoomKit -- MCP tools from a server started as a command (stdio), with a local model.

MCPToolProvider.from_command() starts an MCP server as a subprocess, lists its
tools and stops it at the end; its tools plug into an AIChannel like any other.
The model runs locally through llama.cpp, so nothing else needs to be running.

    AIChannel ── LlamaCppAIProvider (local model)
        └── tool_handler ── MCPToolProvider ── stdio ── examples/mcp_servers/notes_server.py

Requirements:
    pip install roomkit[llamacpp,mcp]

Run:
    uv run python examples/mcp_stdio_tools.py

Environment variables:
    MCP_COMMAND      Another MCP server to use, as a command line
                     (default: this repository's notes server)
    LLAMACPP_MODEL   GGUF to run (default: unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M)
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig
from roomkit.tools import MCPToolProvider

logger = setup_logging("mcp_stdio_tools")

NOTES_SERVER = Path(__file__).resolve().parent / "mcp_servers" / "notes_server.py"


async def main() -> None:
    command = shlex.split(os.environ.get("MCP_COMMAND", f"{sys.executable} {NOTES_SERVER}"))
    provider = LlamaCppAIProvider(
        LlamaCppConfig(
            model=os.environ.get("LLAMACPP_MODEL", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")
        )
    )
    try:
        await provider.start()

        async with MCPToolProvider.from_command(command[0], command[1:]) as mcp:
            logger.info("MCP tools: %s", mcp.tool_names)

            kit = RoomKit()
            user = WebSocketChannel("user")
            kit.register_channel(user)
            kit.register_channel(
                AIChannel(
                    "ai",
                    provider=provider,
                    system_prompt="You are a helpful assistant. Use the tools when they help.",
                    tools=mcp.get_tools(),
                    tool_handler=mcp.as_tool_handler(),
                )
            )

            replies: asyncio.Queue[str] = asyncio.Queue()

            async def on_event(_conn: str, event: RoomEvent) -> None:
                if event.source.channel_id == "ai" and isinstance(event.content, TextContent):
                    await replies.put(event.content.body)

            user.register_connection("me", on_event, room_id="notes")
            await kit.create_room(room_id="notes")
            await kit.attach_channel("notes", "user")
            await kit.attach_channel("notes", "ai", category=ChannelCategory.INTELLIGENCE)

            for question in (
                "Note that the team meeting moved to Thursday at 3pm.",
                "Also note: buy coffee.",
                "What notes do I have?",
            ):
                logger.info("You: %s", question)
                await kit.process_inbound(
                    InboundMessage(
                        channel_id="user", sender_id="me", content=TextContent(body=question)
                    )
                )
                logger.info("AI:  %s", await asyncio.wait_for(replies.get(), timeout=120))

            await kit.close()  # leaving the block then stops the MCP server
    finally:
        await provider.close()  # stops llama-server, even when a step above fails


if __name__ == "__main__":
    asyncio.run(main())
