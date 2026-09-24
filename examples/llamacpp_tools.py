"""RoomKit -- A local model with tools, and nothing to install or start beside it.

LlamaCppAIProvider downloads the llama.cpp build for this machine (CUDA,
Metal or CPU, checked against a pinned SHA-256) and the model on first run,
starts llama-server on a local port, and stops it when the kit closes. Tool
calls use the model's own format, so the usual AIChannel tools just work.

Requirements:
    pip install roomkit[llamacpp]
    ~3 GB of disk for the default model; a GPU helps, the CPU works (slower)

Run:
    uv run python examples/llamacpp_tools.py

Environment variables:
    LLAMACPP_MODEL   GGUF to run, "repo:quant" or a .gguf path
                     (default: unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M)
    LLAMACPP_GPU_LAYERS   0 to force the CPU (default: as many as the GPU holds)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from roomkit.providers.ai.base import AITool
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig

logger = setup_logging("llamacpp_tools")

TOOLS = [
    AITool(
        name="get_time",
        description="The current local date and time",
        parameters={"type": "object", "properties": {}},
    ),
    AITool(
        name="roll_dice",
        description="Roll dice and return each result",
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "sides": {"type": "integer", "minimum": 2, "maximum": 100},
            },
            "required": ["count", "sides"],
        },
    ),
]


async def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool the model called and return its result as JSON."""
    logger.info("Tool call: %s(%s)", name, arguments)
    if name == "get_time":
        return json.dumps({"now": datetime.now().isoformat(timespec="minutes")})
    if name == "roll_dice":
        rolls = [random.randint(1, arguments["sides"]) for _ in range(arguments["count"])]  # noqa: S311
        return json.dumps({"rolls": rolls, "total": sum(rolls)})
    return json.dumps({"error": f"unknown tool {name}"})


async def main() -> None:
    gpu_layers = os.environ.get("LLAMACPP_GPU_LAYERS")
    provider = LlamaCppAIProvider(
        LlamaCppConfig(
            model=os.environ.get("LLAMACPP_MODEL", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"),
            gpu_layers=int(gpu_layers) if gpu_layers else None,
        )
    )
    logger.info("Starting the local model (the first run downloads it)...")
    await provider.start()

    kit = RoomKit()
    user = WebSocketChannel("user")
    kit.register_channel(user)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=provider,
            system_prompt="You are a helpful assistant. Use the tools when they help.",
            tools=TOOLS,
            tool_handler=run_tool,
        )
    )

    replies: asyncio.Queue[str] = asyncio.Queue()

    async def on_event(_conn: str, event: RoomEvent) -> None:
        if event.source.channel_id == "ai" and isinstance(event.content, TextContent):
            await replies.put(event.content.body)

    user.register_connection("me", on_event, room_id="local")
    await kit.create_room(room_id="local")
    await kit.attach_channel("local", "user")
    await kit.attach_channel("local", "ai", category=ChannelCategory.INTELLIGENCE)

    for question in ("What time is it?", "Roll three six-sided dice for me.", "Hi there!"):
        logger.info("You: %s", question)
        await kit.process_inbound(
            InboundMessage(channel_id="user", sender_id="me", content=TextContent(body=question))
        )
        logger.info("AI:  %s", await asyncio.wait_for(replies.get(), timeout=120))

    await kit.close()  # stops llama-server


if __name__ == "__main__":
    asyncio.run(main())
