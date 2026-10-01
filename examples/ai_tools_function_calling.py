"""AI tools and function calling with per-room config.

Demonstrates how to configure AI channels with custom tools for
function calling, and how to set per-room AI configuration via
binding metadata. Shows:
- AITool definitions with JSON schema parameters
- Per-room system_prompt, temperature, and tools via binding metadata
- A tool_handler running the calls the model makes, its result going back
  to the model for the final answer
- MockAIProvider for testing without API keys (it scripts the tool calls a
  real model would decide on)

Run with:
    uv run python examples/ai_tools_function_calling.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider


async def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run the tool the model called; its JSON result goes back to the model."""
    print(f"  Tool call: {name}({arguments})")
    if name == "get_weather":
        return json.dumps({"city": arguments["city"], "temp_c": -5, "conditions": "snow"})
    if name == "search_restaurants":
        return json.dumps({"results": [{"id": "r1", "name": "Le Bouillon", "rating": 4.7}]})
    return json.dumps({"error": f"Unknown tool: {name}"})


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> AIResponse:
    """A model turn that asks for one tool call instead of answering."""
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments)],
    )


async def main() -> None:
    kit = RoomKit()

    # Scripted model turns: each question gets a tool call, then an answer
    # written from the tool's result.
    provider = MockAIProvider(
        ai_responses=[
            tool_call("call-1", "get_weather", {"city": "Montreal"}),
            AIResponse(content="It's -5°C and snowing in Montreal."),
            tool_call(
                "call-2",
                "search_restaurants",
                {"location": "downtown Montreal", "cuisine": "Italian"},
            ),
            AIResponse(content="The top pick nearby is Le Bouillon, rated 4.7."),
        ]
    )
    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        tool_handler=run_tool,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv, room_id="weather-room")

    # --- Room 1: Weather assistant with tools ---
    print("=== Room 1: Weather Assistant ===")
    await kit.create_room(room_id="weather-room")
    await kit.attach_channel("weather-room", "ws-user")
    await kit.attach_channel(
        "weather-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            # Per-room AI configuration
            "system_prompt": (
                "You are a weather assistant. Use the get_weather tool to check conditions."
            ),
            "temperature": 0.3,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            },
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["city"],
                    },
                },
            ],
        },
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What's the weather in Montreal?"),
        )
    )

    print("  User asked about weather")
    for ev in inbox:
        # The tool call itself reaches the room too, as ToolCallContent.
        if ev.source.channel_id == "ai-assistant" and isinstance(ev.content, TextContent):
            print(f"  AI replied: {ev.content.body}")

    # --- Verify per-room config was applied ---
    # Check that the AI context was built with per-room settings
    if provider.calls:
        last_call = provider.calls[-1]
        print("\n  AI Context:")
        print(f"    System prompt: {(last_call.system_prompt or '')[:60]}...")
        print(f"    Temperature: {last_call.temperature}")
        print(f"    Tools: {[t.name for t in last_call.tools]}")

    # --- Room 2: Restaurant finder (different per-room config) ---
    print("\n=== Room 2: Restaurant Finder ===")
    inbox.clear()

    # Detach from previous room first
    await kit.detach_channel("weather-room", "ws-user")
    await kit.detach_channel("weather-room", "ai-assistant")

    await kit.create_room(room_id="restaurant-room")
    await kit.attach_channel("restaurant-room", "ws-user")
    # The same socket now follows a second conversation.
    ws.subscribe("user-conn", "restaurant-room")
    await kit.attach_channel(
        "restaurant-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "system_prompt": (
                "You are a restaurant finder. Help users discover great places to eat."
            ),
            "temperature": 0.9,
            "tools": [
                {
                    "name": "search_restaurants",
                    "description": "Search for restaurants near a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "cuisine": {"type": "string"},
                            "max_results": {"type": "integer", "default": 5},
                        },
                        "required": ["location"],
                    },
                },
                {
                    "name": "get_restaurant_details",
                    "description": "Get details about a specific restaurant",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant_id": {"type": "string"},
                        },
                        "required": ["restaurant_id"],
                    },
                },
            ],
        },
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Find Italian restaurants near downtown Montreal"),
        )
    )

    print("  User asked about restaurants")
    for ev in inbox:
        # The tool call itself reaches the room too, as ToolCallContent.
        if ev.source.channel_id == "ai-assistant" and isinstance(ev.content, TextContent):
            print(f"  AI replied: {ev.content.body}")

    if len(provider.calls) > 2:
        last_call = provider.calls[-1]
        print("\n  AI Context:")
        print(f"    System prompt: {(last_call.system_prompt or '')[:60]}...")
        print(f"    Temperature: {last_call.temperature}")
        print(f"    Tools: {[t.name for t in last_call.tools]}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
