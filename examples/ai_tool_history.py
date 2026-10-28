"""An AI channel's earlier tool calls, read back on the next turn.

The first turn books a room through a tool; its answer does not repeat the
booking number. The second turn asks for that number: the model reads it from
the earlier call's result, which the history replays at its place, as the
assistant's call then the tool's result. Shows:
- the history a later turn sends: the question, the call, its result, the
  answer, then the new question;
- MockAIProvider for running without an API key (it scripts what a real model
  would decide); ``provider.calls`` is what the model was given.

Run with:
    uv run python examples/ai_tool_history.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import (
    AIChannel,
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import (
    AIResponse,
    AITool,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    MockAIProvider,
)

BOOK = AITool(
    name="book_meeting_room",
    description="Book a meeting room",
    parameters={
        "type": "object",
        "properties": {
            "room": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "start": {"type": "string", "description": "HH:MM"},
            "minutes": {"type": "integer"},
        },
        "required": ["room", "date", "start", "minutes"],
    },
)


async def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Book the room; the booking number is only in this result."""
    return json.dumps({"booked": True, "booking_id": "BK-4821", **arguments, "end": "15:30"})


def shown(message: Any) -> str:
    """One message of what the model read, on one line."""
    if isinstance(message.content, str):
        return message.content
    parts = []
    for part in message.content:
        if isinstance(part, AIToolCallPart):
            parts.append(f"call {part.name}({part.arguments})")
        elif isinstance(part, AIToolResultPart):
            parts.append(f"result {part.result}")
    return " | ".join(parts)


async def main() -> None:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="call-1",
                        name="book_meeting_room",
                        arguments={
                            "room": "Quebec",
                            "date": "2026-11-03",
                            "start": "14:00",
                            "minutes": 90,
                        },
                    )
                ],
            ),
            AIResponse(content="The Quebec room is booked for November 3 from 14:00."),
            AIResponse(content="Your booking number is BK-4821; it ends at 15:30."),
        ]
    )
    kit = RoomKit()
    user = WebSocketChannel("user")
    kit.register_channel(user)
    kit.register_channel(
        AIChannel("assistant", provider=provider, tools=[BOOK], tool_handler=run_tool)
    )

    async def on_event(_conn: str, event: RoomEvent) -> None:
        if event.source.channel_id == "assistant" and isinstance(event.content, TextContent):
            print(f"Assistant: {event.content.body}")

    user.register_connection("me", on_event, room_id="office")
    await kit.create_room(room_id="office")
    await kit.attach_channel("office", "user")
    await kit.attach_channel("office", "assistant", category=ChannelCategory.INTELLIGENCE)

    for text in (
        "Book the Quebec room on 2026-11-03 at 14:00 for 90 minutes.",
        "What is my booking number, and when does it end?",
    ):
        print(f"\nYou: {text}")
        await kit.process_inbound(
            InboundMessage(channel_id="user", sender_id="me", content=TextContent(body=text))
        )

    print("\nWhat the model read on the second turn:")
    for message in provider.calls[-1].messages:
        print(f"  [{message.role}] {shown(message)}")
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
