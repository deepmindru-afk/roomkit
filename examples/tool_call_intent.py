"""A tool call's intent: the sentence a model writes with a call, for the person.

With ``tool_call_intents=True``, the agent shows its model each tool's schema
with one optional property more, ``_roomkit_intent``. The model fills it with a
short sentence ("Je recherche les fichiers texte du projet."); RoomKit takes it
out of the call's arguments before the tool sees them and carries it apart, as
the call's ``intent``. Shows:
- the schema the model is shown, beside the tool's own;
- the arguments the tool receives: its own only, the schema being closed;
- the intent on BEFORE_TOOL_USE, on the TOOL_CALL_START and TOOL_CALL_END
  ephemeral events a live UI reads, and on the stored rows a replay reads;
- MockAIProvider for running without an API key (it scripts what a real model
  would write).

Run with:
    uv run python examples/tool_call_intent.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    ToolCallContent,
    ToolCallEvent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.realtime import EphemeralEvent, EphemeralEventType

SHELL = AITool(
    name="shell",
    description="Run a shell command in the project directory.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
)


async def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """The tool: it receives the command, never the intent."""
    print(f"Tool {name} receives: {json.dumps(arguments)}")
    return "notes.txt\ndocs/todo.txt"


async def main() -> None:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="call-1",
                        name="shell",
                        arguments={
                            "command": "find . -name '*.txt'",
                            "_roomkit_intent": "Je recherche les fichiers texte du projet.",
                        },
                    )
                ],
            ),
            AIResponse(content="Deux fichiers texte : notes.txt et docs/todo.txt."),
        ]
    )
    kit = RoomKit()
    user = WebSocketChannel("user")
    kit.register_channel(user)
    kit.register_channel(
        AIChannel(
            "assistant",
            provider=provider,
            tools=[SHELL],
            tool_handler=run_tool,
            tool_call_intents=True,
        )
    )

    async def on_event(_conn: str, event: RoomEvent) -> None:
        if event.source.channel_id == "assistant" and isinstance(event.content, TextContent):
            print(f"Assistant: {event.content.body}")

    user.register_connection("me", on_event, room_id="project")
    await kit.create_room(room_id="project")
    await kit.attach_channel("project", "user")
    await kit.attach_channel("project", "assistant", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="approve")
    async def approve(event: ToolCallEvent, _ctx: Any) -> HookResult:
        print(f"BEFORE_TOOL_USE: {event.name} {json.dumps(event.arguments)}")
        print(f"  intent: {event.intent!r}")
        return HookResult.allow()

    async def live_ui(event: EphemeralEvent) -> None:
        if event.type in (EphemeralEventType.TOOL_CALL_START, EphemeralEventType.TOOL_CALL_END):
            for call in event.data["tool_calls"]:
                print(f"{event.type.value}: {call['name']} intent={call.get('intent')!r}")

    await kit.realtime.subscribe_to_room("project", live_ui)

    await kit.process_inbound(
        InboundMessage(
            channel_id="user",
            sender_id="me",
            content=TextContent(body="Trouve les fichiers texte du projet."),
        )
    )

    print("\nThe tool's own schema:", json.dumps(SHELL.parameters))
    # provider.calls holds what each request showed the model.
    print("The schema the model is shown:", json.dumps(provider.calls[0].tools[0].parameters))
    print("\nStored rows:")
    for row in await kit.store.list_events("project"):
        if isinstance(row.content, ToolCallContent):
            content = row.content
            print(
                f"  {row.type.value}: arguments={json.dumps(content.arguments)} "
                f"intent={content.intent!r}"
            )
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
