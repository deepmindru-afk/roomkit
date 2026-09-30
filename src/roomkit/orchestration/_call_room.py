"""An orchestration tool acts on the room of its call (RFC §19.6, §23.4).

One agent serves every room it is attached to, so a handoff or delegation
tool cannot know its room from how it was wired: a room captured then names
whichever room came first. It reads the room of the call from the tool call
context (RFC §21.4), which the AI channel's tool loops and the realtime
channel install around each call, and refuses a call made outside one.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Collection
from typing import Any

from roomkit.tools.context import current_tool_room_id

NO_CALL_ROOM = json.dumps(
    {"error": "This tool acts on the room of a tool call, and was called outside one"}
)
"""What an orchestration tool answers when no tool call names its room."""

RoomToolServe = Callable[[str, str, dict[str, Any]], Awaitable[Any]]
"""Serves one call: ``(room_id, name, arguments)`` to the result."""

ToolHandlerFn = Callable[[str, dict[str, Any]], Awaitable[Any]]


def in_call_room(name: str, serve: RoomToolServe) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """What serves one call to *name*, in the room of the call: the server a
    channel's registry entry for *name* carries."""

    async def run(arguments: dict[str, Any]) -> Any:
        room_id = current_tool_room_id()
        if room_id is None:
            return NO_CALL_ROOM
        return await serve(room_id, name, arguments)

    return run


def call_room_handler(
    names: Collection[str], serve: RoomToolServe, fallback: ToolHandlerFn | None
) -> ToolHandlerFn:
    """A tool handler serving *names* in the room of each call, the other
    tools by *fallback* (the handler it wraps), or as unknown without one."""

    async def handler(name: str, arguments: dict[str, Any]) -> Any:
        if name not in names:
            if fallback is not None:
                return await fallback(name, arguments)
            return json.dumps({"error": f"Unknown tool: {name}"})
        room_id = current_tool_room_id()
        if room_id is None:
            return NO_CALL_ROOM
        return await serve(room_id, name, arguments)

    return handler
