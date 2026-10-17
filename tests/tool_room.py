"""Call a tool handler directly as if the tool loop of a room had called it.

A handler reads the room of its call from the tool call context (RFC §21.4);
a test that calls one directly, outside any loop, sets that context itself.

The tools orchestration sets up are the room owner's, kept in its store
beside the channel's declaration (RFC §10.6.2): an executor gives a turn the
ones its request lists. A test that runs a channel's turn or calls its tool
handler directly, outside the turn backend, gives them itself
(:func:`owner_tools_in`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from roomkit.channels._tool_registry import _TURN_OWNER_TOOLS, ToolSource, TurnOwnerTools
from roomkit.providers.ai.base import AITool
from roomkit.tools import tool_turn_context


@contextmanager
def tool_call_in(room_id: str, *, chain_depth: int = 0) -> Iterator[None]:
    """Run the enclosed handler calls as calls of *room_id*'s tool loop.

    *chain_depth* is the depth of the response the calling turn produces.
    """
    with tool_turn_context(room_id=room_id, chain_depth=chain_depth):
        yield


@contextmanager
def owner_tools_in(kit: Any, channel_id: str, room_id: str) -> Iterator[None]:
    """Give the enclosed turn of *channel_id* in *room_id*, or its tool
    handler's calls, the tools *kit* serves it, as an executor gives a turn
    the ones its request lists (RFC §10.6.5)."""
    registry = kit._owner_tools.get(channel_id)
    entries = {e.name: e for e in registry.owner_entries(room_id)} if registry else {}
    token = _TURN_OWNER_TOOLS.set(
        TurnOwnerTools(channel_id=channel_id, room_id=room_id, entries=entries)
    )
    try:
        yield
    finally:
        _TURN_OWNER_TOOLS.reset(token)


@contextmanager
def owner_tool_call_in(kit: Any, channel_id: str, room_id: str) -> Iterator[None]:
    """:func:`tool_call_in` *room_id*, *channel_id*'s owner tools given."""
    with tool_call_in(room_id), owner_tools_in(kit, channel_id, room_id):
        yield


def owner_tool_definitions(kit: Any, channel_id: str, room_id: str | None = None) -> list[AITool]:
    """The definitions of the tools orchestration set up for *channel_id*,
    kept by *kit*: those of every room, then *room_id*'s."""
    registry = kit._owner_tools.get(channel_id)
    if registry is None:
        return []
    entries = registry.entries(room_id, source=ToolSource.ORCHESTRATION)
    return [entry.definition for entry in entries]


def room_tool_names(kit: Any, channel_id: str, room_id: str) -> list[str]:
    """The tools orchestration set up for *channel_id* that *room_id*'s turns
    declare, kept by *kit*."""
    registry = kit._owner_tools.get(channel_id)
    if registry is None:
        return []
    entries = registry.entries(room_id, source=ToolSource.ORCHESTRATION)
    return [entry.name for entry in entries]
