"""Call a tool handler directly as if the tool loop of a room had called it.

A handler reads the room of its call from the tool call context (RFC §21.4);
a test that calls one directly, outside any loop, sets that context itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from roomkit.channels._tool_registry import ToolSource
from roomkit.tools import tool_turn_context


@contextmanager
def tool_call_in(room_id: str, *, chain_depth: int = 0) -> Iterator[None]:
    """Run the enclosed handler calls as calls of *room_id*'s tool loop.

    *chain_depth* is the depth of the response the calling turn produces.
    """
    with tool_turn_context(room_id=room_id, chain_depth=chain_depth):
        yield


def room_tool_names(channel: Any, room_id: str) -> list[str]:
    """The tools orchestration set up on *channel* that *room_id*'s turns declare."""
    entries = channel._registry.entries(room_id, source=ToolSource.ORCHESTRATION)
    return [entry.name for entry in entries]
