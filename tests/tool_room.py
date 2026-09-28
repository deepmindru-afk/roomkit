"""Call a tool handler directly as if the tool loop of a room had called it.

A handler reads the room of its call from the tool call context (RFC §21.4);
a test that calls one directly, outside any loop, sets that context itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext


@contextmanager
def tool_call_in(room_id: str) -> Iterator[None]:
    """Run the enclosed handler calls as calls of *room_id*'s tool loop."""
    token = _current_loop_ctx.set(_ToolLoopContext(room_id=room_id))
    try:
        yield
    finally:
        _current_loop_ctx.reset(token)
