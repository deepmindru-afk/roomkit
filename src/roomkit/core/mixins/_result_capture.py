"""Room-scoped capture of a delegated agent's result-tool call.

A delegated agent is one channel object serving every room it is attached to.
The result tool a delegation expects is set up for the child room only (RFC
§19.7): declared in that room's turns and served there, the agent's other rooms,
a customer's among them, never see it. Two delegations to the same agent in two
rooms at once (two supervisor reviews, say) each capture their own room's call,
and each withdraws its own tool when it ends; nothing is swapped on the shared
channel.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import orchestration_tool

if TYPE_CHECKING:
    from roomkit.orchestration.result import ResultTool


@dataclass
class ResultSlot:
    """Where one child room's result-tool call lands."""

    tool: ResultTool
    payload: dict[str, Any] | None = None

    async def receive(self, arguments: dict[str, Any]) -> str:
        """Keep the payload of the result tool's call."""
        self.payload = self.tool.normalize(arguments or {})
        return json.dumps({"status": "received"})


@contextlib.contextmanager
def capture_result(channel: Any, child_room_id: str, tool: ResultTool) -> Iterator[ResultSlot]:
    """Capture *tool*'s call made in *child_room_id* by the agent behind *channel*.

    The tool is declared and served in *child_room_id*'s turns for as long as
    the capture runs.
    """
    slot = ResultSlot(tool)
    registry = channel._registry
    entry = orchestration_tool(tool.tool, slot.receive)
    registry.register(entry, room_id=child_room_id, owner=slot)
    try:
        yield slot
    finally:
        registry.unregister(tool.tool.name, room_id=child_room_id, owner=slot)
