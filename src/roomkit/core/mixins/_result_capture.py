"""Room-scoped capture of a delegated agent's result-tool call.

A delegated agent is one channel object serving every room it is attached to,
so the tool handler a delegation installs on it is shared by all of them. Two
delegations to the same agent in two rooms at once (two supervisor reviews,
say) must neither read each other's result nor leave a handler behind them.
One capture state per channel holds the channel's own handler and a result
slot per child room; the last delegation out restores the handler. The result
tool itself is declared in the child room's turns only (RFC §19.7): the
agent's other rooms, a customer's among them, never see it.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.tools.context import current_tool_call

if TYPE_CHECKING:
    from roomkit.orchestration.result import ResultTool


@dataclass
class ResultSlot:
    """Where one child room's result-tool call lands."""

    tool: ResultTool
    payload: dict[str, Any] | None = None


@dataclass
class _ChannelCapture:
    original_handler: Any
    slots: dict[str, ResultSlot] = field(default_factory=dict)

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """The channel's tool handler while a delegation captures on it."""
        slot = self._slot_for(name)
        if slot is not None:
            slot.payload = slot.tool.normalize(arguments or {})
            return json.dumps({"status": "received"})
        if self.original_handler:
            return await self.original_handler(name, arguments)
        return json.dumps({"error": f"unknown tool {name}"})

    def _slot_for(self, name: str) -> ResultSlot | None:
        """The slot of the room this call runs in, when it expects this tool.

        A call made through a tool loop carries its room; one made outside any
        loop does not, and is claimed only when a single slot expects the tool.
        """
        call = current_tool_call()
        if call is not None and call.room_id:
            slot = self.slots.get(call.room_id)
            return slot if slot is not None and slot.tool.name == name else None
        candidates = [slot for slot in self.slots.values() if slot.tool.name == name]
        return candidates[0] if len(candidates) == 1 else None


# Keyed by the channel object's id: an entry lives only while a delegation
# captures on that channel, which keeps the object alive.
_CAPTURES: dict[int, _ChannelCapture] = {}


@contextlib.contextmanager
def capture_result(channel: Any, child_room_id: str, tool: ResultTool) -> Iterator[ResultSlot]:
    """Capture *tool*'s call made in *child_room_id* by the agent behind *channel*.

    The tool is declared in *child_room_id*'s turns for as long as the capture
    runs; the channel's own handler is restored when the last capture on it
    ends.
    """
    state = _CAPTURES.get(id(channel))
    if state is None:
        state = _ChannelCapture(original_handler=channel.tool_handler)
        _CAPTURES[id(channel)] = state
        channel.tool_handler = state.dispatch
    slot = ResultSlot(tool)
    state.slots[child_room_id] = slot
    room_tools = channel._room_tools.setdefault(child_room_id, [])
    room_tools.append(tool.tool)
    try:
        yield slot
    finally:
        state.slots.pop(child_room_id, None)
        with contextlib.suppress(ValueError):
            room_tools.remove(tool.tool)
        if not room_tools:
            channel._room_tools.pop(child_room_id, None)
        if not state.slots:
            channel.tool_handler = state.original_handler
            del _CAPTURES[id(channel)]
