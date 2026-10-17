"""The tools a kit's channels carry from orchestration, served by the kit
(RFC §10.6.5).

The room owner reads them in the registry of the channel it holds, for each
turn it submits, and runs a call's handler there (under the call's context,
which the owner's calls install).
Until the owner keeps them beside the channel's declaration (RMK-813), the
registry of the live channel is where they are kept.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.turns.base import OwnerTools
from roomkit.turns.wire import OwnerTool

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.channels.base import Channel
    from roomkit.turns.wire import ToolCall, TurnRequest


class RegistryOwnerTools(OwnerTools):
    """The owner's tools of the channels a registry holds (a kit's, a router's)."""

    def __init__(self, channel_of: Callable[[str], Channel | None]) -> None:
        self._channel_of = channel_of

    def declared(self, channel_id: str, room_id: str) -> list[OwnerTool]:
        registry = self._registry(channel_id)
        if registry is None:
            return []
        return [
            OwnerTool(tool=entry.definition, waits_by_design=entry.traits.waits)
            for entry in registry.owner_entries(room_id)
        ]

    async def serve(self, request: TurnRequest, call: ToolCall) -> Any:
        registry = self._registry(request.channel_id)
        entry = registry.owner_lookup(call.name, request.room_id) if registry else None
        if entry is None or entry.serve is None:
            raise UnservedToolCallError(f"No tool {call.name!r} is served by the room owner")
        answer = entry.serve(call.arguments)
        if inspect.isawaitable(answer):
            answer = await answer
        return answer

    def _registry(self, channel_id: str) -> ChannelRegistry | None:
        registry = getattr(self._channel_of(channel_id), "_registry", None)
        return registry if isinstance(registry, ChannelRegistry) else None
