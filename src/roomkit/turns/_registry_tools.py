"""The tools a kit's channels carry from orchestration, served by the kit
(RFC §10.6.5).

The room owner reads them in the registry of the channel it holds, for each
turn it submits, and runs a call's handler there under the call's context.
Until the owner keeps them beside the channel's declaration (RMK-813), the
registry of the live channel is where they are kept.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.tools.context import ToolCallContext, owner_served_call
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

    async def serve(
        self, request: TurnRequest, call: ToolCall
    ) -> tuple[Any, dict[str, Any] | None]:
        context = call.context
        if (context.room_id, context.channel_id) != (request.room_id, request.channel_id):
            # The room and the channel are the turn's, as the owner submitted
            # it: a call never acts for another.
            raise UnservedToolCallError(
                f"A call of the turn in room {request.room_id!r} named room "
                f"{context.room_id!r} and channel {context.channel_id!r}"
            )
        registry = self._registry(request.channel_id)
        entry = registry.owner_lookup(call.name, request.room_id) if registry else None
        if entry is None or entry.serve is None:
            raise UnservedToolCallError(f"No tool {call.name!r} is served by the room owner")
        record = ToolCallContext(
            room_id=request.room_id,
            tool_call_id=context.tool_call_id,
            channel_id=request.channel_id,
        )
        room = request.context.room
        with owner_served_call(
            record,
            room=room if room.id == request.room_id else None,
            actor_id=context.actor_id,
            requester=context.requester,
            chain_depth=context.chain_depth,
            allowed_names=context.allowed_names,
        ):
            answer = entry.serve(call.arguments)
            if inspect.isawaitable(answer):
                answer = await answer
        return answer, record.structured_content

    def _registry(self, channel_id: str) -> ChannelRegistry | None:
        registry = getattr(self._channel_of(channel_id), "_registry", None)
        return registry if isinstance(registry, ChannelRegistry) else None
