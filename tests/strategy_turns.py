"""An agent's turn as the room's owner runs it, for tests that drive a
strategy against a mocked kit (RFC §10.6.1)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from roomkit.channels._strategy_turn import strategy_turn
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent


async def owner_turn(
    kit: Any, agent: Any, event: RoomEvent, binding: ChannelBinding, context: RoomContext
) -> ChannelOutput:
    """*agent*'s turn: its strategy's runner when one takes the room's turns,
    each generation the agent's own answer (a strategy turn); the agent's
    ``on_event`` otherwise."""

    async def generation(
        event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        with strategy_turn(True):
            return await agent.on_event(event, binding, context)

    kit._strategy_turn = AsyncMock(side_effect=generation)
    room_id = context.room.id if context.room else event.room_id
    runner = kit._turn_runners.get(room_id, agent.channel_id)
    return await (runner or agent.on_event)(event, binding, context)
