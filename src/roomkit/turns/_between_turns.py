"""What a turn's work reaches its room owner by once the turn ended (RFC
§10.6.3, §10.6.5).

Work a turn began can outlive it: a background thinker reports its thought
after its turn returned, a hook's task runs on. Its host and its outlet then
reach the owner outside the turn's link, through the port the backend gave
the executor: a host verb as a ``room_call``, at the turn's chain depth, and
a publication as a ``room_publication``, both naming the turn, which the
owner checks against the last turn it sent for the channel and room.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.turns._codec import rebuild_error
from roomkit.turns.wire import RoomCall, RoomPublication

if TYPE_CHECKING:
    from roomkit.turns.base import RoomOwner
    from roomkit.turns.wire import Ephemeral, FrameworkEventFrame, TurnRequest

logger = logging.getLogger("roomkit.turns")


class BetweenTurns:
    """One turn's way to its owner once the turn ended."""

    def __init__(self, owner: RoomOwner, request: TurnRequest, executor_id: str) -> None:
        self._owner = owner
        self._request = request
        self._executor_id = executor_id

    async def call(self, verb: str, arguments: dict[str, Any]) -> Any:
        """Send a host verb for the turn's room: its result.

        Raises:
            What the owner refused or the verb raised there.
        """
        request = self._request
        answer = await self._owner.room_call(
            RoomCall(
                channel_id=request.channel_id,
                room_id=request.room_id,
                executor_id=self._executor_id,
                turn_id=request.turn_id,
                # What the turn's response is at: the bound on agent loops.
                chain_depth=request.event.chain_depth + 1,
                verb=verb,
                arguments=arguments,
            )
        )
        if answer.error is not None:
            raise rebuild_error(answer.error, where="the room owner")
        return answer.result

    async def publish(self, frame: Ephemeral | FrameworkEventFrame) -> None:
        """Send what the channel published; never raises."""
        request = self._request
        message = RoomPublication(
            channel_id=request.channel_id,
            room_id=request.room_id,
            executor_id=self._executor_id,
            turn_id=request.turn_id,
            frame=frame,
        )
        try:
            await self._owner.room_publication(message)
        except Exception:
            logger.debug("A publication after turn %s did not reach its owner", request.turn_id)
