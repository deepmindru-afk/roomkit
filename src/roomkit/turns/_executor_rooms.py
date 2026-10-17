"""The rooms an executor's channels hold state for (RFC §10.6.2), and the
channel control it applies to them (RFC §10.6.3).

A channel keeps state per room between two turns on its executor. The
executor records which turn left it, and drops it before a turn whose
previous one is another: the room's turns ran elsewhere since, and what they
made obsolete must not come back. Dropping is the channel's own
``on_room_detached``, which lets go of everything it holds for the room.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.turns._codec import error_record
from roomkit.turns._rooms import RoomRecords, stored_room_state
from roomkit.turns.wire import ChannelCall, ChannelResult, RoomAttached, RoomDetached

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.turns.wire import ChannelControl, TurnRequest

logger = logging.getLogger("roomkit.turns")


class ExecutorRooms:
    """Which turn left each room's state of a channel on one executor."""

    def __init__(self, executor_id: str) -> None:
        self._executor_id = executor_id
        self._left: RoomRecords[str] = RoomRecords()

    def enter(self, request: TurnRequest) -> bool:
        """Record *request*'s turn as the one leaving its room's state of the
        channel: whether the state held now was left by another turn than
        the request's previous one, and must be dropped before it runs."""
        previous = request.previous_turn.turn_id if request.previous_turn else None
        left = self._left.get(request.channel_id, request.room_id)
        self._left.put(request.channel_id, request.room_id, request.turn_id)
        return left != previous

    async def prepare(self, channel: Channel, request: TurnRequest, *, stale: bool) -> None:
        """Ready *channel* for *request*'s turn: what it holds for the room
        dropped when *stale*, then what the room keeps for it handed back."""
        if stale:
            await channel.on_room_detached(request.room_id)
        room = request.context.room
        stored = stored_room_state(room.metadata, request.channel_id) if room else {}
        channel.restore_room_state(request.room_id, stored)

    async def apply(
        self, channel: Channel | None, message: ChannelControl
    ) -> ChannelResult | None:
        """Apply one channel control message to *channel* (``None``: the
        factory has none): a ``channel_call``'s result; nothing for the
        others, whose failure is the channel's to log."""
        if isinstance(message, ChannelCall):
            return await self._call(channel, message)
        if isinstance(message, RoomDetached):
            self._left.pop(message.channel_id, message.room_id)
        if channel is None:
            return None
        try:
            if isinstance(message, RoomAttached):
                await channel.on_room_attached(message.room_id, message.binding)
            else:
                await channel.on_room_detached(message.room_id)
        except Exception:
            logger.exception(
                "Channel %s failed to apply %s for room %s",
                message.channel_id,
                message.type,
                message.room_id,
            )
        return None

    async def _call(self, channel: Channel | None, message: ChannelCall) -> ChannelResult:
        """Run a per-room verb of *channel*: its result, and what the room
        keeps for the channel when the verb changed it."""
        if channel is None:
            missing = ChannelNotRegisteredError(
                f"No channel {message.channel_id!r} on executor {self._executor_id}"
            )
            return ChannelResult(error=error_record(missing))
        channel.restore_room_state(message.room_id, message.room_state)
        try:
            result = await channel.channel_call(message.room_id, message.verb, message.arguments)
        except Exception as exc:
            return ChannelResult(error=error_record(exc), room_state=_changed(channel, message))
        answer = ChannelResult(result=result, room_state=_changed(channel, message))
        try:
            answer.model_dump_json()
        except (TypeError, ValueError) as exc:
            refusal = TypeError(
                f"The result of {message.verb!r} cannot cross the turn backend: {exc}"
            )
            return ChannelResult(error=error_record(refusal), room_state=answer.room_state)
        return answer


def _changed(channel: Channel, message: ChannelCall) -> dict[str, object] | None:
    """What the room keeps for the channel once *message* ran, when it is not
    what the owner sent."""
    state = channel.room_state(message.room_id)
    return state if state != message.room_state else None
