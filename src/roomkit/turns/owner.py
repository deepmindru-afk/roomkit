"""The room owner's side of the turn backend (RFC §10.6).

:class:`TurnClient` is how the pipeline runs a turn: it builds the request,
submits it, and reads the turn back as the channel's ``ChannelOutput``. The
kit's own executor is served on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from roomkit.channels._discussion_turn import DISCUSSION_TURN, discussion_notes, discussion_turn
from roomkit.turns._owner_turn import OwnerTurn
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.wire import TurnRequest

if TYPE_CHECKING:
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.turns.base import TurnBackend
    from roomkit.turns.executor import TurnExecutor


def turn_request(
    event: RoomEvent, binding: ChannelBinding, context: RoomContext, *, answers: bool = True
) -> TurnRequest:
    """The request for one turn of *binding*'s channel on *event*.

    A discussion's turn mark is this process's: its notes travel in the
    request, and the event goes without it (RFC §10.6.3).
    """
    in_place = discussion_turn(event) is not None
    notes = discussion_notes(event) if in_place else []
    if DISCUSSION_TURN in (event.metadata or {}):
        metadata = {k: v for k, v in event.metadata.items() if k != DISCUSSION_TURN}
        event = event.model_copy(update={"metadata": metadata})
    return TurnRequest(
        turn_id=f"turn-{uuid4().hex}",
        room_id=event.room_id,
        channel_id=binding.channel_id,
        event=event,
        binding=binding,
        context=context,
        answers=answers,
        notes=notes,
        in_place=in_place,
    )


class TurnClient:
    """Runs turns through a backend and reads them back (RFC §10.6).

    Args:
        backend: The turn backend.
        executor: An executor this client serves on the backend before its
            first turn (the kit's own), or ``None``.
        timeouts: The bounds the owner keeps.
    """

    def __init__(
        self,
        backend: TurnBackend,
        *,
        executor: TurnExecutor | None = None,
        timeouts: TurnTimeouts | None = None,
    ) -> None:
        self.backend = backend
        self.executor = executor
        self._timeouts = timeouts or TurnTimeouts()
        self._serving = executor is None

    async def run(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        *,
        answers: bool = True,
    ) -> ChannelOutput:
        """Run one turn of *binding*'s channel on *event*: its output, its
        stream reading the executor's (RFC §10.6.4)."""
        if not self._serving:
            self._serving = True
            assert self.executor is not None
            await self.backend.serve(self.executor)
        link = await self.backend.submit(turn_request(event, binding, context, answers=answers))
        return await OwnerTurn(link, self._timeouts).output()

    async def close(self) -> None:
        """Cut the turns in flight and close the own executor, then the backend."""
        if self.executor is not None:
            await self.executor.close()
        await self.backend.close()
