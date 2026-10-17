"""The room owner's side of the turn backend (RFC §10.6).

:class:`TurnClient` is how the pipeline runs a turn: it builds the request,
submits it, and reads the turn back as the channel's ``ChannelOutput``. The
kit's own executor is served on first use.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING
from uuid import uuid4

from roomkit.channels._discussion_turn import DISCUSSION_TURN, discussion_notes, discussion_turn
from roomkit.models.steering import Cancel as SteeringCancel
from roomkit.turns._owner_inputs import OwnerInputs
from roomkit.turns._owner_turn import OwnerTurn
from roomkit.turns._registry_tools import RegistryOwnerTools
from roomkit.turns._remote_stream import RemoteStream
from roomkit.turns._rooms import RoomRecords
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.executor import TurnExecutor
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.turns.wire import PreviousTurn, RoomDetached, TurnRequest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from roomkit.channels.base import Channel
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.steering import SteeringDirective
    from roomkit.turns.base import OwnerHost, OwnerTools, TurnBackend
    from roomkit.turns.wire import ChannelControl, ChannelResult, OwnerTool


def turn_request(
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    *,
    answers: bool = True,
    strategy_turn: bool = False,
    owner_tools: list[OwnerTool] | None = None,
    previous_turn: PreviousTurn | None = None,
) -> TurnRequest:
    """The request for one turn of *binding*'s channel on *event*, with the
    tools the owner serves it (*owner_tools*) and the room's previous turn
    of the channel (*previous_turn*).

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
        strategy_turn=strategy_turn,
        notes=notes,
        in_place=in_place,
        owner_tools=owner_tools or [],
        previous_turn=previous_turn,
    )


class TurnClient:
    """Runs turns through a backend and reads them back (RFC §10.6).

    Args:
        backend: The turn backend.
        executor: An executor this client serves on the backend before its
            first turn (the kit's own), or ``None``.
        timeouts: The bounds the owner keeps.
        tools: The tools the owner serves its turns (RFC §10.6.5), or ``None``.
        host: What the owner runs for its turns' ``TurnHost`` (RFC §10.6.5),
            or ``None``.
    """

    def __init__(
        self,
        backend: TurnBackend,
        *,
        executor: TurnExecutor | None = None,
        timeouts: TurnTimeouts | None = None,
        tools: OwnerTools | None = None,
        host: OwnerHost | None = None,
    ) -> None:
        self.backend = backend
        self.executor = executor
        self._timeouts = timeouts or TurnTimeouts()
        self._tools = tools
        self._host = host
        self._serving = executor is None
        self._open: weakref.WeakSet[OwnerTurn] = weakref.WeakSet()
        # The turns this client started and that still run, in start order:
        # what steering resolves its targets from (RFC §10.6.8).
        self._started: list[OwnerTurn] = []
        # The owner-served calls running for this client's turns (RFC §10.6.5).
        self._calls: set[asyncio.Task[None]] = set()
        # The open human-input requests of its turns, answered here.
        self.inputs = OwnerInputs(self._timeouts.input_hold_timeout)
        # Each room's last turn of a channel submitted here (RFC §10.6.2).
        self._previous: RoomRecords[PreviousTurn] = RoomRecords()

    async def run(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        *,
        answers: bool = True,
        strategy_turn: bool = False,
    ) -> ChannelOutput:
        """Run one turn of *binding*'s channel on *event*: its output, its
        stream reading the executor's (RFC §10.6.4); a strategy's generation
        when *strategy_turn* (RFC §10.6.1)."""
        await self._serve_own()
        channel_id = binding.channel_id
        declared = self._tools.declared(channel_id, event.room_id) if self._tools else []
        previous = self._previous.get(channel_id, event.room_id)
        request = turn_request(
            event,
            binding,
            context,
            answers=answers,
            strategy_turn=strategy_turn,
            owner_tools=declared,
            previous_turn=previous.model_copy() if previous is not None else None,
        )
        link = await self.backend.submit(request)
        submitted = PreviousTurn(turn_id=request.turn_id)
        self._previous.put(channel_id, request.room_id, submitted)
        turn = OwnerTurn(link, self._timeouts, self._tools, self._calls, self._host, self.inputs)
        self._open.add(turn)
        self._started = [*(t for t in self._started if not t.over), turn]
        try:
            return await turn.output()
        finally:
            # Known once the executor took the turn, which its output follows.
            submitted.executor_id = turn.executor_id

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        """Send a channel control message (RFC §10.6.3): a ``channel_call``'s
        result; nothing for ``room_attached`` and ``room_detached``. After a
        ``room_detached``, the room's next turn of the channel names no
        previous turn, so an executor that missed the message still drops
        the room's state (RFC §10.6.3)."""
        await self._serve_own()
        if isinstance(message, RoomDetached):
            self._previous.pop(message.channel_id, message.room_id)
        return await self.backend.control(message)

    async def _serve_own(self) -> None:
        """Serve the client's own executor on the backend, once."""
        if not self._serving:
            self._serving = True
            assert self.executor is not None
            await self.backend.serve(self.executor)

    async def steer(
        self,
        directive: SteeringDirective,
        channel_id: str,
        *,
        room_id: str | None = None,
        turn_id: str | None = None,
    ) -> int:
        """Steer the running turns of *channel_id* this client started (RFC
        §21.3, §10.6.8): the turn *turn_id*; else the turns of *room_id*, all
        of them for a ``Cancel`` and the most recent for any other directive;
        else the most recent turn of the channel. How many loops it reached.

        Raises:
            ValueError: both *room_id* and *turn_id* given.
        """
        if room_id is not None and turn_id is not None:
            raise ValueError("steer() addresses a room_id or a turn_id, not both")
        running = [t for t in self._started if not t.over and t.request.channel_id == channel_id]
        if turn_id is not None:
            targets = [t for t in running if t.request.turn_id == turn_id]
        elif room_id is not None:
            in_room = [t for t in running if t.request.room_id == room_id]
            targets = in_room if isinstance(directive, SteeringCancel) else in_room[-1:]
        else:
            targets = running[-1:]
        reached = await asyncio.gather(*(turn.steer(directive) for turn in targets))
        return sum(reached)

    async def close(self) -> None:
        """Close the own executor, which cuts its turns, stop holding every
        turn still open (a stream nobody read), wait for the calls its turns
        made to end (but the one running this close), then close the backend."""
        if self.executor is not None:
            await self.executor.close()
        for turn in list(self._open):
            turn.abandon()
        current = asyncio.current_task()
        calls = [task for task in self._calls if task is not current]
        for task in calls:
            if not task.cancelling():
                # One already cancelled is let finish its cleanup.
                task.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
        self._started.clear()
        await self.backend.close()


def drop_unread(stream: object) -> None:
    """Give up a turn's stream nobody will read: a stream of the turn
    backend is closed in the background; a plain generator is left to its
    collection, which closes it."""
    if isinstance(stream, RemoteStream):
        stream.drop()


def registry_turn_client(
    channels: Mapping[str, Channel],
    backend: TurnBackend | None = None,
    *,
    host: OwnerHost | None = None,
) -> TurnClient:
    """A client whose own executor serves the channels of a registry (a
    kit's, a router's), on *backend* or an in-memory one, its turns' host
    verbs run by *host*."""
    return TurnClient(
        backend or InMemoryTurnBackend(),
        executor=TurnExecutor(channels.get, hosts=channels.__contains__),
        tools=RegistryOwnerTools(channels.get),
        host=host,
    )
