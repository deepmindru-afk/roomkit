"""The open human-input requests of a room owner's turns (RFC §10.6.5).

A request is answerable from its creation: an answer or a rejection for an id
the owner does not know yet is held until its request arrives, or for the
hold bound (``input_hold``, 60 seconds by default), and a request answered
once is not answered again. The outcome goes to the turn's executor as an
``input_answer``; a request whose turn ends is withdrawn.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.turns.wire import InputAnswer

if TYPE_CHECKING:
    from roomkit.turns._owner_turn import OwnerTurn
    from roomkit.turns.wire import InputRequest

SETTLED_KEPT = 1024
"""How many settled request ids the owner remembers, refusing a second answer."""


@dataclass(slots=True)
class _Held:
    answer: InputAnswer
    room_id: str | None
    until: float


class OwnerInputs:
    """The open requests of a client's turns, and the answers held for ones
    not open yet."""

    def __init__(self, hold: float) -> None:
        self._hold = hold
        self._open: dict[str, OwnerTurn] = {}
        self._held: dict[str, _Held] = {}
        self._settled: OrderedDict[str, None] = OrderedDict()
        self._sending: set[asyncio.Task[Any]] = set()

    def opened(self, turn: OwnerTurn, request: InputRequest) -> None:
        """*turn* opened a request: an answer held for it is sent now."""
        self._prune()
        held = self._held.pop(request.request_id, None)
        if held is not None and held.room_id in (None, turn.request.room_id):
            self._deliver(turn, held.answer)
            return
        self._open[request.request_id] = turn

    def closed(self, request_id: str) -> None:
        """The request settled on its executor without an answer from here."""
        if self._open.pop(request_id, None) is not None:
            self._remember(request_id)

    def turn_ended(self, turn: OwnerTurn) -> None:
        """*turn* is over: its open requests are withdrawn."""
        for request_id in [rid for rid, open_turn in self._open.items() if open_turn is turn]:
            del self._open[request_id]
            self._remember(request_id)

    async def answer(self, request_id: str, value: str, *, room_id: str | None = None) -> bool:
        """Record *value* as the answer to *request_id*: whether it was taken."""
        return self._record(
            InputAnswer(request_id=request_id, outcome="answered", value=value), room_id
        )

    async def reject(self, request_id: str, reason: str, *, room_id: str | None = None) -> bool:
        """Record a rejection of *request_id*: whether it was taken."""
        return self._record(
            InputAnswer(request_id=request_id, outcome="rejected", reason=reason), room_id
        )

    def _record(self, answer: InputAnswer, room_id: str | None) -> bool:
        """Send *answer* to the request's turn, or hold it until the request
        arrives; ``False`` for a request already settled, or of another room
        than *room_id* when it names one."""
        self._prune()
        request_id = answer.request_id
        if request_id in self._settled or request_id in self._held:
            return False
        turn = self._open.get(request_id)
        if turn is None:
            until = asyncio.get_running_loop().time() + self._hold
            self._held[request_id] = _Held(answer, room_id, until)
            return True
        if room_id is not None and turn.request.room_id != room_id:
            return False
        del self._open[request_id]
        self._deliver(turn, answer)
        return True

    def _deliver(self, turn: OwnerTurn, answer: InputAnswer) -> None:
        self._remember(answer.request_id)
        task = asyncio.get_running_loop().create_task(turn.send_control(answer))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)

    def _remember(self, request_id: str) -> None:
        self._settled[request_id] = None
        while len(self._settled) > SETTLED_KEPT:
            self._settled.popitem(last=False)

    def _prune(self) -> None:
        """Let go of the answers held past their bound."""
        now = asyncio.get_running_loop().time()
        for request_id in [rid for rid, held in self._held.items() if held.until <= now]:
            del self._held[request_id]
