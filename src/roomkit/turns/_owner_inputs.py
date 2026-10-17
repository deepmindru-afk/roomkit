"""The open human-input requests of a room owner's turns (RFC §10.6.5).

A request is answerable from its creation: an answer or a rejection the kit
is given for an id it does not know yet is held until its request arrives, or
for the hold bound (``input_hold_timeout``, 60 seconds by default), at most
:data:`HELD_KEPT` at once; and a request answered once is not answered again.
The outcome goes to the turn's executor as an ``input_answer``; a request
whose turn ends is withdrawn.

A request belongs to the turn that opened it: an id already open, or already
settled, is not opened again, and only its turn closes it. The executor, where
the tool waits, settles the request first come first served (an answer its
channel's handler received, a hook's block, its bound, a release) and closes
it here; an answer recorded here goes to the executor, where it settles the
request if it is still open.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.turns.wire import InputAnswer

if TYPE_CHECKING:
    from roomkit.turns._owner_turn import OwnerTurn
    from roomkit.turns.wire import InputRequest

logger = logging.getLogger("roomkit.turns")

SETTLED_KEPT = 1024
"""How many settled request ids the owner remembers, refusing a second answer."""

HELD_KEPT = 1024
"""How many early answers the owner holds at once, the oldest let go first."""


@dataclass(slots=True)
class _Held:
    answer: InputAnswer
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
        request_id = request.request_id
        if request_id in self._open or request_id in self._settled:
            # Another turn's, or settled: a request is opened once.
            return
        held = self._held.pop(request_id, None)
        if held is not None:
            self._deliver(turn, held.answer)
            return
        self._open[request_id] = turn

    def closed(self, turn: OwnerTurn, request_id: str) -> None:
        """*turn*'s request settled on its executor without an answer from here."""
        if self._open.get(request_id) is turn:
            del self._open[request_id]
            self._remember(request_id)

    def turn_ended(self, turn: OwnerTurn) -> None:
        """*turn* is over: its open requests are withdrawn."""
        for request_id in [rid for rid, open_turn in self._open.items() if open_turn is turn]:
            del self._open[request_id]
            self._remember(request_id)

    async def answer(self, request_id: str, value: str) -> bool:
        """Record *value* as the answer to *request_id*: whether it was taken."""
        return self._record(InputAnswer(request_id=request_id, outcome="answered", value=value))

    async def reject(self, request_id: str, reason: str) -> bool:
        """Record a rejection of *request_id*: whether it was taken."""
        return self._record(InputAnswer(request_id=request_id, outcome="rejected", reason=reason))

    def _record(self, answer: InputAnswer) -> bool:
        """Send *answer* to the request's turn, or hold it until the request
        arrives; ``False`` for a request already settled or answered."""
        self._prune()
        request_id = answer.request_id
        if request_id in self._settled or request_id in self._held:
            return False
        turn = self._open.get(request_id)
        if turn is None:
            self._hold_early(answer)
            return True
        del self._open[request_id]
        self._deliver(turn, answer)
        return True

    def _hold_early(self, answer: InputAnswer) -> None:
        """Hold *answer* for its request, the oldest let go past the cap."""
        until = asyncio.get_running_loop().time() + self._hold
        self._held[answer.request_id] = _Held(answer, until)
        while len(self._held) > HELD_KEPT:
            del self._held[next(iter(self._held))]

    def _deliver(self, turn: OwnerTurn, answer: InputAnswer) -> None:
        self._remember(answer.request_id)
        task = asyncio.get_running_loop().create_task(_sent(turn, answer))
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


async def _sent(turn: OwnerTurn, answer: InputAnswer) -> None:
    try:
        await turn.send_control(answer)
    except Exception:
        logger.warning(
            "The answer to human-input request %s could not reach its turn",
            answer.request_id,
            exc_info=True,
        )
