"""The human-input requests of the turn the current code runs in (RFC §10.6.5).

A leaf module. A tool waiting for a human's answer runs on the turn's
executor, and an answer may arrive on the room owner, which keeps the turn's
open requests. The executor sets, for every turn it runs, what carries a
request's life to the owner: its opening (before ``ON_USER_INPUT_REQUIRED``
fires), and its settling where the tool waits (an answer the channel's own
handler received, a hook's block, its bound, a release), after which the
owner answers it no more. Outside a turn the backend runs, there is none,
and a handler settles its requests alone.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from roomkit.models.pending_input import PendingInput


class TurnInputs(Protocol):
    """How one turn's human-input requests reach its room owner."""

    async def opened(self, pending: PendingInput, settle: InputSettler) -> None:
        """The request is open: the owner learns it, and answers it through
        *settle* (a held early answer included)."""
        ...

    def closed(self, pending_id: str, reason: str) -> None:
        """The request settled here, not by the owner's answer."""
        ...

    @property
    def open(self) -> bool:
        """Whether the turn still runs: a request opened after it ended is
        the handler's alone."""
        ...


class InputSettler(Protocol):
    """What settles a request with the outcome its owner recorded."""

    def settle(self, pending_id: str, *, value: str | None, reason: str | None) -> bool:
        """Settle *pending_id* with an answer, or a rejection; whether it was open."""
        ...

    def withdraw(self, pending_id: str) -> None:
        """Withdraw *pending_id*: its turn ended while it was open."""
        ...


_TURN_INPUTS: contextvars.ContextVar[TurnInputs | None] = contextvars.ContextVar(
    "roomkit_turn_inputs", default=None
)


def current_turn_inputs() -> TurnInputs | None:
    """The human-input requests of the turn the current code runs in."""
    return _TURN_INPUTS.get()


def set_turn_inputs(inputs: TurnInputs | None) -> None:
    """Set them for the turn the current context runs: an executor sets them
    for every turn it runs."""
    _TURN_INPUTS.set(inputs)
