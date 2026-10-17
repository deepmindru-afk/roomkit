"""The human-input requests of the turn the current code runs in (RFC §10.6.5).

A leaf module. A tool waiting for a human's answer runs on the turn's
executor, and the answer arrives on the room owner, which keeps the turn's
open requests. The executor sets, for every turn it runs, what carries a
request's life to the owner: its opening (before ``ON_USER_INPUT_REQUIRED``
fires), its close without an answer, and an answer or a rejection the
channel's own handler receives, which the owner records so both doors answer
the request once. Outside a turn the backend runs, there is none, and a
handler settles its requests itself.
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
        """The request settled here without the owner's answer."""
        ...

    def forward(self, pending_id: str, *, value: str | None, reason: str | None) -> None:
        """An answer (*value*) or a rejection (*reason*) the channel's handler
        received: the owner records it and answers the request."""
        ...


class InputSettler(Protocol):
    """What settles a request with the outcome its owner recorded."""

    def settle(self, pending_id: str, *, value: str | None, reason: str | None) -> bool:
        """Settle *pending_id* with an answer, or a rejection; whether it was open."""
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
