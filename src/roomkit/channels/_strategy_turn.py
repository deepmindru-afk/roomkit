"""Whether the turn a channel runs is a generation a strategy asks (RFC §10.6.1).

A strategy that takes an agent's turns in a room (a loop, a supervisor's
delegation passes) runs on the room's owner and asks each generation of the
agent as a turn of its own, marked ``strategy_turn``. The executor enters every
turn through the channel's ``on_event``, so a subclass that wraps it sees every
turn the channel runs, and says here whether the turn is such a generation: an
AI channel then answers, past its speak policy and its thinker (§6.4).

The executor sets it for every turn it runs, so a turn submitted from inside a
strategy's generation never inherits it.
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

_STRATEGY_TURN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "roomkit_strategy_turn", default=False
)


@contextmanager
def strategy_turn(on: bool) -> Generator[None]:
    """The turn run within is a strategy's generation when *on*."""
    token = _STRATEGY_TURN.set(on)
    try:
        yield
    finally:
        _STRATEGY_TURN.reset(token)


def in_strategy_turn() -> bool:
    """Whether the current turn is a generation a strategy asks."""
    return _STRATEGY_TURN.get()
