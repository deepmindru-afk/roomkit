"""A steering directive applied to a turn's loops on its executor (RFC §10.6.8)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.models.steering import SteeringDirective
    from roomkit.turns.wire import TurnRequest

logger = logging.getLogger("roomkit.turns")


def _takes(function: Callable[..., Any], name: str) -> bool:
    """Whether *function* takes the keyword argument *name*."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


async def steer_turn(
    channel: Channel | None, directive: SteeringDirective, request: TurnRequest
) -> int | None:
    """Apply *directive* to the loops of *request*'s turn, through the
    channel's ``steer()``, sync or async: how many it reached; ``None`` for a
    channel without steering, not built yet, or whose ``steer()`` cannot reach
    the turn yet (answers ``None``), whose owner cuts the turn on a
    ``Cancel``."""
    steer = getattr(channel, "steer", None)
    if steer is None:
        return None
    # A steer() of the channel's own from before turns were addressed takes
    # the turn's room instead.
    address = (
        {"turn_id": request.turn_id} if _takes(steer, "turn_id") else {"room_id": request.room_id}
    )
    try:
        reached = steer(directive, **address)
        if inspect.isawaitable(reached):
            reached = await reached
    except Exception:
        logger.exception("Turn %s: the channel's steer() failed", request.turn_id)
        return 0
    return reached if reached is None or isinstance(reached, int) else 0
