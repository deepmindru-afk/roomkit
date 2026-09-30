"""The turn's reasoning settings against a provider's configuration (RFC §6.7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.providers.ai.base import AIContext

# The efforts a turn may ask for, least to most; ``none`` is off, not a level.
_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def turn_setting[T](turn: T | None, configured: T | None) -> T | None:
    """The turn's value of a reasoning setting when it has one, else the value
    the provider's configuration carries under the same name.

    ``None`` alone means unset: a turn's ``enable_thinking=False`` outranks a
    configured ``True``, where ``turn or configured`` would let it through.
    """
    return turn if turn is not None else configured


def thinking_switch(context: AIContext, configured: bool | None = None) -> bool | None:
    """Whether the model reasons on this turn, as the turn states it.

    ``thinking_budget`` states it first (``0`` off, above ``0`` on, and a
    negative one off, save where the vendor gives it a meaning of its own,
    as Gemini's dynamic ``-1``), then ``enable_thinking``, and a
    ``reasoning_effort`` of ``none`` states off. What the turn leaves
    unstated falls to *configured*, the switch the provider's configuration
    carries, a vendor setting of its own included (RFC §6.7). ``None`` when
    nothing states it: the model's default applies.
    """
    if context.thinking_budget is not None:
        return context.thinking_budget > 0
    if context.enable_thinking is not None:
        return context.enable_thinking
    if context.reasoning_effort == "none":
        return False
    return configured


def nearest_level(effort: str | None, levels: Sequence[str]) -> str | None:
    """The level of *levels* (least to most) nearest to *effort*, the lower
    one on a tie; ``None`` for an effort outside the shared scale, ``none``
    included, or a model that takes no level (RFC §6.7)."""
    if effort not in _EFFORTS or not levels:
        return None
    if effort in levels:
        return effort
    rank = _EFFORTS.index(effort)
    return min(
        levels, key=lambda level: (abs(_EFFORTS.index(level) - rank), _EFFORTS.index(level))
    )
