"""The turn's reasoning settings against a provider's configuration (RFC §6.7)."""

from __future__ import annotations


def turn_setting[T](turn: T | None, configured: T | None) -> T | None:
    """The turn's value of a reasoning setting when it has one, else the value
    the provider's configuration carries under the same name.

    ``None`` alone means unset: a turn's ``enable_thinking=False`` outranks a
    configured ``True``, where ``turn or configured`` would let it through.
    """
    return turn if turn is not None else configured
