"""The turn's reasoning settings as an Anthropic request spells them (RFC §6.7).

Whether the model reasons and how much: the ``thinking`` block, the
``output_config.effort`` and the ``temperature`` a turn sends, for the model
the catalogue (``models.py``) declares. A model it does not carry is sent
the turn's effort on Anthropic's levels, and nothing where the turn states
off.
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.ai.reasoning import (
    floored_effort,
    nearest_level,
    reasoning_floor,
    thinking_switch,
)
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.models import (
    EFFORT_NO_XHIGH,
    EFFORT_REFUSED,
    EFFORT_UP_TO_HIGH,
    THINKING_OFF_BETWEEN_TOOLS,
    THINKING_OFF_DISABLED,
    THINKING_OFF_EFFORT_HIGH,
)

# The ``thinking`` block that turns reasoning off, by the catalogue tag
# declaring it.
_OFF_BLOCKS = (
    (THINKING_OFF_DISABLED, "disabled"),
    (THINKING_OFF_BETWEEN_TOOLS, "between_tools"),
)

# Anthropic's efforts, least to most, under the turn's ``minimal``, which no
# Claude model takes (400, measured 2026-10-12): it goes as ``low``.
_SCALE = ("minimal", "low", "medium", "high", "xhigh", "max")
_LEVELS = _SCALE[1:]
_UP_TO_HIGH = ("low", "medium", "high")
_NO_XHIGH = ("low", "medium", "high", "max")


def thinking_or_temperature(
    config: AnthropicConfig, context: AIContext, model: ModelInfo | None
) -> dict[str, Any]:
    """The turn's ``thinking`` block and its ``temperature``, as each applies.

    Anthropic ignores temperature while thinking, so a thinking turn never
    sends one. Newer models (Opus 4.7/4.8, Fable 5) reject the
    ``budget_tokens`` shape and want adaptive thinking instead;
    ``display: "summarized"`` keeps the reasoning trace visible (its default
    is "omitted" on those models).

    Thinking is on as the turn states it (RFC §6.7): a positive budget, or
    ``enable_thinking=True``. A model without adaptive thinking needs its
    budget, so ``enable_thinking=True`` alone leaves it off there. Where the
    turn states off, a model that reasons when ``thinking`` is left out is
    sent the block its catalogue entry declares (:func:`_thinking_off`).

    ``messages.stream()`` has no ``temperature`` parameter in anthropic 1.x,
    while the models profiled as taking one still do, so it rides
    ``extra_body``, which the SDK merges into the request JSON as it is.
    """
    budget = context.thinking_budget
    switch = thinking_switch(context)
    if switch:
        if config.use_adaptive_thinking:
            return {"thinking": {"type": "adaptive", "display": "summarized"}}
        if budget:
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    sent: dict[str, Any] = {}
    off = _thinking_off(model) if switch is False else None
    if off is not None:
        sent["thinking"] = off
    if context.temperature is not None and config.supports_custom_temperature:
        sent["extra_body"] = {"temperature": context.temperature}
    return sent


def _thinking_off(model: ModelInfo | None) -> dict[str, Any] | None:
    """The ``thinking`` block turning the model's reasoning off, as its
    catalogue entry declares it: ``disabled``, or ``between_tools`` on a model
    that refuses ``disabled``. ``None`` where leaving ``thinking`` out is off
    already, where the model cannot stop (off is then its lowest effort,
    :func:`turn_effort`), or where the catalogue does not carry the model."""
    tags = model.capabilities if model is not None else []
    for tag, kind in _OFF_BLOCKS:
        if tag in tags:
            return {"type": kind}
    return None


def turn_effort(context: AIContext, model: ModelInfo | None) -> str | None:
    """The ``output_config.effort`` the turn sends, ``None`` for none.

    The turn's effort goes as the nearest level the model takes, as its
    catalogue entry declares them (RFC §6.7), on a turn with tools as on one
    without: Anthropic's own ``max`` included, ``minimal`` as ``low``, and
    ``none`` is not one of them, it states reasoning off. Where the turn
    states off, a model that cannot stop reasoning is sent its lowest level,
    and one that takes no effort above ``high`` with reasoning off is sent
    ``high`` for a higher one. A model that refuses the parameter is sent
    none, and so is an effort outside Anthropic's scale.
    """
    tags = model.capabilities if model is not None else []
    if EFFORT_REFUSED in tags:
        return None
    effort = floored_effort(context, None, reasoning_floor(model))
    off = thinking_switch(context) is False
    return nearest_level(effort, _levels(tags, off), scale=_SCALE)


def _levels(tags: list[str], off: bool) -> tuple[str, ...]:
    """The efforts a model takes, least to most, as its catalogue *tags*
    declare them; every Anthropic level for a model the catalogue does not
    carry. With reasoning *off*, none above ``high`` where the model takes
    none there."""
    if EFFORT_UP_TO_HIGH in tags or (off and THINKING_OFF_EFFORT_HIGH in tags):
        return _UP_TO_HIGH
    if EFFORT_NO_XHIGH in tags:
        return _NO_XHIGH
    return _LEVELS
