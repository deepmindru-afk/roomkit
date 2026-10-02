"""Every driver of the conformance suite, and the providers no driver stands for."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from types import ModuleType

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.ai.mock import MockAIProvider
from tests.text_conformance import (
    anthropic_wire,
    gemini_wire,
    mistral_wire,
    ollama_wire,
    openai_wire,
    polargrid_wire,
)
from tests.text_conformance.driver import Driver

# One module per wire family; ``wires()`` builds a fresh driver per provider.
_FAMILIES: tuple[ModuleType, ...] = (
    anthropic_wire,
    gemini_wire,
    mistral_wire,
    ollama_wire,
    openai_wire,
    polargrid_wire,
)


def _fresh(family: ModuleType, position: int) -> Driver:
    return family.wires()[position]


def drivers() -> dict[str, Callable[[], Driver]]:
    """Each driver's label and a factory: every test gets a fresh driver, which
    records the requests of its own provider."""
    factories: dict[str, Callable[[], Driver]] = {}
    for family in _FAMILIES:
        for position, wire in enumerate(family.wires()):
            factories[wire.label] = partial(_fresh, family, position)
    return factories


def covered() -> set[type[AIProvider]]:
    """Every provider class some driver stands for."""
    return {cls for factory in drivers().values() for cls in factory().covers}


EXEMPT: dict[type[AIProvider], str] = {
    MockAIProvider: "a test double: it hands back what the test scripted, on no wire",
}
"""Providers no driver stands for, each with its reason."""
