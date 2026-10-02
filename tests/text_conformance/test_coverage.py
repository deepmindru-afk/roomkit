"""No text provider escapes the suite: each has a driver, or an exemption that
says why."""

from __future__ import annotations

import importlib
import pkgutil

import roomkit.providers as providers_package
from roomkit.providers.ai.base import AIProvider
from tests.text_conformance.registry import EXEMPT, covered


def _name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _provider_classes() -> set[str]:
    """Every AIProvider class under roomkit.providers whose module imports, by
    name: a test that reloads a module leaves a second class object behind."""
    for module in pkgutil.walk_packages(providers_package.__path__, "roomkit.providers."):
        try:
            importlib.import_module(module.name)
        except ImportError:
            continue
    found: set[str] = set()
    pending = list(AIProvider.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if cls.__module__.startswith("roomkit.providers."):
            found.add(_name(cls))
    return found


def test_every_text_provider_has_a_driver_or_says_why_not() -> None:
    accounted = {_name(cls) for cls in (*covered(), *EXEMPT)}
    missing = _provider_classes() - accounted

    assert not missing, "add a driver in tests/text_conformance for " + ", ".join(sorted(missing))
