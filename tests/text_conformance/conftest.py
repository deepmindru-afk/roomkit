"""Every scenario runs on every driver; most run streamed and through ``generate()``."""

from __future__ import annotations

import pytest

from tests.text_conformance.driver import Driver
from tests.text_conformance.registry import drivers

_DRIVERS = drivers()


@pytest.fixture(params=sorted(_DRIVERS))
def driver(request: pytest.FixtureRequest) -> Driver:
    """A fresh driver: it records the requests of its own provider."""
    return _DRIVERS[request.param]()


@pytest.fixture(params=["stream", "generate"])
def mode(request: pytest.FixtureRequest) -> str:
    return request.param
