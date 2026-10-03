"""Run a report to its end even if the task awaiting it is cancelled."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_REPORTS_IN_FLIGHT: set[asyncio.Task[None]] = set()
"""Reports a cancelled task shields: held here so none is collected mid-run."""


async def shielded(report: Coroutine[Any, Any, None]) -> None:
    """Run *report* to its end even if the task awaiting it is cancelled meanwhile."""
    task = asyncio.ensure_future(report)
    _REPORTS_IN_FLIGHT.add(task)
    task.add_done_callback(_REPORTS_IN_FLIGHT.discard)
    await asyncio.shield(task)
