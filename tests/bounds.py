"""Library bounds a test expires itself.

A budget under test is configured as :data:`CUT` seconds, a value no default
uses: :func:`opened_with_cut` keeps every ``asyncio.timeout`` opened with it
(or with a bound the library sets itself), ``asyncio.wait_for``'s included,
and :func:`expire` ends them at the moment the test chooses. A bound the
test itself opens after the capture is kept too: open the test's own
deadline before it. The cut goes
through the library's own bound and its own timeout path, on a slow runner as
on a fast one, and costs no wall-clock wait.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

CUT = 20.0


def opened_with_cut(
    monkeypatch: pytest.MonkeyPatch, seconds: float | None = CUT
) -> list[asyncio.Timeout]:
    """Every ``asyncio.timeout`` opened with *seconds* from now on: :data:`CUT`
    by default, a bound the library sets itself when given, every bound when
    None (a test proving that none cuts a step)."""
    opened: list[asyncio.Timeout] = []
    opens = asyncio.timeouts.timeout

    def timeout(delay: float | None) -> asyncio.Timeout:
        bound = opens(delay)
        if seconds is None or delay == seconds:
            opened.append(bound)
        return bound

    monkeypatch.setattr(asyncio, "timeout", timeout)
    # asyncio.wait_for opens its bound through the module, not the package.
    monkeypatch.setattr(asyncio.timeouts, "timeout", timeout)
    return opened


def expire(bounds: list[asyncio.Timeout]) -> None:
    """End every bound of *bounds* now; one that already ended stays as it was
    (a pass bounded alike before the work the test cuts)."""
    now = asyncio.get_running_loop().time()
    for bound in bounds:
        with contextlib.suppress(RuntimeError):
            bound.reschedule(now)
