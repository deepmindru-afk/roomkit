"""Coroutines collected never awaited: Python's RuntimeWarning, raised at an
unrelated moment, which fails a host's tests under ``-W error`` (RMK-901,
RMK-909)."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import warnings
from collections.abc import Iterator
from typing import Any


@contextlib.contextmanager
def never_awaited() -> Iterator[list[str]]:
    """The coroutines collected unawaited within the block, a collection at
    its end included. The block frees what it held first (its tasks, a
    loop's name), or what they hold is collected after it, unseen."""
    found: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield found
        gc.collect()
    found.extend(str(w.message) for w in caught if "never awaited" in str(w.message))


async def cancelled_before_start(tasks: list[asyncio.Task[Any]]) -> list[str]:
    """Cancel *tasks* before their first step and free them: the coroutines
    collected unawaited meanwhile."""
    with never_awaited() as found:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        tasks.clear()
        del task
        await asyncio.sleep(0)
    return found
