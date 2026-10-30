"""Closing a connection pool a Postgres component opened itself."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("roomkit.store")


async def close_own_pool(pool: Any, timeout: float, owner: str) -> None:
    """Close *pool*, which *owner* opened: the connections still in use get
    *timeout* seconds to come back, then are terminated, so a close never
    waits on work that holds one past its cut (RFC §12.10.4)."""
    try:
        await asyncio.wait_for(pool.close(), timeout)
    except TimeoutError:
        logger.warning(
            "%s closed its connection pool with connections still in use after %.1f s: terminated",
            owner,
            timeout,
        )
        pool.terminate()
