"""PostgreSQL advisory-lock RoomLockManager for multi-process deployments.

An in-memory ``RoomLockManager`` only serializes rooms within a single process.
When several RoomKit processes share one PostgreSQL store (a load-balanced
deployment), room processing must be serialized *across* processes — otherwise
two workers can interleave on the same room (RFC §8.1 / §13.5). This manager
uses PostgreSQL **session** advisory locks for that.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from roomkit.core.locks import RoomLockManager, _has_room_lock, _mark_room_locked
from roomkit.store._pg_pool import close_own_pool


def _advisory_key(room_id: str) -> int:
    """Map a room id to a signed 64-bit key for ``pg_advisory_lock``.

    Uses a stable hash (not Python's salted ``hash()``). Distinct rooms may
    collide onto the same key — harmless: they merely serialize together.
    """
    digest = hashlib.blake2b(room_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class PostgresAdvisoryLockManager(RoomLockManager):
    """Cross-process per-room locking via PostgreSQL session advisory locks.

    A session-level advisory lock is bound to the connection that took it, held
    for the whole ``locked()`` block. This manager therefore MUST use a
    connection pool **separate** from the store's query pool: a held lock
    connection would otherwise starve the query the locked block itself needs,
    deadlocking. Size ``max_size`` for the number of rooms processed
    concurrently.

    Reentrant within an execution context (mirrors ``InMemoryLockManager``), so
    a tool handler that re-enters the pipeline under the same room lock does not
    deadlock.
    """

    _close_timeout: float = 1.0

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: Any = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        try:
            import asyncpg as _asyncpg
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PostgresAdvisoryLockManager. "
                "Install it with: pip install roomkit[postgres]"
            ) from exc
        self._asyncpg = _asyncpg
        self._dsn = dsn
        self._pool = pool
        self._owns_pool = pool is None
        self._min_size = min_size
        self._max_size = max_size

    async def init(self) -> None:
        """Create the (separate) advisory-lock connection pool if needed."""
        if self._pool is None:
            self._pool = await self._asyncpg.create_pool(
                self._dsn, min_size=self._min_size, max_size=self._max_size
            )

    async def close(self) -> None:
        """Release the connection pool if we own it, once its connections
        are back: a lock held is held until its holder lets it go.
        Idempotent."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None

    async def _close_bounded(self) -> None:
        """:meth:`close` as the kit's close runs it (RFC §12.10.4): the
        connections still in use (a lock held past its holder's cut) get
        ``_close_timeout`` seconds to come back, then are terminated. A pool
        handed in stays its owner's. Idempotent."""
        if self._pool is not None and self._owns_pool:
            pool, self._pool = self._pool, None
            await close_own_pool(pool, self._close_timeout, "PostgresAdvisoryLockManager")

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        """Hold a cross-process exclusive lock for *room_id* (reentrant)."""
        if _has_room_lock(room_id, self):
            # Reentrant: this execution context already holds the lock.
            yield
            return
        if self._pool is None:
            raise RuntimeError("PostgresAdvisoryLockManager.init() must be called before use")
        key = _advisory_key(room_id)
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT pg_advisory_lock($1)", key)
            try:
                with _mark_room_locked(self, room_id):
                    yield
            finally:
                await conn.execute("SELECT pg_advisory_unlock($1)", key)
