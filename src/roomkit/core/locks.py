"""Per-room async locking with LRU eviction."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

logger = logging.getLogger("roomkit.locks")

# ContextVar tracking which rooms the current execution context holds locks
# for.  asyncio.gather() copies the parent context to child tasks, so
# children see the parent's held set and can re-enter without deadlocking.
_held_rooms: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_room_locks_held", default=frozenset()
)


@dataclass
class _LockLease:
    manager: RoomLockManager
    room_id: str
    active: bool = True


_held_leases: contextvars.ContextVar[tuple[_LockLease, ...]] = contextvars.ContextVar(
    "_room_lock_leases", default=()
)


def _has_room_lock(room_id: str, manager: RoomLockManager | None = None) -> bool:
    """Inherited context grants reentrancy only while its acquisition is live."""
    if isinstance(manager, ReleaseTolerantLockManager):
        manager = manager.inner
    return room_id in _held_rooms.get() and any(
        lease.active and lease.room_id == room_id and (manager is None or lease.manager is manager)
        for lease in _held_leases.get()
    )


def unlocked_context() -> contextvars.Context:
    """A copy of the current context that holds no room lock: a task started
    in it from under a lock asks for the lock as any later caller does, and
    waits for its holder, never entering as the holder's reentrant child."""
    context = contextvars.copy_context()
    context.run(_held_rooms.set, frozenset())
    context.run(_held_leases.set, ())
    return context


@contextmanager
def _mark_room_locked(manager: RoomLockManager, room_id: str) -> Generator[None, None, None]:
    lease = _LockLease(manager, room_id)
    rooms_token = _held_rooms.set(_held_rooms.get() | {room_id})
    leases_token = _held_leases.set((*_held_leases.get(), lease))
    try:
        yield
    finally:
        # Children keep copies of the context, but share this revocable lease.
        lease.active = False
        _held_leases.reset(leases_token)
        _held_rooms.reset(rooms_token)


class RoomLockManager(ABC):
    """Abstract base for per-room locking.

    Implement this to plug in any locking backend (Redis, Postgres
    advisory locks, etc.).  The library ships with ``InMemoryLockManager``
    for single-process deployments.

    Implementations should be **reentrant** within the same execution
    context (including child tasks spawned by ``asyncio.gather``): if
    a coroutine already holds the lock for a room and awaits code that
    tries to acquire the same room lock, the inner acquisition must
    succeed without deadlocking.  This is required because tool handlers
    (e.g. handoff) may update room state while the inbound pipeline
    already holds the room lock.
    """

    @abstractmethod
    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        """Acquire an exclusive lock for *room_id*.

        Acquisition MUST be cancellation-safe: the inbound pipeline bounds the
        wait with ``process_timeout`` (RFC §13.6), so a caller that gives up
        queueing is cancelled here. An implementation that leaves the lock
        taken, or a reference held, on cancellation strands the room — every
        later event for it queues behind a lock nobody owns.
        """
        yield  # pragma: no cover

    async def close(self) -> None:
        """Release any resources held by the lock manager (e.g. a connection
        pool). Called by ``RoomKit.close()``. Default is a no-op; overrides
        MUST be idempotent."""
        return None


class InMemoryLockManager(RoomLockManager):
    """In-process per-room asyncio locks with LRU eviction.

    Reentrant within the same execution context: if the current context
    already holds the lock for a given room (including child tasks
    spawned by ``asyncio.gather``), ``locked()`` yields immediately
    instead of deadlocking.

    Suitable for single-process deployments.  For multi-process or
    distributed setups, provide a custom ``RoomLockManager`` backed by
    Redis, Postgres advisory locks, or similar.
    """

    def __init__(self, max_locks: int = 1024) -> None:
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._refcounts: dict[str, int] = {}
        self._max_locks = max_locks
        self._mgr_lock = asyncio.Lock()

    def _get_lock(self, room_id: str) -> asyncio.Lock:
        if room_id in self._locks:
            self._locks.move_to_end(room_id)
            self._refcounts[room_id] = self._refcounts.get(room_id, 0) + 1
            return self._locks[room_id]

        lock = asyncio.Lock()
        self._locks[room_id] = lock
        self._refcounts[room_id] = 1
        self._evict()
        return lock

    def _release_ref(self, room_id: str) -> None:
        """Decrement the reference count for a room lock."""
        count = self._refcounts.get(room_id, 0) - 1
        if count <= 0:
            self._refcounts.pop(room_id, None)
        else:
            self._refcounts[room_id] = count

    def _evict(self) -> None:
        if len(self._locks) <= self._max_locks:
            return
        to_remove: list[str] = []
        for key, lock in self._locks.items():
            if len(self._locks) - len(to_remove) <= self._max_locks:
                break
            if not lock.locked() and self._refcounts.get(key, 0) <= 0:
                to_remove.append(key)
        for key in to_remove:
            self._locks.pop(key)
            self._refcounts.pop(key, None)

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        """Acquire the lock for a room (reentrant via ContextVar)."""
        if _has_room_lock(room_id, self):
            # Reentrant: this execution context already holds the lock.
            yield
            return

        async with self._mgr_lock:
            lock = self._get_lock(room_id)
        try:
            async with lock:
                with _mark_room_locked(self, room_id):
                    yield
        finally:
            async with self._mgr_lock:
                self._release_ref(room_id)

    @property
    def size(self) -> int:
        """Return the number of locks currently held."""
        return len(self._locks)


# The failed releases met by the operation that set the list (RFC §13.5).
_release_failures: contextvars.ContextVar[list[Exception] | None] = contextvars.ContextVar(
    "_room_lock_release_failures", default=None
)


@contextmanager
def collecting_release_failures() -> Generator[list[Exception], None, None]:
    """Collect the failed releases of the room locks taken within, so that
    the operation can report them (RFC §13.5)."""
    failures: list[Exception] = []
    token = _release_failures.set(failures)
    try:
        yield failures
    finally:
        _release_failures.reset(token)


def failed_release() -> Exception | None:
    """The first failed release the collecting operation met so far, if any."""
    failures = _release_failures.get()
    return failures[0] if failures else None


class ReleaseTolerantLockManager(RoomLockManager):
    """The kit's hold on a lock manager: a release that fails changes
    nothing the locked section decided or wrote (RFC §13.5).

    The failure is logged and handed to the operation collecting it, never
    raised: an event committed stays committed, a refusal stays a refusal.
    A failure met inside the section stays the one raised. What a failed
    release leaves behind is the wrapped manager's to recover; the next
    section asks it for the lock again.
    """

    def __init__(self, inner: RoomLockManager) -> None:
        self.inner = inner

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        lock = self.inner.locked(room_id)
        await lock.__aenter__()
        try:
            yield
        except BaseException as exc:
            try:
                await lock.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception as release_error:
                # A cancellation met while releasing is the caller's, not a
                # failed release: it is raised, as on the normal path.
                if release_error is not exc:
                    _log_failed_release(room_id, release_error)
            raise
        try:
            await lock.__aexit__(None, None, None)
        except Exception as release_error:
            _log_failed_release(room_id, release_error)
            failures = _release_failures.get()
            if failures is not None:
                failures.append(release_error)

    async def close(self) -> None:
        await self.inner.close()


def _log_failed_release(room_id: str, error: BaseException) -> None:
    logger.warning("Releasing the lock of room %s failed: %r", room_id, error)
