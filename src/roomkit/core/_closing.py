"""What the kit's ``close()`` cuts (RFC §12.10.4): the calls into a room
under way, and the deliveries of the calls it took.

``close()`` stops admission at once: a call that waits for its room's lock
is cut then, and a call still running its hooks or its writes once the
close's grace ran out is cut then. :class:`DoorRuns` holds them. Every
delivery under way is cut at once too, its turns first: :class:`Cascades`
holds them.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import weakref
from collections.abc import Generator
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any

from roomkit.core._reentrance import ReentryBudget
from roomkit.core.exceptions import KitClosedError
from roomkit.core.lanes import DeliveryCascade
from roomkit.models.delivery import InboundResult

KIT_CLOSED = KitClosedError.code

_NESTED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "roomkit_nested_call", default=False
)
"""Whether the current call into a room runs inside another call of the same
task (a hook's), whose run it is."""


def cut_by_close() -> InboundResult:
    """The result of a call into a room the kit's close cut before its
    commit (RFC §12.10.4): nothing of it is stored."""
    return InboundResult(blocked=True, reason=KIT_CLOSED, cancellation_reason=KIT_CLOSED)


def refuse_closed(closed: bool, what: str = "call") -> None:
    """Refuse a *what* on a kit whose ``close()`` has started.

    Raises:
        KitClosedError: the kit is closing or closed.
    """
    if closed:
        raise KitClosedError(f"The kit is closing: no {what} is taken")


class Cascades:
    """The deliveries of the calls the kit took, each its cascade: the ones
    still under way are cut at once when the kit closes (RFC §12.10.4)."""

    def __init__(self) -> None:
        self._live: weakref.WeakSet[DeliveryCascade] = weakref.WeakSet()
        self.closed = False

    def new(self, room_id: str, *, reentry_budget: int | ReentryBudget) -> DeliveryCascade:
        """A cascade for a call into *room_id*; cut at once when the kit is
        closing already: its turns never start, its delivery is dropped."""
        cascade = DeliveryCascade(room_id, reentry_budget=reentry_budget)
        self._live.add(cascade)
        if self.closed:
            cascade.cancel(KIT_CLOSED)
        return cascade

    def cut(self, spared: set[asyncio.Task[Any]]) -> None:
        """Cut every cascade under way, its turns at once (the turn work in
        *spared*, which the close runs under, left to end)."""
        self.closed = True
        for cascade in list(self._live):
            if cascade.in_flight:
                cascade.cancel(KIT_CLOSED, spared=spared, wake=False)

    def turn_work(self) -> set[asyncio.Task[Any]]:
        """The turn work of the cascades the close cut, still unwinding: a
        cut turn stores what it streamed, its open tool calls closed."""
        return {
            task
            for cascade in list(self._live)
            if cascade.cancelled is not None
            for task in cascade.unwinding()
        }


class DoorRuns:
    """The calls into a room under way (``process_inbound``,
    ``send_event``, ``commit_event``, a regeneration), from the call's start
    to its commit: what ``close()`` cuts. One waiting for its room's lock is
    cut at once; one still running (its hooks, its writes) once the close's
    grace ran out. A cut call ends with :class:`KitClosedError`, never a
    ``CancelledError`` its caller did not ask for."""

    def __init__(self) -> None:
        self._running: set[asyncio.Task[Any]] = set()
        self._waiting: set[asyncio.Task[Any]] = set()
        self._cut: set[asyncio.Task[Any]] = set()
        self.closed = False

    @contextlib.contextmanager
    def run(self, what: str = "call") -> Generator[None]:
        """Within it, the current task runs a call into a room, cut as the
        kit closes. A call made from inside another in the same task (a
        hook's) is the outer call's run. Refused on a closing kit.

        Raises:
            KitClosedError: the kit is closing, or its close cut the call.
        """
        refuse_closed(self.closed, what)
        task = asyncio.current_task()
        nested = task is None or task in self._running or task in self._waiting
        if not nested:
            assert task is not None
            self._running.add(task)
        token = _NESTED.set(nested)
        try:
            yield
        except asyncio.CancelledError:
            if nested or not self._was_cut(task):
                raise
            raise KitClosedError(
                f"The kit closed before this {what} was committed: nothing of it is stored"
            ) from None
        finally:
            _NESTED.reset(token)
            if not nested:
                self._leave(task)

    def committed(self) -> None:
        """The current call is past its commit: the close no longer cuts it,
        its delivery is its cascade's. A nested call's commit is not its
        outer call's."""
        task = asyncio.current_task()
        if task is not None and not _NESTED.get():
            self._running.discard(task)

    async def enter_lock(
        self, stack: AsyncExitStack, lock: AbstractAsyncContextManager[Any]
    ) -> None:
        """Wait for the room's *lock* as a call into the room: cut at once
        if the kit closes meanwhile, refused if it is closing already."""
        task = asyncio.current_task()
        if self.closed:
            raise KitClosedError(
                "The kit closed before this call took its room's lock: nothing of it is stored"
            )
        if task is None or task not in self._running:
            await stack.enter_async_context(lock)
            return
        self._running.discard(task)
        self._waiting.add(task)
        try:
            await stack.enter_async_context(lock)
        finally:
            self._waiting.discard(task)
            if not task.done():
                self._running.add(task)

    def cut_waiting(self, spared: set[asyncio.Task[Any]]) -> None:
        """Stop admission (RFC §12.10.4): the calls waiting for their room's
        lock are cut at once."""
        self.closed = True
        for task in list(self._waiting - spared):
            self._cut_task(task)

    def cut_running(self, spared: set[asyncio.Task[Any]]) -> None:
        """Cut every call still running before its commit, but the ones
        *spared* (a call the close runs under)."""
        self.closed = True
        for task in list((self._running | self._waiting) - spared):
            self._cut_task(task)

    def running(self) -> set[asyncio.Task[Any]]:
        """The calls still running before their commit."""
        return {task for task in self._running | self._waiting if not task.done()}

    def _cut_task(self, task: asyncio.Task[Any]) -> None:
        if task in self._cut or task.done():
            return
        self._cut.add(task)
        task.cancel(KIT_CLOSED)

    def _was_cut(self, task: asyncio.Task[Any] | None) -> bool:
        """Whether the close cut *task*, the cancellation it raised the
        close's own: taken back, it is no cancellation of the caller. A
        cancellation from elsewhere besides stays the caller's."""
        if task is None or task not in self._cut:
            return False
        self._cut.discard(task)
        return task.uncancel() == 0 if task.cancelling() else True

    def _leave(self, task: asyncio.Task[Any] | None) -> None:
        """The call *task* ran ended: a cut that never reached it (the call
        ended first) is taken back, so it cuts nothing of its caller's."""
        if task is None:
            return
        self._running.discard(task)
        self._waiting.discard(task)
        if task in self._cut:
            self._cut.discard(task)
            if task.cancelling():
                task.uncancel()
