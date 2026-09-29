"""Shared asyncio task utilities."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger("roomkit.tasks")


async def _finish_cleanup(coro: Coroutine[Any, Any, object]) -> None:
    """Finish releasing resources before propagating a caller's cancellation."""
    task = asyncio.create_task(coro, name="resource_cleanup")
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def cancel_and_wait(
    *tasks: asyncio.Future[Any] | None, log_errors_to: logging.Logger | None = None
) -> None:
    """Cancel *tasks* and wait until each has ended, without eating the caller's cancellation.

    ``task.cancel()`` then ``with suppress(CancelledError): await task`` also
    swallows a cancellation aimed at the caller: the ``CancelledError`` that
    reaches it while it waits cannot be told from the one the task raises, so
    the caller carries on as if nobody had cancelled it. ``asyncio.wait``
    never raises a task's outcome, so here the two are told apart: the tasks
    still end before the caller moves on, as awaiting them guaranteed, and
    the caller's cancellation is raised once they have.

    ``None`` entries and the current task are skipped: a task tearing itself
    down cannot wait for its own end. A task's own exception is raised, as
    awaiting it would, unless *log_errors_to* is given: it is then logged
    there, at debug, and the teardown goes on. A cancelled caller gets its
    cancellation, never a task's exception.
    """
    current = asyncio.current_task()
    pending = [t for t in dict.fromkeys(tasks) if t is not None and t is not current]
    for task in pending:
        task.cancel()
    cancelled: asyncio.CancelledError | None = None
    while any(not t.done() for t in pending):
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError as exc:
            # The caller's own: the teardown finishes first
            cancelled = cancelled or exc
    errors: list[BaseException] = [
        error for t in pending if not t.cancelled() and (error := t.exception()) is not None
    ]
    if cancelled is not None:
        for error in errors:
            logger.debug("Task failed before its cancellation: %s", error, exc_info=error)
        raise cancelled
    if errors and log_errors_to is None:
        raise errors[0]
    for error in errors:
        (log_errors_to or logger).debug(
            "Task failed before its cancellation: %s", error, exc_info=error
        )


def cancellation_requests() -> int:
    """The current task's pending cancellation requests; 0 outside a task.

    Read before awaiting a task that someone else may cancel, then again on
    its ``CancelledError``: a larger count means the caller itself is being
    cancelled, not only the task it awaited. Comparing counts rather than
    reading ``cancelling()`` alone keeps a request swallowed earlier in the
    task from passing for a new one.
    """
    task = asyncio.current_task()
    return task.cancelling() if task is not None else 0


def log_task_exception(task: asyncio.Task[Any]) -> None:
    """Done-callback that logs unhandled exceptions from fire-and-forget tasks.

    Attach to any :func:`asyncio.create_task` result to prevent silent
    exception loss::

        task = loop.create_task(some_coro())
        task.add_done_callback(log_task_exception)

    Cancelled tasks are silently ignored.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Unhandled exception in task %s: %s",
            task.get_name(),
            exc,
            exc_info=exc,
        )
