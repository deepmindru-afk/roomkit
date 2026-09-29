"""Tests for roomkit.core.task_utils."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from roomkit.core.task_utils import (
    cancel_and_wait,
    cancellation_requests,
    log_task_exception,
)


class TestLogTaskException:
    async def test_logs_exception(self, caplog):
        async def _fail():
            raise ValueError("boom")

        task = asyncio.create_task(_fail())
        with contextlib.suppress(ValueError):
            await task
        with caplog.at_level("ERROR", logger="roomkit.tasks"):
            log_task_exception(task)
        assert "boom" in caplog.text

    async def test_no_log_on_success(self, caplog):
        async def _ok():
            return 42

        task = asyncio.create_task(_ok())
        await task
        with caplog.at_level("ERROR", logger="roomkit.tasks"):
            log_task_exception(task)
        assert caplog.text == ""

    async def test_no_log_on_cancel(self, caplog):
        async def _hang():
            await asyncio.sleep(999)

        task = asyncio.create_task(_hang())
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with caplog.at_level("ERROR", logger="roomkit.tasks"):
            log_task_exception(task)
        assert caplog.text == ""


class TestCancelAndWait:
    async def test_cancels_and_waits_for_the_task(self):
        ended = asyncio.Event()

        async def _hang():
            try:
                await asyncio.sleep(999)
            finally:
                ended.set()

        task = asyncio.create_task(_hang())
        await asyncio.sleep(0)
        await cancel_and_wait(task)
        assert task.cancelled()
        assert ended.is_set()

    async def test_the_callers_cancellation_reaches_the_caller_after_the_teardown(self):
        """RMK-288: a caller cancelled while it waits is cancelled, not resumed,
        and only once the task it waits for has ended."""
        release = asyncio.Event()
        resumed = False

        async def _slow_to_stop():
            try:
                await asyncio.sleep(999)
            finally:
                # Cleanup that outlasts the caller's own cancellation.
                await release.wait()

        async def _caller(task: asyncio.Task[None]) -> None:
            nonlocal resumed
            await cancel_and_wait(task)
            resumed = True

        task = asyncio.create_task(_slow_to_stop())
        await asyncio.sleep(0)
        caller = asyncio.create_task(_caller(task))
        await asyncio.sleep(0)
        caller.cancel()
        await asyncio.wait({caller}, timeout=0.05)
        assert not caller.done()  # still finishing the teardown

        release.set()
        await asyncio.wait({caller}, timeout=1.0)
        assert task.done()
        assert caller.cancelled()
        assert not resumed

    async def test_a_cancelled_caller_gets_its_cancellation_not_the_tasks_error(self):
        async def _fail_on_cancel():
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
                raise ValueError("boom") from None

        async def _caller(task: asyncio.Task[None]) -> None:
            await cancel_and_wait(task)

        task = asyncio.create_task(_fail_on_cancel())
        await asyncio.sleep(0)
        caller = asyncio.create_task(_caller(task))
        await asyncio.sleep(0)
        caller.cancel()
        await asyncio.wait({caller}, timeout=1.0)
        assert caller.cancelled()

    async def test_a_tasks_own_error_is_raised(self):
        async def _fail_on_cancel():
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                raise ValueError("boom") from None

        task = asyncio.create_task(_fail_on_cancel())
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="boom"):
            await cancel_and_wait(task)

    async def test_a_tasks_own_error_is_logged_when_asked(self, caplog):
        async def _fail():
            raise ValueError("boom")

        task = asyncio.create_task(_fail())
        await asyncio.sleep(0)
        with caplog.at_level("DEBUG", logger="roomkit.tasks"):
            await cancel_and_wait(task, log_errors_to=logging.getLogger("roomkit.tasks"))
        assert "boom" in caplog.text

    async def test_skips_none_and_the_current_task(self):
        async def _tear_down_self() -> str:
            await cancel_and_wait(None, asyncio.current_task())
            return "done"

        assert await asyncio.create_task(_tear_down_self()) == "done"

    async def test_waits_for_every_task(self):
        async def _hang():
            await asyncio.sleep(999)

        tasks = [asyncio.create_task(_hang()) for _ in range(3)]
        await asyncio.sleep(0)
        await cancel_and_wait(*tasks)
        assert all(t.cancelled() for t in tasks)


class TestCancellationRequests:
    async def test_tells_an_interrupt_from_the_callers_own_cancellation(self):
        outcomes: list[str] = []

        async def _playback():
            await asyncio.sleep(999)

        async def _caller(inner: asyncio.Task[None]) -> None:
            requested = cancellation_requests()
            try:
                await inner
            except asyncio.CancelledError:
                if cancellation_requests() > requested:
                    outcomes.append("caller cancelled")
                    raise
                outcomes.append("interrupted")

        inner = asyncio.create_task(_playback())
        caller = asyncio.create_task(_caller(inner))
        await asyncio.sleep(0)
        inner.cancel()  # a playback interrupt
        await caller

        inner = asyncio.create_task(_playback())
        caller = asyncio.create_task(_caller(inner))
        await asyncio.sleep(0)
        caller.cancel()
        await asyncio.wait({caller})

        assert outcomes == ["interrupted", "caller cancelled"]
        assert caller.cancelled()
