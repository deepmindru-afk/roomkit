"""A strategy's worker run whose worker posted its progress, then was cut by
its timeout or cancelled: the run's terminal entry names the task, so the
task's line on the status bus ends with it (RFC §19.7.3, §23.4; RMK-875)."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.orchestration._worker_run import WorkerStatus, follow_worker, run_worker
from roomkit.tasks.handback import CALLER_HANDS_BACK
from roomkit.tasks.status import post_task_progress, room_task_lines
from roomkit.tools.context import tool_turn_context
from tests.tasks.test_cancel_task import _Endless, _Room


async def _worker_run(room: _Room, door: str, timeout: float | None) -> asyncio.Future[object]:
    """A sequential strategy's run of ``worker`` in room ``r``: inline, or
    following a task the runner runs in the background."""
    status = WorkerStatus({"room_id": "r", "strategy": "sequential"})
    if door == "inline":
        run = run_worker(room.kit, "r", "worker", "Count.", timeout=timeout, status=status)
        return asyncio.ensure_future(run)
    task = await room.kit.delegate(
        "r", "worker", "Count.", notify=CALLER_HANDS_BACK, post_status=False
    )
    return asyncio.ensure_future(follow_worker(room.kit, task, timeout=timeout, status=status))


@pytest.mark.parametrize("door", ["inline", "background"])
@pytest.mark.parametrize("end", ["timed-out", "cancelled"])
async def test_a_worker_cut_after_its_progress_leaves_its_line_ended(door: str, end: str) -> None:
    room = await _Room(_Endless()).open()
    run = await _worker_run(room, door, timeout=None if end == "cancelled" else 0.5)
    await asyncio.wait_for(room.worker.started.wait(), 5)
    child = next(r.id for r in await room.kit.store.list_rooms() if r.id.startswith("r::task-"))
    with tool_turn_context(room_id=child):
        assert await post_task_progress(room.kit, "3 of 5")

    if end == "cancelled":
        run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    lines = await room_task_lines(room.kit.status_bus, "r", 50)
    await room.kit.close()

    assert [(line["task_id"].startswith("task-"), line["status"]) for line in lines] == [
        (True, "failed")
    ]
    assert lines[0]["progress"] == "3 of 5"
