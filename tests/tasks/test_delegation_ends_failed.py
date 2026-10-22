"""A delegated task that would otherwise never reach its end ends failed, once,
as a task that ran does: its task runner refused it, or its run raised outside
its turn's outcome, inline or in the background (RFC §23.3; RMK-871).

Each end shows everywhere a task's end is read: its line on the StatusBus,
``ON_TASK_COMPLETED``, its child room's status, its completion callback, its
waiters, its span, and ``cancel_task``'s answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import roomkit.core.mixins.delegation as delegation_module
import roomkit.tasks.memory as memory_runner
from roomkit.models.enums import TaskStatus
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks import CancelTaskTool
from roomkit.tasks.models import DelegatedTaskResult
from roomkit.tasks.status import room_task_lines
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from tests.conference.test_conference_realtime import until
from tests.tasks.test_cancel_task import _cancel, _Room


class _Abort(BaseException):
    """What no turn's outcome holds: an exception that is not an Exception."""


async def _aborting_turn(*args: Any, **kwargs: Any) -> str:
    raise _Abort("the worker's process is going")


def _raises_outside_the_outcome(raises: str, module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the run of a task *module* runs raise outside its turn's outcome:
    its turn raises what no outcome holds, or reading its result raises."""
    if raises == "its-turn":
        monkeypatch.setattr(delegation_module, "run_agent_in_child_room", _aborting_turn)
        return
    read = module.finished_task_fields

    def broken(response: Any, failure: Any, context: Any, turns: Any = None) -> dict[str, Any]:
        # The ending's own read (no turn record) still works.
        if turns is not None:
            raise _Abort("the result could not be read")
        return read(response, failure, context)

    monkeypatch.setattr(module, "finished_task_fields", broken)


async def _room() -> tuple[_Room, MockTelemetryProvider, list[DelegatedTaskResult]]:
    room = _Room(MockAIProvider(responses=["Counted."]))
    telemetry = MockTelemetryProvider()
    room.kit._telemetry = telemetry
    await room.open()
    return room, telemetry, []


async def _assert_ended_failed(
    room: _Room, telemetry: MockTelemetryProvider, completed: list[DelegatedTaskResult]
) -> None:
    """The room's one task ended failed wherever its end is read."""
    await until(lambda: bool(room.ended) and bool(completed), timeout=5)
    [line] = await room_task_lines(room.kit.status_bus, "r", 50)
    task_id = line["task_id"]
    child = await room.kit.get_room(completed[0].child_room_id)
    answer = await _cancel(CancelTaskTool(room.kit), task_id)
    [span] = telemetry.get_spans(SpanKind.DELEGATION)

    assert line["status"] == "failed"
    assert [e["task_status"] for e in room.ended] == [TaskStatus.FAILED]
    assert [r.status for r in completed] == [TaskStatus.FAILED]
    assert child.metadata["task_status"] == TaskStatus.FAILED
    assert answer["status"] == "failed"
    assert span.status == "error"
    assert room.kit._room_tasks.running == []


async def test_a_task_its_runner_refuses_ends_failed() -> None:
    room, telemetry, completed = await _room()

    async def full(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the task runner is full")

    async def record(result: DelegatedTaskResult) -> None:
        completed.append(result)

    room.kit._task_runner.submit = full  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="full"):
        await room.kit.delegate("r", "worker", "Count.", notify="speaker", on_complete=record)

    await _assert_ended_failed(room, telemetry, completed)
    assert room.handed_back == []  # its caller received the error
    await room.kit.close()


@pytest.mark.parametrize("raises", ["its-turn", "its-result"])
async def test_an_inline_run_that_raises_outside_its_turn_s_outcome_ends_failed(
    raises: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    room, telemetry, completed = await _room()
    _raises_outside_the_outcome(raises, delegation_module, monkeypatch)

    async def record(result: DelegatedTaskResult) -> None:
        completed.append(result)

    with pytest.raises(_Abort):
        await room.kit.delegate("r", "worker", "Count.", wait=True, on_complete=record)

    await _assert_ended_failed(room, telemetry, completed)
    await room.kit.close()


@pytest.mark.parametrize("raises", ["its-turn", "its-result"])
async def test_a_background_run_that_raises_outside_its_turn_s_outcome_ends_failed(
    raises: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    room, telemetry, completed = await _room()
    _raises_outside_the_outcome(raises, memory_runner, monkeypatch)

    async def record(result: DelegatedTaskResult) -> None:
        completed.append(result)

    task = await room.kit.delegate("r", "worker", "Count.", notify="speaker", on_complete=record)
    result = await asyncio.wait_for(task.wait(), 5)

    assert result.status == TaskStatus.FAILED
    await _assert_ended_failed(room, telemetry, completed)
    await room.kit.close()
