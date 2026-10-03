"""A delegated worker that its round cap cuts has no answer (RMK-414).

RFC §23.3 step 6, RFC §6.4: the task fails, its error naming how the turn
ended, its output the worker's last narration and its metadata the
``loop_end_reason``; streamed or buffered, inline or in the background, with
a transport shared into the child room or not. A worker that owes a result
and submitted it before the cut keeps it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, TaskCutShortError
from roomkit.channels.agent import Agent
from roomkit.models.enums import TaskStatus
from roomkit.orchestration.strategies.loop import _execute_loop
from roomkit.orchestration.strategies.supervisor.results import _result_output
from roomkit.orchestration.strategies.supervisor.supervised import _supervisor_dispatch
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult, task_work
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _kit(responses: list[AIResponse], *, streaming: bool) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(
        Agent(
            "worker",
            provider=MockAIProvider(ai_responses=responses, streaming=streaming),
            tools=[LOOKUP],
            tool_handler=_found,
            tool_search=False,
            max_tool_rounds=1,
        )
    )
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    return kit


def _assert_cut(result: Any) -> None:
    assert result is not None
    assert result.status == TaskStatus.FAILED
    assert result.error == "The worker's turn ended max_rounds before its answer"
    assert result.output == "Still checking."
    assert result.metadata["loop_end_reason"] == "max_rounds"


@pytest.mark.parametrize("streaming", [False, True], ids=["generates", "streams"])
@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared-transport"])
async def test_an_inline_task_fails_with_its_narration(streaming: bool, shared: bool) -> None:
    kit = await _kit([LOOPING] * 3, streaming=streaming)

    task = await kit.delegate(
        "parent",
        "worker",
        "Find it.",
        share_channels=["email-out"] if shared else None,
        wait=True,
    )

    _assert_cut(task.result)
    await kit.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["generates", "streams"])
async def test_a_background_task_fails_with_its_narration(streaming: bool) -> None:
    kit = await _kit([LOOPING] * 3, streaming=streaming)

    task = await kit.delegate("parent", "worker", "Find it.")
    result = await task.wait(timeout=5)

    _assert_cut(result)
    await kit.close()


async def test_a_worker_that_finished_still_completes() -> None:
    kit = await _kit([AIResponse(content="Found it.")], streaming=True)

    task = await kit.delegate("parent", "worker", "Find it.", wait=True)

    assert task.result is not None
    assert (task.result.status, task.result.output) == (TaskStatus.COMPLETED, "Found it.")
    assert "loop_end_reason" not in task.result.metadata
    await kit.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["generates", "streams"])
async def test_a_result_submitted_before_the_cut_counts(streaming: bool) -> None:
    submitted = {"status": "completed", "summary": "found it", "data": {}}
    submit = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="s1", name="submit_result", arguments=submitted)],
    )
    kit = await _kit([submit, LOOPING, LOOPING], streaming=streaming)

    task = await kit.delegate(
        "parent", "worker", "Find it.", wait=True, require_structured_result=True
    )

    assert task.result is not None
    assert task.result.status == TaskStatus.COMPLETED
    assert json.loads(task.result.output or "{}")["summary"] == "found it"
    await kit.close()


async def test_a_worker_owing_a_result_fails_when_cut_without_one() -> None:
    kit = await _kit([LOOPING] * 6, streaming=True)

    task = await kit.delegate(
        "parent", "worker", "Find it.", wait=True, require_structured_result=True
    )

    _assert_cut(task.result)
    await kit.close()


async def test_a_loop_does_not_take_a_cut_producers_narration_for_its_work() -> None:
    """Every orchestration reader reads a failed task's work as none (RMK-414)."""
    kit = await _kit([LOOPING] * 3, streaming=True)
    reviewer = Agent("reviewer", provider=MockAIProvider(responses=["APPROVED"] * 3))
    kit.register_channel(reviewer)
    producer = kit.channels["worker"]

    out = await _execute_loop(
        kit=kit,
        room_id="parent",
        producer=producer,
        reviewers=[reviewer],
        strategy=None,
        task_desc="Find it.",
        max_iterations=2,
    )

    assert out["approved"] is False
    assert out["output"] == ""
    await kit.close()


def test_a_supervisor_reads_a_cut_worker_as_failed() -> None:
    cut = DelegatedTaskResult(
        task_id="t",
        child_room_id="c",
        parent_room_id="p",
        agent_id="worker",
        status=TaskStatus.FAILED,
        output="Still checking.",
        error="The worker's turn ended max_rounds before its answer",
    )

    assert task_work(cut) == ""
    assert _result_output(cut) == "The task failed."


async def test_a_supervisor_cut_framing_the_first_task_hands_on_the_goal() -> None:
    kit = await _kit([LOOPING] * 3, streaming=True)
    first = Agent("first", provider=MockAIProvider(responses=["done"]))
    kit.register_channel(first)

    framed = await _supervisor_dispatch(
        kit,
        kit.channels["worker"],
        "parent",
        goal="Find the flight.",
        workers=[first],
        share_channels=None,
        task_timeout=10.0,
    )

    assert framed == "Find the flight."
    await kit.close()


async def test_the_cut_is_logged_once_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    kit = await _kit([LOOPING] * 3, streaming=True)

    with caplog.at_level(logging.DEBUG, logger="roomkit.tasks"):
        await kit.delegate("parent", "worker", "Find it.", wait=True)

    [record] = [r for r in caplog.records if "failed" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert record.exc_info is None
    await kit.close()


async def test_on_task_completed_names_the_turns_end() -> None:
    kit = await _kit([LOOPING] * 3, streaming=True)
    seen: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def completed(event: Any, ctx: Any) -> None:
        seen.append(event.metadata)

    await kit.delegate("parent", "worker", "Find it.", wait=True)
    await asyncio.sleep(0.05)

    [metadata] = seen
    assert (metadata["task_status"], metadata["loop_end_reason"]) == ("failed", "max_rounds")
    await kit.close()


def test_the_error_names_the_turns_end() -> None:
    error = TaskCutShortError("budget_exceeded", "Halfway there.")

    assert (error.reason, error.narration) == ("budget_exceeded", "Halfway there.")
    assert "budget_exceeded" in str(error)
