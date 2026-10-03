"""A delegated worker that its round cap cuts has no answer (RMK-414).

RFC §23.3 step 6, RFC §6.4: the task fails, its error naming how the turn
ended, its output the worker's last narration and its metadata the
``loop_end_reason``; streamed or buffered, inline or in the background, with
a transport shared into the child room or not, whether the turn wrote text or
none. An ACP worker whose prompt stopped on any reason but ``end_turn`` is cut
too (RMK-418). A worker that owes a result and submitted it before the cut
keeps it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import acp
import pytest
from acp.schema import PromptResponse

from roomkit import HookExecution, HookTrigger, RoomKit, TaskCutShortError
from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, TaskStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.orchestration.strategies.loop import _execute_loop
from roomkit.orchestration.strategies.supervisor.results import _result_output
from roomkit.orchestration.strategies.supervisor.supervised import _supervisor_dispatch
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult, task_work
from tests.test_channels.test_acp import _channel
from tests.test_framework import AILikeChannel, SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)
SILENT_LOOPING = LOOPING.model_copy(update={"content": ""})


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


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "background"])
@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared-transport"])
async def test_a_task_cut_before_any_text_fails_naming_the_end(shared: bool, inline: bool) -> None:
    kit = await _kit([SILENT_LOOPING] * 3, streaming=True)
    seen: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def completed(event: Any, ctx: Any) -> None:
        seen.append(event.metadata)

    task = await kit.delegate(
        "parent",
        "worker",
        "Find it.",
        share_channels=["email-out"] if shared else None,
        wait=inline,
    )
    result = task.result if inline else await task.wait(timeout=5)
    await asyncio.sleep(0.05)

    assert result is not None
    assert (result.status, result.output) == (TaskStatus.FAILED, None)
    assert result.error == "The worker's turn ended max_rounds before its answer"
    assert result.metadata["loop_end_reason"] == "max_rounds"
    [metadata] = seen
    assert (metadata["error"], metadata["loop_end_reason"]) == (result.error, "max_rounds")
    await kit.close()


class _BufferedWorker(AILikeChannel):
    """A worker that answers buffered, its turn's end named on its record only."""

    def __init__(self, channel_id: str, narration: str | None) -> None:
        super().__init__(channel_id)
        self._narration = narration

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        rows = [] if self._narration is None else [_message(self.channel_id, self._narration)]
        return ChannelOutput(
            responded=True,
            response_events=rows,
            response_metadata=ResponseMetadata({"loop_end_reason": "timeout"}),
        )


def _message(channel_id: str, body: str) -> RoomEvent:
    return RoomEvent(
        room_id="child",
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.AI),
        content=TextContent(body=body),
    )


@pytest.mark.parametrize("narration", [None, "Halfway."], ids=["no-text", "narrated"])
async def test_a_buffered_worker_whose_record_names_a_cut_fails(narration: str | None) -> None:
    kit = RoomKit()
    kit.register_channel(_BufferedWorker("worker", narration))
    await kit.create_room(room_id="parent")

    task = await kit.delegate("parent", "worker", "Find it.", wait=True)

    assert task.result is not None
    assert (task.result.status, task.result.output) == (TaskStatus.FAILED, narration)
    assert task.result.error == "The worker's turn ended timeout before its answer"
    await kit.close()


def _acp_turn(stop_reason: str) -> Any:
    async def prompt(connection: Any, session_id: str, *args: Any, **kwargs: Any) -> Any:
        await connection.client.session_update(
            session_id, acp.update_agent_message_text("Let me look into that.")
        )
        return PromptResponse(stop_reason=stop_reason)

    return prompt


@pytest.mark.parametrize(
    "stop_reason", ["max_tokens", "max_turn_requests", "cancelled", "refusal"]
)
async def test_an_acp_worker_that_stopped_short_fails_with_its_narration(
    tmp_path: Any, stop_reason: str
) -> None:
    kit = RoomKit()
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    turn = _acp_turn(stop_reason)
    connection.prompt = lambda *a, **k: turn(connection, *a, **k)  # type: ignore[method-assign]
    kit.register_channel(channel)
    await kit.create_room(room_id="parent")

    task = await kit.delegate("parent", channel.channel_id, "Find it.", wait=True)

    assert task.result is not None
    assert (task.result.status, task.result.output) == (
        TaskStatus.FAILED,
        "Let me look into that.",
    )
    assert task.result.error == f"The worker's turn ended {stop_reason} before its answer"
    assert task.result.metadata["loop_end_reason"] == stop_reason
    await kit.close()


async def test_an_acp_worker_that_ended_its_turn_completes(tmp_path: Any) -> None:
    kit = RoomKit()
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    turn = _acp_turn("end_turn")
    connection.prompt = lambda *a, **k: turn(connection, *a, **k)  # type: ignore[method-assign]
    kit.register_channel(channel)
    await kit.create_room(room_id="parent")

    task = await kit.delegate("parent", channel.channel_id, "Find it.", wait=True)

    assert task.result is not None
    assert (task.result.status, task.result.output) == (
        TaskStatus.COMPLETED,
        "Let me look into that.",
    )
    await kit.close()


async def test_a_room_turn_with_no_text_reports_its_end_to_the_caller() -> None:
    kit = RoomKit()
    kit.register_channel(
        Agent(
            "agent",
            provider=MockAIProvider(ai_responses=[SILENT_LOOPING] * 3),
            tools=[LOOKUP],
            tool_handler=_found,
            tool_search=False,
            max_tool_rounds=1,
        )
    )
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Find it."))
    )

    assert result.response_metadata["loop_end_reason"] == "max_rounds"
    await kit.close()
