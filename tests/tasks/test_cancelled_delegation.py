"""A delegation cancelled from outside ends as every task ends (RMK-434, RFC §23.3).

Inline (its caller's timeout) or in the background (the runner's ``cancel``
or ``close``, even before the task ran a line): the task ends ``cancelled``,
``ON_TASK_COMPLETED`` and its completion callback run, the notified agent is
told, and the delegation span ends with the task's status. A supervisor's
worker is free again once its task was cancelled.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ChannelCategory, HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import TaskStatus
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.orchestration.strategies.supervisor.execution import _run_sequential
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
CALL = AIResponse(
    content="Checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


async def _slow(name: str, arguments: dict[str, Any]) -> str:
    await asyncio.sleep(5)
    return "found"


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _kit(
    responses: list[AIResponse] | None = None, handler: Any = _slow
) -> tuple[RoomKit, MockTelemetryProvider, list[Any]]:
    telemetry = MockTelemetryProvider()
    kit = RoomKit(telemetry=telemetry)
    kit.register_channel(
        Agent(
            "worker",
            provider=MockAIProvider(ai_responses=responses or [CALL] * 9, streaming=True),
            tools=[LOOKUP],
            tool_handler=handler,
            tool_search=False,
            max_tool_rounds=3,
        )
    )
    await kit.create_room(room_id="p")
    completed: list[Any] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC, name="completed")
    async def on_completed(event: Any, ctx: Any) -> None:
        completed.append((event.metadata.get("task_status"), event.metadata.get("error")))

    return kit, telemetry, completed


def _delegation_spans(telemetry: MockTelemetryProvider) -> list[tuple[str, str]]:
    assert not [s for s in telemetry.get_active_spans() if s.kind == SpanKind.DELEGATION]
    return [(s.name, s.status) for s in telemetry.completed_spans if s.kind == SpanKind.DELEGATION]


async def _settle(completed: list[Any]) -> None:
    for _ in range(100):
        if completed:
            return
        await asyncio.sleep(0.01)


async def test_an_inline_delegation_its_caller_times_out_ends_cancelled() -> None:
    kit, telemetry, completed = await _kit()
    ended: list[DelegatedTaskResult] = []

    async def on_complete(result: DelegatedTaskResult) -> None:
        ended.append(result)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            kit.delegate("p", "worker", "go", wait=True, on_complete=on_complete), 0.2
        )
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert [r.status for r in ended] == [TaskStatus.CANCELLED]
    assert _delegation_spans(telemetry) == [("delegation.inline", "cancelled")]
    await kit.close()


async def test_a_background_delegation_the_runner_cancels_ends_cancelled() -> None:
    kit, telemetry, completed = await _kit()
    ended: list[DelegatedTaskResult] = []

    async def on_complete(result: DelegatedTaskResult) -> None:
        ended.append(result)

    task = await kit.delegate("p", "worker", "go", on_complete=on_complete)
    await asyncio.sleep(0.2)

    assert await kit.task_runner.cancel(task.id) is True
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert [r.status for r in ended] == [TaskStatus.CANCELLED]
    assert task.result is not None and task.result.error == "cancelled"
    assert _delegation_spans(telemetry) == [("delegation.background", "cancelled")]
    await kit.close()


async def test_a_task_cancelled_before_it_ran_a_line_still_ends() -> None:
    kit, telemetry, completed = await _kit()

    task = await kit.delegate("p", "worker", "go")
    await kit.task_runner.cancel(task.id)

    assert (await task.wait(timeout=1)).status == TaskStatus.CANCELLED
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert _delegation_spans(telemetry) == [("delegation.background", "cancelled")]
    await kit.close()


async def test_closing_the_runner_ends_its_tasks_cancelled() -> None:
    kit, _, completed = await _kit()
    task = await kit.delegate("p", "worker", "go")
    await asyncio.sleep(0.2)

    await kit.task_runner.close()

    assert task.status == TaskStatus.CANCELLED
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    await kit.close()


async def test_the_notified_agent_is_told_its_task_was_cancelled() -> None:
    kit, _, completed = await _kit()
    notified = AIChannel("assistant", provider=MockAIProvider(responses=["ok"]))
    kit.register_channel(notified)
    kit.register_channel(SimpleChannel("phone"))
    await kit.attach_channel("p", "phone")
    await kit.attach_channel("p", "assistant", category=ChannelCategory.INTELLIGENCE)
    task = await kit.delegate("p", "worker", "go", notify="assistant")
    await asyncio.sleep(0.2)

    await kit.task_runner.cancel(task.id)
    for _ in range(100):
        if notified._provider.calls:
            break
        await asyncio.sleep(0.01)

    [call] = notified._provider.calls
    assert "[Background task from worker cancelled." in str(call.messages[-1].content)
    await kit.close()


@pytest.mark.parametrize(
    ("responses", "span_status"),
    [([AIResponse(content="Done.")], "ok"), ([CALL] * 9, "error")],
)
async def test_a_delegation_span_ends_with_its_task_status(
    responses: list[AIResponse], span_status: str
) -> None:
    kit, telemetry, _ = await _kit(responses, _found)

    await kit.delegate("p", "worker", "go", wait=True)

    assert _delegation_spans(telemetry) == [("delegation.inline", span_status)]
    await kit.close()


def _delegating(n: int) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"d{n}", name="delegate_to_worker", arguments={"task": "go"})],
    )


async def test_a_supervisors_worker_is_free_once_its_task_was_cancelled() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    supervisor = Agent(
        "sup",
        provider=MockAIProvider(
            ai_responses=[
                _delegating(1),
                AIResponse(content="Started."),
                _delegating(2),
                AIResponse(content="Started again."),
            ],
            streaming=True,
        ),
        tool_search=False,
    )
    worker = Agent(
        "worker",
        provider=MockAIProvider(ai_responses=[CALL], streaming=True),
        tools=[LOOKUP],
        tool_handler=_slow,
        tool_search=False,
    )
    kit.register_channel(supervisor)
    kit.register_channel(worker)
    delegated: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC, name="delegated")
    async def on_delegated(event: Any, ctx: Any) -> None:
        delegated.append(event.metadata["task_id"])

    await kit.create_room(
        room_id="r",
        orchestration=Supervisor(supervisor=supervisor, workers=[worker], wait_for_result=False),
    )
    await kit.attach_channel("r", "sms")

    async def ask(text: str) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=text))
        )
        await asyncio.sleep(0.2)

    await ask("Find it.")
    await kit.task_runner.cancel(delegated[0])
    await ask("Try again.")

    # The second call delegated anew, the worker no longer "already running".
    assert len(delegated) == 2
    await kit.close()


async def test_a_supervisors_task_timeout_ends_the_task_cancelled() -> None:
    kit, telemetry, completed = await _kit()

    await _run_sequential(kit, "p", [kit.channels["worker"]], "go", task_timeout=0.2)  # type: ignore[list-item]
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert _delegation_spans(telemetry) == [("delegation.inline", "cancelled")]
    await kit.close()


async def test_a_cancel_cut_while_the_task_ends_still_ends_it() -> None:
    kit, _, completed = await _kit()

    async def slow_on_complete(result: DelegatedTaskResult) -> None:
        await asyncio.sleep(0.2)

    task = await kit.delegate("p", "worker", "go", on_complete=slow_on_complete)
    await asyncio.sleep(0.1)
    cancelling = asyncio.create_task(kit.task_runner.cancel(task.id))
    await asyncio.sleep(0.05)
    cancelling.cancel()

    assert (await task.wait(timeout=1)).status == TaskStatus.CANCELLED
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    await kit.close()
