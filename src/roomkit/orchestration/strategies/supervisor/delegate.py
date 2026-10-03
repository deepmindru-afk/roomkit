"""Delegation entry points that wire worker execution to the supervisor.

The framework-driven auto-delegate helpers (one-pass / two-pass), the
background runner that hands results back to the supervisor, and the strategy
dispatcher that routes to the supervised, sequential, or parallel runner.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.core._failure_log import log_failure
from roomkit.core._fallback import FALLBACK_FAILED
from roomkit.core.event_router import StreamingResponse
from roomkit.core.mixins._child_execution import persist_tool_calls
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.enums import EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_MAX_REVISIONS,
    _DEFAULT_TASK_TIMEOUT_SECONDS,
    WorkerStrategy,
    _post_worker_status,
    logger,
)
from roomkit.orchestration.strategies.supervisor.execution import (
    _run_parallel,
    _run_sequential,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _extract_output_text,
    _format_worker_results,
    _present_worker_results,
)
from roomkit.orchestration.strategies.supervisor.supervised import (
    _run_supervised_sequential,
)
from roomkit.tasks.handback import bounded, hand_back, result_text
from roomkit.tools.context import _current_turn_chain_depth

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


def _build_pass1_instruction() -> str:
    """Build the default pass-1 instruction."""
    return (
        "Extract the core topic or subject from the user's request. "
        "Output only the topic, nothing else. No questions, no instructions, "
        "no formatting. Example: user says 'analyse anthropic' → 'Anthropic'"
    )


async def _async_run_and_deliver(
    *,
    kit: RoomKit,
    room_id: str,
    supervisor_id: str,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    task_desc: str,
    share_channels: list[str] | None = None,
    on_done: Callable[..., None],
) -> None:
    """Background: run workers → hand their results back to *supervisor_id*.

    Started as a task by the tool call that dispatched the workers, so the
    context it copied is that call's (RFC §21.4): the results continue the
    chain of the turn that made it (§23.3), and a supervisor re-dispatching on
    every result stops at ``max_chain_depth``.

    Individual worker lifecycle events are posted to ``kit.status_bus``
    inside ``_run_sequential`` / ``_run_parallel``. This helper emits
    one additional terminal entry under ``agent_id="orchestration"``
    so subscribers can observe the pipeline as a whole.

    ``on_done`` is called in ``finally`` with ``success=<bool>`` regardless
    of outcome, so callers can distinguish success from failure — e.g. to
    evict cached dispatch responses that should not be re-served after a
    failed pipeline.
    """
    chain_depth = _current_turn_chain_depth()
    pipeline_meta = {
        "room_id": room_id,
        "strategy": str(strategy) if strategy else None,
        "workers": [w.channel_id for w in workers],
    }
    pipeline_success = False
    try:
        worker_results = await _run_workers(
            kit,
            room_id,
            strategy,
            workers,
            task_desc,
            share_channels=share_channels,
        )
        await _deliver_worker_results(kit, room_id, supervisor_id, worker_results, chain_depth)
        _post_worker_status(
            kit,
            "orchestration",
            StatusLevel.COMPLETED,
            action="pipeline",
            detail=f"{len(workers)} worker(s) completed",
            metadata=pipeline_meta,
        )
        pipeline_success = True
    except Exception as exc:
        logger.exception("[async_delegate] Pipeline failed")
        _post_worker_status(
            kit,
            "orchestration",
            StatusLevel.FAILED,
            action="pipeline",
            detail=str(exc),
            metadata=pipeline_meta,
        )
    finally:
        on_done(success=pipeline_success)


async def _deliver_worker_results(
    kit: RoomKit,
    room_id: str,
    supervisor_id: str,
    worker_results: list[dict[str, Any]],
    chain_depth: int,
) -> None:
    """Hand the workers' results back to the supervisor, at the dispatching turn's depth.

    As a background delegation's result is (RFC §19.7.3, §23.3): an instruction
    addressed to the supervisor, each worker's output bounded.
    """
    each_bounded = [{**r, "output": bounded(str(r.get("output") or ""))} for r in worker_results]
    logger.info("[async_delegate] Workers completed, handing results back")
    text = result_text(
        "[Your background workers completed. Share their results with the user.]",
        _format_worker_results(each_bounded),
    )
    await hand_back(kit, room_id, supervisor_id, text, chain_depth)


def _results_event(event: RoomEvent, body: str) -> RoomEvent:
    """The workers' results, standing in for the event the supervisor answers.

    As deep as the event and in its thread, so the supervisor's answer stays
    one deeper than the event it answers (RFC §8.3, §19.7.3).
    """
    return RoomEvent(
        room_id=event.room_id,
        type=event.type,
        source=EventSource(channel_id="system", channel_type=_ChannelType.SYSTEM),
        content=TextContent(body=body),
        chain_depth=event.chain_depth,
        parent_event_id=event.parent_event_id,
    )


async def _run_workers(
    kit: RoomKit,
    room_id: str,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    task_desc: str,
    *,
    supervisor: Agent | None = None,
    max_revisions: int = _DEFAULT_MAX_REVISIONS,
    share_channels: list[str] | None = None,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run workers according to strategy and return their reviewed results.

    Sequential goes through the supervised hub-&-spoke loop when a *supervisor*
    is given (every output returns to the supervisor, which validates it and
    frames the next worker's task); parallel runs all workers on the same task.
    """
    if strategy == WorkerStrategy.SEQUENTIAL and supervisor is not None:
        return await _run_supervised_sequential(
            kit,
            room_id,
            supervisor,
            workers,
            task_desc,
            max_revisions=max_revisions,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
    if strategy == WorkerStrategy.SEQUENTIAL:
        result_json = await _run_sequential(
            kit,
            room_id,
            workers,
            task_desc,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
    else:
        result_json = await _run_parallel(
            kit,
            room_id,
            workers,
            task_desc,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
    parsed = json.loads(result_json)
    return parsed.get("results", [])


async def _formulate_task(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    original_on_event: Any,
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    instruction: str | None,
) -> _Pass1:
    """Pass 1: the supervisor turns the request into a task for its workers.

    The task-formulation instruction rides this call only, on a copy of the
    binding whose prompt is the one the turn would have had followed by the
    instruction (RFC §19.7.3). The supervisor serves every room it is attached
    to: its own prompt is never the carrier, so a room running meanwhile never
    reads another's instruction.
    """
    pass1_instruction = instruction or _build_pass1_instruction()
    _, settings = await supervisor._resolve_turn(binding, context)
    base = settings.get("system_prompt")
    prompt = f"{base}\n\n{pass1_instruction}" if base else pass1_instruction
    pass1_binding = binding.model_copy(
        update={"metadata": {**binding.metadata, "system_prompt": prompt}}
    )
    pass1_output = await original_on_event(event, pass1_binding, context)
    return await _pass1_task(kit, room_id, supervisor, event, pass1_output, context)


@dataclass
class _Pass1:
    """What the task-formulation pass gave: its output, the task it hands
    the workers, and how its turn ended."""

    output: ChannelOutput
    task: str = ""
    end: str | None = None


async def _pass1_task(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    event: RoomEvent,
    output: ChannelOutput,
    context: RoomContext,
) -> _Pass1:
    """The task pass 1 hands on: its final answer, as every streamed turn is
    read, its tool calls stored in the room as any turn's (RFC §19.7.3). A
    pass that failed hands its error to the turn's caller, who logs it."""
    if output.error is not None or output.response_stream is None:
        return _Pass1(output, await _extract_output_text(output))
    stream = StreamingResponse(
        stream=output.response_stream,
        source_channel_id=supervisor.channel_id,
        source_channel_type=supervisor.channel_type,
        trigger_event=event,
        response_metadata=output.response_metadata,
    )
    try:
        task, end = await persist_tool_calls(kit, room_id, stream, context)
    except Exception as exc:
        # Logged once, as a room turn's: the caller receives the error.
        log_failure(
            logger, exc, f"Pass 1 of {supervisor.channel_id} in room {room_id}", caller_logs=True
        )
        return _Pass1(output.model_copy(update={"error": exc}))
    return _Pass1(output, task, end)


def _pass1_answer(supervisor: Agent, event: RoomEvent, pass1: _Pass1) -> ChannelOutput:
    """What the room reads of a pass that handed on no task: the supervisor's
    fallback when the pass was cut short, so the message it answered gets an
    answer (RFC §19.7.3); else the pass's own output, its error included."""
    if pass1.end in (None, "completed"):
        return pass1.output
    fallback = RoomEvent(
        room_id=event.room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id=supervisor.channel_id, channel_type=_ChannelType.AI),
        content=TextContent(body=FALLBACK_FAILED),
        chain_depth=event.chain_depth + 1,
        parent_event_id=event.parent_event_id,
        metadata={"loop_end_reason": pass1.end},
    )
    return ChannelOutput(responded=True, response_events=[fallback])


async def _two_pass_delegate(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    original_on_event: Any,
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    *,
    instruction: str | None = None,
    share_channels: list[str] | None = None,
    max_revisions: int = _DEFAULT_MAX_REVISIONS,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> ChannelOutput:
    """Two-pass: supervisor formulates task → workers run (validated between
    steps by the supervisor in sequential mode) → supervisor presents."""
    pass1 = await _formulate_task(
        kit, room_id, supervisor, original_on_event, event, binding, context, instruction
    )
    refined_task = pass1.task

    logger.debug("Pass 1 refined task: %s", refined_task[:200] if refined_task else "(empty)")

    if not refined_task:
        return _pass1_answer(supervisor, event, pass1)

    # Run workers with the refined task — supervised between steps in sequential.
    worker_results = await _run_workers(
        kit,
        room_id,
        strategy,
        workers,
        refined_task,
        supervisor=supervisor,
        max_revisions=max_revisions,
        share_channels=share_channels,
        task_timeout=task_timeout,
    )

    # Pass 2: inject worker results and generate final response
    results_event = _results_event(event, _present_worker_results(worker_results))

    # Ingest the results so the supervisor sees them in context
    try:
        await supervisor._memory.ingest(
            event.room_id, results_event, channel_id=supervisor.channel_id
        )
    except Exception:
        logger.warning("Failed to ingest worker results", exc_info=True)

    return await original_on_event(results_event, binding, context)


async def _one_pass_delegate(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    original_on_event: Any,
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    *,
    share_channels: list[str] | None = None,
    max_revisions: int = _DEFAULT_MAX_REVISIONS,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> ChannelOutput:
    """One-pass: workers run on raw message (validated between steps by the
    supervisor in sequential mode) → supervisor presents."""
    # Extract user's raw message
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body

    if not user_message:
        return await original_on_event(event, binding, context)

    # Run workers with the raw user message — supervised between steps in sequential.
    worker_results = await _run_workers(
        kit,
        room_id,
        strategy,
        workers,
        user_message,
        supervisor=supervisor,
        max_revisions=max_revisions,
        share_channels=share_channels,
        task_timeout=task_timeout,
    )

    # Inject results into context and let supervisor present
    results_event = _results_event(
        event, f"The user asked: {user_message}\n\n{_present_worker_results(worker_results)}"
    )

    try:
        await supervisor._memory.ingest(
            event.room_id, results_event, channel_id=supervisor.channel_id
        )
    except Exception:
        logger.warning("Failed to ingest worker results", exc_info=True)

    return await original_on_event(results_event, binding, context)
