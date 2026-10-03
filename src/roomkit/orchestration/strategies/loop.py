"""Loop orchestration strategy.

An agent produces output, reviewers evaluate it, and the cycle
repeats until all reviewers approve or max iterations are reached.
The framework controls the flow — agents just produce content.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import orchestration_tool, schema_tool
from roomkit.core.exceptions import RoomKitError, TaskCutShortError
from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration._call_room import in_call_room
from roomkit.orchestration._installs import set_up_for_voice_room
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.state import (
    ConversationState,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.orchestration.status_bus import StatusLevel, post_agent_lifecycle
from roomkit.orchestration.strategies.supervisor import WorkerStrategy
from roomkit.tasks.models import task_cut_reason, task_work
from roomkit.tools.context import _current_turn_chain_depth

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.channels.ai import ToolResult
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

logger = logging.getLogger("roomkit.orchestration.strategies.loop")


class Loop(Orchestration):
    """Loop orchestration strategy.

    The producing agent generates output, then reviewers evaluate it.
    If all reviewers approve, the loop ends. Otherwise, feedback is
    routed back to the producer for revision.

    Examples::

        # Single reviewer
        Loop(agent=writer, reviewers=[editor], max_iterations=3)

        # Multiple reviewers — sequential (chained)
        Loop(
            agent=coder,
            reviewers=[security, perf, style],
            strategy="sequential",
        )

        # Multiple reviewers — parallel (fan-out)
        Loop(
            agent=coder,
            reviewers=[security, perf, style],
            strategy="parallel",
        )

        # Voice — async delivery
        Loop(
            agent=writer,
            reviewers=[editor],
            async_delivery=True,
        )
    """

    def __init__(
        self,
        agent: Agent,
        reviewers: list[Agent] | None = None,
        reviewer: Agent | None = None,
        max_iterations: int = 3,
        *,
        strategy: WorkerStrategy | str | None = None,
        async_delivery: bool = False,
    ) -> None:
        """Initialise the loop strategy.

        Args:
            agent: The producing agent.
            reviewers: List of reviewing agents. For multiple reviewers,
                use *strategy* to control execution order.
            reviewer: Single reviewer (convenience, same as
                ``reviewers=[reviewer]``).
            max_iterations: Maximum number of produce-review cycles.
            strategy: How reviewers execute when there are multiple:

                - ``"sequential"``: reviewers chain — each sees the
                  previous reviewer's feedback.
                - ``"parallel"``: reviewers fan-out — all review
                  independently, feedback combined.
                - ``None`` (default): sequential for multiple reviewers,
                  single reviewer doesn't need a strategy.

            async_delivery: If ``True``, the loop runs in the background
                and results are delivered via ``kit.deliver()`` when
                ready. The conversation continues uninterrupted.
        """
        self._agent = agent

        # Accept either reviewers=[...] or reviewer=single
        if reviewers and reviewer:
            msg = "Provide either 'reviewers' or 'reviewer', not both"
            raise ValueError(msg)
        if reviewer:
            self._reviewers = [reviewer]
        elif reviewers:
            self._reviewers = list(reviewers)
        else:
            msg = "At least one reviewer is required"
            raise ValueError(msg)

        self._max_iterations = max_iterations
        self._strategy = WorkerStrategy(strategy) if strategy else None
        self._async_delivery = async_delivery

    def agents(self) -> list[Agent]:
        """Return the producer — it presents results to the user."""
        if self._async_delivery:
            return []
        return [self._agent]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire the framework-driven loop."""
        producer = self._agent
        reviewers = self._reviewers
        max_iter = self._max_iterations
        async_delivery = self._async_delivery

        # Register all reviewers on the kit (not attached to room)
        for rev in reviewers:
            if rev.channel_id not in kit.channels:
                kit.register_channel(rev)

        # Also register producer if async (not attached to room)
        if async_delivery and producer.channel_id not in kit.channels:
            kit.register_channel(producer)

        if async_delivery:
            self._install_async_loop(kit, room_id)
        else:
            self._install_sync_loop(kit, room_id)

        # Set initial state
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase=producer.channel_id,
            active_agent_id=producer.channel_id,
            context={
                "_loop_iteration": 0,
                "_loop_approved": False,
                "_loop_max_iterations": max_iter,
            },
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)

    def _install_sync_loop(self, kit: RoomKit, room_id: str) -> None:
        """Take the producer's turns in *room_id* with this loop.

        The producer serves every room it is attached to (RFC §19.7): the loop
        runs in the room it was installed in, with this install's reviewers
        and limits, and the producer answers as itself everywhere else.
        """
        turns = _LoopTurns(kit, self._agent, self._reviewers, self._strategy, self._max_iterations)
        self._agent._registry.set_turn_runner(room_id, turns.run, owner=self)

    # -- Async delivery (voice) -----------------------------------------------

    def _install_async_loop(self, kit: RoomKit, room_id: str) -> None:
        """Serve ``delegate_loop`` in *room_id*'s realtime sessions.

        The tool runs this install's loop in the background for the room of
        the call (RFC §19.7): another room's sessions do not declare it, and a
        second room's install serves its own reviewers.
        """
        tool = schema_tool(_loop_tool(self._reviewers))
        server = _VoiceLoopServer(
            kit, self._agent, self._reviewers, self._strategy, self._max_iterations
        )
        entry = orchestration_tool(tool, in_call_room(tool.name, server.serve), waits=True)
        set_up_for_voice_room(kit, room_id, self, lambda _channel: entry)


class _LoopTurns:
    """One loop install's turns: the producer produces, the reviewers review."""

    def __init__(
        self,
        kit: RoomKit,
        producer: Agent,
        reviewers: list[Agent],
        strategy: WorkerStrategy | None,
        max_iterations: int,
    ) -> None:
        self._kit = kit
        self._producer = producer
        self._reviewers = reviewers
        self._strategy = strategy
        self._max_iterations = max_iterations

    async def run(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Take one of the producer's turns in the installed room."""
        producer = self._producer
        if event.source.channel_id == producer.channel_id:
            return ChannelOutput.empty()
        if event.source.channel_type in (ChannelType.AI, ChannelType.SYSTEM):
            return await producer._respond(event, binding, context)
        return await _run_loop(
            kit=self._kit,
            room_id=context.room.id if context.room else event.room_id,
            producer=producer,
            reviewers=self._reviewers,
            strategy=self._strategy,
            event=event,
            max_iterations=self._max_iterations,
        )


def _loop_tool(reviewers: list[Agent]) -> dict[str, Any]:
    """The ``delegate_loop`` declaration a voice channel carries."""
    reviewer_roles = ", ".join(getattr(r, "role", None) or r.channel_id for r in reviewers)
    return {
        "name": "delegate_loop",
        "description": (
            f"Submit work for review by specialists ({reviewer_roles}). "
            f"The producer will create content and reviewers will evaluate it. "
            f"Pass the topic or task description."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The topic or task",
                },
            },
            "required": ["task"],
        },
    }


class _VoiceLoopServer:
    """Serves a voice channel's ``delegate_loop``: runs the loop in the
    background for the room of the call, and answers at once."""

    def __init__(
        self,
        kit: RoomKit,
        producer: Agent,
        reviewers: list[Agent],
        strategy: WorkerStrategy | None,
        max_iterations: int,
    ) -> None:
        self._kit = kit
        self._producer = producer
        self._reviewers = reviewers
        self._strategy = strategy
        self._max_iterations = max_iterations
        self._running: set[str] = set()  # rooms whose loop is running

    async def serve(self, rid: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Answer one ``delegate_loop`` call made in room *rid*."""
        running = self._running
        if rid in running:
            return json.dumps({"status": "already_running", "message": "Loop is already running."})
        running.add(rid)
        # If create_task raises (shutdown race), release the room so it
        # isn't stuck in already_running.
        try:
            task = asyncio.create_task(
                _async_loop_and_deliver(
                    kit=self._kit,
                    room_id=rid,
                    producer=self._producer,
                    reviewers=self._reviewers,
                    strategy=self._strategy,
                    task_desc=arguments.get("task", ""),
                    max_iterations=self._max_iterations,
                    on_done=lambda: running.discard(rid),
                )
            )
        except BaseException:
            running.discard(rid)
            raise
        task.add_done_callback(log_task_exception)
        return json.dumps(
            {
                "status": "started",
                "message": "Loop is running. Results will be delivered when ready.",
            }
        )


# ---------------------------------------------------------------------------
# Loop execution
# ---------------------------------------------------------------------------


async def _run_loop(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    event: RoomEvent,
    max_iterations: int,
) -> ChannelOutput:
    """Run the full produce/review loop using child rooms."""
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body
    if not user_message:
        return ChannelOutput.empty()

    outcome = await _execute_loop(
        kit=kit,
        room_id=room_id,
        producer=producer,
        reviewers=reviewers,
        strategy=strategy,
        task_desc=user_message,
        max_iterations=max_iterations,
    )
    # The producer's failure reaches the caller whether an earlier output goes
    # out or not (RFC §19.7.4, §23.3).
    failure = _producer_failure(outcome) if outcome.stopped == "producer_failed" else None
    if not outcome.output:
        # No output at all: the turn has no answer, and the caller reads why.
        return ChannelOutput(responded=False, error=failure)

    # The producer's response to the event, one deeper (RFC §8.3, §19.7.4).
    result_event = RoomEvent(
        room_id=room_id,
        type=event.type,
        source=EventSource(
            channel_id=producer.channel_id,
            channel_type=ChannelType.AI,
        ),
        content=TextContent(body=outcome.output),
        chain_depth=event.chain_depth + 1,
        parent_event_id=event.parent_event_id,
        metadata={
            "approved": outcome.approved,
            "iteration": outcome.iteration,
            "stopped": outcome.stopped,
        },
    )
    return ChannelOutput(responded=True, response_events=[result_event], error=failure)


@dataclass
class _LoopOutcome:
    """How a loop ended (RFC §19.7.4): ``stopped`` is ``approved``,
    ``max_iterations`` or ``producer_failed``; ``output`` is the last output
    reviewed, from ``iteration``, the last iteration completed; ``failure`` is
    the producer's failed task, when one stopped the loop."""

    approved: bool = False
    iteration: int = 0
    output: str = ""
    stopped: str = "max_iterations"
    failure: DelegatedTaskResult | None = None

    @property
    def cut_reason(self) -> str | None:
        """How the failed producer's turn was cut short, when it was."""
        return _cut_reason(self.failure)


def _cut_reason(result: DelegatedTaskResult | None) -> str | None:
    """How a task's turn was cut short, when a cut ended it (RFC §23.3)."""
    return task_cut_reason(result)


def _producer_failure(outcome: _LoopOutcome) -> Exception:
    """The producer's failure, as the loop's caller reads it: its cut, or its
    task's error."""
    if reason := outcome.cut_reason:
        return TaskCutShortError(reason, None)
    error = outcome.failure.error if outcome.failure is not None else None
    return RoomKitError(
        f"The producer's task failed: {error}" if error else "The producer's task gave no output"
    )


def _failed_task_text(result: DelegatedTaskResult | None) -> str:
    """A producer's failed task, named without its error (RFC §23.3 step 8)."""
    reason = _cut_reason(result)
    return (
        f"the producer's task failed (cut short: {reason})"
        if reason
        else ("the producer's task failed")
    )


def _delivered_text(outcome: _LoopOutcome) -> str:
    """What an async loop delivers: how it ended, then its output."""
    if outcome.stopped != "producer_failed":
        status = "approved" if outcome.approved else "max iterations reached"
        return f"The review loop has completed ({status}).\n\n{outcome.output}"
    why = _failed_task_text(outcome.failure)
    if not outcome.output:
        return f"The review loop stopped before any output: {why}."
    return f"The review loop stopped: {why}. Its last output, not approved:\n\n{outcome.output}"


async def _async_loop_and_deliver(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    task_desc: str,
    max_iterations: int,
    on_done: Any,
) -> None:
    """Background: run loop → deliver results via kit.deliver().

    Started as a task by the tool call that asked for the loop, so the context
    it copied is that call's (RFC §21.4): the results continue the chain of
    the turn that made it (§23.3).
    """
    chain_depth = _current_turn_chain_depth()
    try:
        outcome = await _execute_loop(
            kit=kit,
            room_id=room_id,
            producer=producer,
            reviewers=reviewers,
            strategy=strategy,
            task_desc=task_desc,
            max_iterations=max_iterations,
        )
        logger.info("[loop] Complete (%s), delivering results", outcome.stopped)
        await kit.deliver(room_id, _delivered_text(outcome), chain_depth=chain_depth)
    except Exception:
        logger.exception("[loop] Async loop failed")
    finally:
        on_done()


async def _execute_loop(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    task_desc: str,
    max_iterations: int,
) -> _LoopOutcome:
    """Core loop logic shared by sync and async modes: produce, review,
    revise, until approved, out of iterations, or the producer's task fails."""
    outcome = _LoopOutcome()
    current_input = task_desc

    for iteration in range(1, max_iterations + 1):
        logger.info("[loop] Iteration %d/%d — producer", iteration, max_iterations)
        delegated = await _produce(
            kit, room_id, producer, current_input, iteration, max_iterations
        )
        producer_output = task_work(delegated.result)
        if not producer_output:
            logger.info("[loop] The producer's task failed at iteration %d: stopping", iteration)
            outcome.stopped, outcome.failure = "producer_failed", delegated.result
            break
        outcome.iteration, outcome.output = iteration, producer_output

        logger.info("[loop] Iteration %d/%d — reviewers", iteration, max_iterations)
        review_results = await _run_reviewers(kit, room_id, reviewers, strategy, producer_output)
        if all(r["approved"] for r in review_results):
            outcome.approved, outcome.stopped = True, "approved"
            logger.info("[loop] All reviewers approved at iteration %d", iteration)
            break
        current_input = _revision_prompt(producer_output, review_results)

    await _save_loop_state(kit, room_id, outcome)
    return outcome


async def _produce(
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    task: str,
    iteration: int,
    max_iterations: int,
) -> DelegatedTask:
    """One producer iteration: its task delegated, with its status posts."""
    metadata = {"room_id": room_id, "role": "producer", "iteration": iteration}
    post_agent_lifecycle(
        kit,
        producer.channel_id,
        StatusLevel.PENDING,
        action="iteration",
        detail=task,
        metadata={**metadata, "max_iterations": max_iterations},
    )
    try:
        delegated = await kit.delegate(room_id, producer.channel_id, task, wait=True)
    except Exception as exc:
        post_agent_lifecycle(
            kit,
            producer.channel_id,
            StatusLevel.FAILED,
            action="iteration",
            detail=str(exc),
            metadata=metadata,
        )
        raise
    output = task_work(delegated.result)
    post_agent_lifecycle(
        kit,
        producer.channel_id,
        StatusLevel.COMPLETED if output else StatusLevel.FAILED,
        action="iteration",
        detail=output or _failed_task_text(delegated.result).capitalize() + ".",
        metadata={**metadata, "task_id": delegated.id},
    )
    return delegated


def _revision_prompt(producer_output: str, review_results: list[dict[str, Any]]) -> str:
    """The producer's next task: its output, with the feedback of every
    reviewer who did not approve it."""
    combined_feedback = "\n\n".join(
        f"[{r['reviewer']}]: {r['feedback']}" for r in review_results if not r["approved"]
    )
    return (
        f"Revise your previous work based on this feedback:\n\n"
        f"--- Your previous output ---\n{producer_output}\n\n"
        f"--- Reviewer feedback ---\n{combined_feedback}"
    )


async def _save_loop_state(kit: RoomKit, room_id: str, outcome: _LoopOutcome) -> None:
    """Record how the loop ended on the room's conversation state."""
    room = await kit.get_room(room_id)
    state = get_conversation_state(room)
    ctx = dict(state.context)
    ctx["_loop_approved"] = outcome.approved
    ctx["_loop_iteration"] = outcome.iteration
    ctx["_loop_stopped"] = outcome.stopped
    state = state.model_copy(update={"context": ctx})
    await kit.store.update_room(set_conversation_state(room, state))


async def _run_reviewers(
    kit: RoomKit,
    room_id: str,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    producer_output: str,
) -> list[dict[str, Any]]:
    """Run reviewers according to strategy."""
    review_prompt = (
        "Review the following content and decide if it meets quality standards.\n"
        "If approved, your response MUST contain the word APPROVED.\n"
        "If not approved, provide specific feedback for revision.\n\n"
        f"--- Content to review ---\n{producer_output}"
    )

    if len(reviewers) == 1 or strategy != WorkerStrategy.PARALLEL:
        # Sequential: each reviewer sees the content (+ previous feedback)
        return await _review_sequential(kit, room_id, reviewers, review_prompt)

    # Parallel: all reviewers see the same content
    return await _review_parallel(kit, room_id, reviewers, review_prompt)


async def _review_sequential(
    kit: RoomKit,
    room_id: str,
    reviewers: list[Agent],
    review_input: str,
) -> list[dict[str, Any]]:
    """Run reviewers sequentially — each sees previous feedback."""
    results: list[dict[str, Any]] = []
    current_input = review_input

    for reviewer in reviewers:
        post_agent_lifecycle(
            kit,
            reviewer.channel_id,
            StatusLevel.PENDING,
            action="review",
            detail=current_input,
            metadata={"room_id": room_id, "role": "reviewer", "strategy": "sequential"},
        )
        try:
            delegated = await kit.delegate(room_id, reviewer.channel_id, current_input, wait=True)
        except Exception as exc:
            post_agent_lifecycle(
                kit,
                reviewer.channel_id,
                StatusLevel.FAILED,
                action="review",
                detail=str(exc),
                metadata={"room_id": room_id, "role": "reviewer", "strategy": "sequential"},
            )
            raise
        output = task_work(delegated.result)
        is_approved = "APPROVED" in output.upper() if output else False

        name = getattr(reviewer, "role", None) or reviewer.channel_id
        post_agent_lifecycle(
            kit,
            reviewer.channel_id,
            StatusLevel.COMPLETED if is_approved else StatusLevel.INFO,
            action="review",
            detail=output,
            metadata={
                "room_id": room_id,
                "role": "reviewer",
                "strategy": "sequential",
                "approved": is_approved,
                "task_id": delegated.id,
            },
        )
        results.append(
            {
                "reviewer": name,
                "approved": is_approved,
                "feedback": output,
            }
        )

        # Next reviewer sees previous feedback appended
        if not is_approved and output:
            current_input = f"{current_input}\n\n--- {name} feedback ---\n{output}"

    return results


async def _review_parallel(
    kit: RoomKit,
    room_id: str,
    reviewers: list[Agent],
    review_input: str,
) -> list[dict[str, Any]]:
    """Run all reviewers in parallel on the same content."""

    async def _review_one(reviewer: Agent) -> dict[str, Any]:
        post_agent_lifecycle(
            kit,
            reviewer.channel_id,
            StatusLevel.PENDING,
            action="review",
            detail=review_input,
            metadata={"room_id": room_id, "role": "reviewer", "strategy": "parallel"},
        )
        try:
            delegated = await kit.delegate(room_id, reviewer.channel_id, review_input, wait=True)
        except Exception as exc:
            post_agent_lifecycle(
                kit,
                reviewer.channel_id,
                StatusLevel.FAILED,
                action="review",
                detail=str(exc),
                metadata={"room_id": room_id, "role": "reviewer", "strategy": "parallel"},
            )
            raise
        output = task_work(delegated.result)
        is_approved = "APPROVED" in output.upper() if output else False
        name = getattr(reviewer, "role", None) or reviewer.channel_id
        post_agent_lifecycle(
            kit,
            reviewer.channel_id,
            StatusLevel.COMPLETED if is_approved else StatusLevel.INFO,
            action="review",
            detail=output,
            metadata={
                "room_id": room_id,
                "role": "reviewer",
                "strategy": "parallel",
                "approved": is_approved,
                "task_id": delegated.id,
            },
        )
        return {"reviewer": name, "approved": is_approved, "feedback": output}

    results = await asyncio.gather(*[_review_one(r) for r in reviewers])
    return list(results)
