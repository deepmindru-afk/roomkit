"""Loop orchestration strategy.

An agent produces output, reviewers evaluate it, and the cycle
repeats until all reviewers approve or max iterations are reached.
The framework controls the flow — agents just produce content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration._call_room import call_room_handler
from roomkit.orchestration._installs import first_install
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.state import (
    ConversationState,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.orchestration.status_bus import StatusLevel, post_agent_lifecycle
from roomkit.orchestration.strategies.supervisor import WorkerStrategy
from roomkit.tools.context import _current_turn_chain_depth

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.channels.ai import ToolResult
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.loop")

# A loop installed in several rooms wires a voice channel once (RFC §19.7):
# the voice channels it serves ``delegate_loop`` on, with the rooms each one
# was installed in.
_LOOP_VOICE_SERVING: weakref.WeakSet[Any] = weakref.WeakSet()
_LOOP_VOICE_ROOMS: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()


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
        """Inject the ``delegate_loop`` tool into RealtimeVoiceChannel, once.

        The voice channel serves every room the loop is installed in (RFC
        §19.7): a second room's install adds its room, each call runs the loop
        for the room it came from, and a call from a room the loop was not
        installed in is refused, since the channel declares the same tools in
        every room.
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        producer = self._agent
        reviewers = self._reviewers
        max_iter = self._max_iterations
        strategy = self._strategy

        voice_channel: RealtimeVoiceChannel | None = None
        for ch in kit.channels.values():
            if isinstance(ch, RealtimeVoiceChannel):
                voice_channel = ch
                break

        if voice_channel is None:
            logger.warning("async_delivery=True but no RealtimeVoiceChannel found")
            return

        tool_def = _loop_tool(reviewers)

        rooms = _LOOP_VOICE_ROOMS.setdefault(voice_channel, set())
        rooms.add(room_id)
        if not first_install(_LOOP_VOICE_SERVING, voice_channel):
            return
        voice_channel._inject_orchestration_tool(tool_def)

        original_handler = voice_channel.tool_handler
        server = _VoiceLoopServer(kit, rooms, producer, reviewers, strategy, max_iter)
        voice_channel.tool_handler = call_room_handler(
            {"delegate_loop"}, server.serve, original_handler
        )


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
        rooms: set[str],
        producer: Agent,
        reviewers: list[Agent],
        strategy: WorkerStrategy | None,
        max_iterations: int,
    ) -> None:
        self._kit = kit
        self._rooms = rooms
        self._producer = producer
        self._reviewers = reviewers
        self._strategy = strategy
        self._max_iterations = max_iterations
        self._running: set[str] = set()  # rooms whose loop is running

    async def serve(self, rid: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Answer one ``delegate_loop`` call made in room *rid*."""
        if rid not in self._rooms:
            return json.dumps({"error": "delegate_loop is not available in this room"})
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

    result = await _execute_loop(
        kit=kit,
        room_id=room_id,
        producer=producer,
        reviewers=reviewers,
        strategy=strategy,
        task_desc=user_message,
        max_iterations=max_iterations,
    )

    # The producer's response to the event, one deeper (RFC §8.3, §19.7.4).
    result_event = RoomEvent(
        room_id=room_id,
        type=event.type,
        source=EventSource(
            channel_id=producer.channel_id,
            channel_type=ChannelType.AI,
        ),
        content=TextContent(body=result["output"]),
        chain_depth=event.chain_depth + 1,
        parent_event_id=event.parent_event_id,
        metadata={
            "approved": result["approved"],
            "iteration": result["iteration"],
        },
    )
    return ChannelOutput(responded=True, response_events=[result_event])


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
        result = await _execute_loop(
            kit=kit,
            room_id=room_id,
            producer=producer,
            reviewers=reviewers,
            strategy=strategy,
            task_desc=task_desc,
            max_iterations=max_iterations,
        )

        status = "approved" if result["approved"] else "max iterations reached"
        logger.info("[loop] Complete (%s), delivering results", status)

        await kit.deliver(
            room_id,
            f"The review loop has completed ({status}).\n\n{result['output']}",
            chain_depth=chain_depth,
        )
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
) -> dict[str, Any]:
    """Core loop logic shared by sync and async modes."""
    current_input = task_desc
    approved = False
    final_output = ""
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        logger.info("[loop] Iteration %d/%d — producer", iteration, max_iterations)

        post_agent_lifecycle(
            kit,
            producer.channel_id,
            StatusLevel.PENDING,
            action="iteration",
            detail=current_input,
            metadata={
                "room_id": room_id,
                "role": "producer",
                "iteration": iteration,
                "max_iterations": max_iterations,
            },
        )
        try:
            delegated = await kit.delegate(room_id, producer.channel_id, current_input, wait=True)
        except Exception as exc:
            post_agent_lifecycle(
                kit,
                producer.channel_id,
                StatusLevel.FAILED,
                action="iteration",
                detail=str(exc),
                metadata={
                    "room_id": room_id,
                    "role": "producer",
                    "iteration": iteration,
                },
            )
            raise
        producer_output = (delegated.result.output if delegated.result else "") or ""
        post_agent_lifecycle(
            kit,
            producer.channel_id,
            StatusLevel.COMPLETED if producer_output else StatusLevel.FAILED,
            action="iteration",
            detail=producer_output or "empty output",
            metadata={
                "room_id": room_id,
                "role": "producer",
                "iteration": iteration,
                "task_id": delegated.id,
            },
        )
        if not producer_output:
            logger.warning("[loop] Producer returned empty output")
            break

        # Run reviewers
        logger.info("[loop] Iteration %d/%d — reviewers", iteration, max_iterations)
        review_results = await _run_reviewers(kit, room_id, reviewers, strategy, producer_output)

        # Check if ALL reviewers approved
        all_approved = all(r["approved"] for r in review_results)
        if all_approved:
            approved = True
            final_output = producer_output
            logger.info("[loop] All reviewers approved at iteration %d", iteration)
            break

        # Combine feedback from reviewers who didn't approve
        feedback_parts = []
        for r in review_results:
            if not r["approved"]:
                reviewer_name = r["reviewer"]
                feedback_parts.append(f"[{reviewer_name}]: {r['feedback']}")

        combined_feedback = "\n\n".join(feedback_parts)
        current_input = (
            f"Revise your previous work based on this feedback:\n\n"
            f"--- Your previous output ---\n{producer_output}\n\n"
            f"--- Reviewer feedback ---\n{combined_feedback}"
        )
        final_output = producer_output

    # Update state
    room = await kit.get_room(room_id)
    state = get_conversation_state(room)
    ctx = dict(state.context)
    ctx["_loop_approved"] = approved
    ctx["_loop_iteration"] = iteration
    state = state.model_copy(update={"context": ctx})
    room = set_conversation_state(room, state)
    await kit.store.update_room(room)

    return {"approved": approved, "iteration": iteration, "output": final_output}


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
        output = (delegated.result.output if delegated.result else "") or ""
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
        output = (delegated.result.output if delegated.result else "") or ""
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
