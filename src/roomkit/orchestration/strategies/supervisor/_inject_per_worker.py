"""Per-worker delegation wiring for the supervisor.

Mixin for :class:`Supervisor`: injects one ``delegate_to_<id>`` tool per worker
and lets the AI decide when to delegate (manual mode). Host attributes are
declared as annotations; they are set in ``Supervisor.__init__``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import orchestration_tool
from roomkit.orchestration._background import (
    BackgroundRun,
    background_failure_text,
    run_in_background,
    start_background_run,
)
from roomkit.orchestration._call_room import in_call_room
from roomkit.orchestration._worker_run import WorkerOutcome, WorkerStatus, run_worker
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _post_worker_status,
    logger,
)
from roomkit.providers.ai.base import AITool
from roomkit.tasks.handback import bounded, result_text

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


class _PerWorkerToolMixin:
    """Inject per-worker ``delegate_to_<id>`` tools (the AI decides)."""

    _supervisor: Agent
    _workers: list[Agent]
    _wait_for_result: bool
    _share_channels: list[str]
    _task_timeout: float

    def _inject_per_worker_tools(self, kit: RoomKit, room_id: str) -> None:
        """Declare per-worker ``delegate_to_<id>`` tools in *room_id*'s turns,
        and serve them there.

        The tools are set up for the installed room (RFC §19.7), not in every
        room the supervisor serves, and reach this install's workers with its
        settings; a call delegates from the room of the call (RFC §23.4).
        """
        tool_to_worker = {f"delegate_to_{w.channel_id}": w.channel_id for w in self._workers}
        server = _PerWorkerToolServer(
            kit,
            self._supervisor,
            tool_to_worker,
            wait=self._wait_for_result,
            share_channels=self._share_channels,
            task_timeout=self._task_timeout,
        )
        entries = [
            orchestration_tool(tool, in_call_room(tool.name, server.serve), waits=True)
            for tool in map(_worker_tool, self._workers)
        ]
        self._supervisor._registry.register_all(entries, room_id=room_id, owner=self)


class _PerWorkerToolServer:
    """Serves ``delegate_to_<id>``: delegates to that worker from the room of the call."""

    def __init__(
        self,
        kit: RoomKit,
        supervisor: Agent,
        tool_to_worker: dict[str, str],
        *,
        wait: bool,
        share_channels: list[str],
        task_timeout: float,
    ) -> None:
        self._kit = kit
        self._supervisor = supervisor
        self._tool_to_worker = tool_to_worker
        self._wait = wait
        self._share_channels = share_channels
        self._task_timeout = task_timeout
        # Per room: a worker busy in one room is free in another.
        self._pending: set[tuple[str, str]] = set()  # (room_id, worker_id)

    async def serve(self, rid: str, name: str, arguments: dict[str, Any]) -> str:
        """Answer one ``delegate_to_<id>`` call made in room *rid*."""
        worker_id = self._tool_to_worker[name]
        task_desc = arguments.get("task", "")
        try:
            if self._wait:
                return await self._delegate_and_wait(rid, worker_id, task_desc)
            return self._delegate_in_background(rid, worker_id, task_desc)
        except Exception:
            # Raised on: the channel reads it as any failed call, the class for
            # the model and the message for the observers (RFC §9.3).
            logger.exception("Delegation to %s failed", worker_id)
            raise

    async def _delegate_and_wait(self, rid: str, worker_id: str, task_desc: str) -> str:
        """Run the worker on *task_desc*, bounded by the task timeout, and
        answer with its result."""
        outcome = await run_worker(
            self._kit,
            rid,
            worker_id,
            task_desc,
            timeout=self._task_timeout,
            status=WorkerStatus({"room_id": rid, "mode": "per_worker_wait"}),
            share_channels=self._share_channels,
        )
        result = outcome.result
        return json.dumps(
            {
                "status": result.status if result else "failed",
                "worker": worker_id,
                "result": outcome.output,
            }
        )

    def _delegate_in_background(self, rid: str, worker_id: str, task_desc: str) -> str:
        """Start the worker on *task_desc* in the background, and answer at
        once; its outcome is handed back to the supervisor (RFC §19.7.3)."""
        pending = self._pending
        if (rid, worker_id) in pending:
            return _already_working(worker_id)
        pending.add((rid, worker_id))
        # If the start raises (shutdown race), free the worker so it isn't
        # stuck in already_running.
        try:
            run = self._background_run(rid, worker_id, task_desc)
            start_background_run(self._kit, run_in_background(self._kit, run))
        except BaseException:
            pending.discard((rid, worker_id))
            raise
        return _dispatched(worker_id)

    def _background_run(
        self, rid: str, worker_id: str, task_desc: str
    ) -> BackgroundRun[WorkerOutcome]:
        """The worker's background run: its delegation, bounded and followed
        as a waited one is, the worker freed in *rid* before its outcome is
        handed back."""
        kit = self._kit
        metadata = {"room_id": rid, "mode": "per_worker_async"}

        def post(level: StatusLevel, detail: str) -> None:
            _post_worker_status(
                kit,
                "orchestration",
                level,
                action="worker",
                detail=detail,
                metadata={**metadata, "worker": worker_id},
            )

        async def work() -> WorkerOutcome:
            return await run_worker(
                kit,
                rid,
                worker_id,
                task_desc,
                timeout=self._task_timeout,
                status=WorkerStatus(metadata),
                share_channels=self._share_channels,
            )

        return BackgroundRun(
            room_id=rid,
            notify=self._supervisor.channel_id,
            work=work,
            told=lambda outcome: _worker_told(worker_id, outcome),
            ended=_worker_ended,
            post=post,
            release=lambda _returned: self._pending.discard((rid, worker_id)),
        )


def _worker_told(worker_id: str, outcome: WorkerOutcome | None) -> str:
    """What the supervisor reads of a background worker: its outcome bounded
    and set apart as a worker's; for a run that raised (``None``), that the
    task could not be completed."""
    if outcome is None:
        return background_failure_text(f"task for {worker_id}")
    status = "completed" if outcome.completed else "did not complete"
    return result_text(
        f"[Your background task for {worker_id} {status}. Share the outcome with the user.]",
        bounded(outcome.output or "No output"),
    )


def _worker_ended(outcome: WorkerOutcome) -> tuple[StatusLevel, str]:
    """A background worker's terminal entry, once handed back: completed or
    failed, as its task ended."""
    if outcome.completed:
        return StatusLevel.COMPLETED, "completed"
    return StatusLevel.FAILED, "not completed"


def _already_working(worker_id: str) -> str:
    """What the model reads when the worker is still on this room's task."""
    return json.dumps(
        {
            "status": "already_running",
            "worker": worker_id,
            "message": (
                f"{worker_id} is already working on this. "
                "Do NOT call this tool again. "
                "Tell the user to ask again shortly."
            ),
        }
    )


def _dispatched(worker_id: str) -> str:
    """What the model reads when the worker was started in the background."""
    return json.dumps(
        {
            "status": "delegated",
            "worker": worker_id,
            "message": (
                f"Task dispatched to {worker_id}. "
                "It is running in the background. "
                "Do NOT call this tool again. "
                "Tell the user to ask again shortly "
                "for results."
            ),
        }
    )


def _worker_tool(worker: Agent) -> AITool:
    """The ``delegate_to_<id>`` declaration for one worker."""
    desc = getattr(worker, "description", None) or f"Worker agent {worker.channel_id}"
    return AITool(
        name=f"delegate_to_{worker.channel_id}",
        description=f"Delegate a task to {worker.channel_id}. {desc}",
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the task to delegate",
                },
            },
            "required": ["task"],
        },
    )
