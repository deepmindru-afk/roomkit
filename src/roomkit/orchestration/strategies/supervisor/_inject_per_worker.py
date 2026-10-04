"""Per-worker delegation wiring for the supervisor.

Mixin for :class:`Supervisor`: injects one ``delegate_to_<id>`` tool per worker
and lets the AI decide when to delegate (manual mode). Host attributes are
declared as annotations; they are set in ``Supervisor.__init__``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import orchestration_tool
from roomkit.orchestration._call_room import in_call_room
from roomkit.orchestration._worker_run import (
    WorkerStatus,
    run_worker,
    task_completed,
    task_output,
)
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _post_worker_status,
    logger,
)
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask


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
            return await self._delegate_in_background(rid, worker_id, task_desc)
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

    async def _delegate_in_background(self, rid: str, worker_id: str, task_desc: str) -> str:
        """Start the worker on *task_desc*, and answer at once."""
        kit = self._kit
        pending = self._pending
        if (rid, worker_id) in pending:
            return _already_working(worker_id)

        delegated = await kit.delegate(
            rid,
            worker_id,
            task_desc,
            notify=self._supervisor.channel_id,
            share_channels=self._share_channels,
        )
        pending.add((rid, worker_id))
        _post_worker_status(
            kit,
            worker_id,
            StatusLevel.PENDING,
            detail=task_desc,
            metadata={
                "room_id": rid,
                "mode": "per_worker_async",
                "task_id": delegated.id,
            },
        )

        self._track_completion(rid, worker_id, delegated)
        return _dispatched(worker_id, delegated.id)

    def _track_completion(self, rid: str, worker_id: str, delegated: DelegatedTask) -> None:
        """Free the worker in *rid* and post its outcome when *delegated* ends."""
        pending = self._pending
        kit = self._kit
        original_set = delegated._set_result
        task_id = delegated.id

        def _patched_set(r: Any) -> None:
            pending.discard((rid, worker_id))
            _post_worker_status(
                kit,
                worker_id,
                StatusLevel.COMPLETED if task_completed(r) else StatusLevel.FAILED,
                detail=task_output(r),
                metadata={"room_id": rid, "mode": "per_worker_async", "task_id": task_id},
            )
            original_set(r)

        delegated._set_result = _patched_set  # ty: ignore[invalid-assignment]


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


def _dispatched(worker_id: str, task_id: str) -> str:
    """What the model reads when the worker was started in the background."""
    return json.dumps(
        {
            "status": "delegated",
            "task_id": task_id,
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
