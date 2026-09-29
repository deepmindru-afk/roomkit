"""Framework-driven auto-delegate wiring for the supervisor.

Mixin for :class:`Supervisor`: wraps the supervisor's ``on_event`` (sync) or
injects a background ``delegate_workers`` tool on a ``RealtimeVoiceChannel``
(async). Host attributes are declared as annotations; they are set in
``Supervisor.__init__``.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.event import RoomEvent
from roomkit.orchestration._call_room import call_room_handler
from roomkit.orchestration._installs import first_install
from roomkit.orchestration.strategies.supervisor._common import (
    WorkerStrategy,
    _is_subtask_room,
    logger,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _async_run_and_deliver,
    _one_pass_delegate,
    _two_pass_delegate,
)
from roomkit.orchestration.strategies.supervisor.results import _worker_roles_csv

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

# The supervisors whose ``on_event`` already runs the framework-driven
# delegation, and the voice channels already serving ``delegate_workers``: a
# second room's install wraps nothing.
_AUTO_DELEGATING: weakref.WeakSet[Any] = weakref.WeakSet()
_VOICE_SERVING: weakref.WeakSet[Any] = weakref.WeakSet()
# The rooms each voice channel's ``delegate_workers`` was installed in.
_VOICE_ROOMS: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()


class _AutoDelegateInstallMixin:
    """Install framework-driven delegation (sync wrap or async voice tool)."""

    _supervisor: Agent
    _workers: list[Agent]
    _strategy: WorkerStrategy | None
    _async_delivery: bool
    _refine_task: bool
    _refine_instruction: str | None
    _share_channels: list[str]
    _max_revisions: int
    _task_timeout: float

    def _install_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Install framework-driven delegation (sync or async), once.

        The supervisor and the voice channel serve every room the strategy is
        installed in (RFC §19.7): a second room's install finds them wired,
        and each delegation runs for the room it came from.
        """
        if self._async_delivery:
            self._install_async_auto_delegate(kit, room_id)
        else:
            self._install_sync_auto_delegate(kit)

    def _install_sync_auto_delegate(self, kit: RoomKit) -> None:
        """Wrap supervisor's on_event — blocks until workers complete."""
        if not first_install(_AUTO_DELEGATING, self._supervisor):
            return
        supervisor = self._supervisor
        strategy = self._strategy
        workers = self._workers
        refine = self._refine_task
        refine_instruction = self._refine_instruction
        share_channels = self._share_channels
        max_revisions = self._max_revisions
        task_timeout = self._task_timeout
        original_on_event = supervisor.on_event

        async def auto_delegate_on_event(
            event: RoomEvent,
            binding: ChannelBinding,
            context: RoomContext,
        ) -> ChannelOutput:
            if event.source.channel_id == supervisor.channel_id:
                return ChannelOutput.empty()
            if event.source.channel_type == _ChannelType.AI:
                return ChannelOutput.empty()

            rid = context.room.id if context.room else event.room_id
            # Only the parent room drives delegation. Inside a child task room
            # (e.g. a supervisor review room created by the supervised loop, or
            # any delegated worker room), the supervisor must run NORMALLY —
            # otherwise the review prompt would be treated as a fresh user task
            # and re-trigger delegation, recursing without bound.
            if _is_subtask_room(rid):
                return await original_on_event(event, binding, context)

            if refine:
                return await _two_pass_delegate(
                    kit,
                    rid,
                    supervisor,
                    original_on_event,
                    event,
                    binding,
                    context,
                    strategy,
                    workers,
                    instruction=refine_instruction,
                    share_channels=share_channels,
                    max_revisions=max_revisions,
                    task_timeout=task_timeout,
                )
            return await _one_pass_delegate(
                kit,
                rid,
                supervisor,
                original_on_event,
                event,
                binding,
                context,
                strategy,
                workers,
                share_channels=share_channels,
                max_revisions=max_revisions,
                task_timeout=task_timeout,
            )

        supervisor.on_event = auto_delegate_on_event  # ty: ignore[invalid-assignment]

    def _install_async_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Inject delegate_workers tool into RealtimeVoiceChannel.

        The tool handler runs workers in the background and returns
        immediately. Results are delivered via kit.deliver().
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        strategy = self._strategy
        workers = self._workers
        share_channels = self._share_channels

        # Find the RealtimeVoiceChannel in registered channels
        voice_channel: RealtimeVoiceChannel | None = None
        for ch in kit.channels.values():
            if isinstance(ch, RealtimeVoiceChannel):
                voice_channel = ch
                break

        if voice_channel is None:
            logger.warning("async_delivery=True but no RealtimeVoiceChannel found")
            return

        # Build tool definition
        worker_roles = _worker_roles_csv(workers)
        tool_def = {
            "name": "delegate_workers",
            "description": (
                f"Delegate analysis to specialist workers ({worker_roles}). "
                f"Call when the user requests analysis, research, or investigation. "
                f"Workers run in {strategy} mode. Pass the topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The topic to analyze",
                    },
                },
                "required": ["task"],
            },
        }

        # Declared and served once per voice channel: a second room's install
        # adds its room, and one handler serves the installed rooms' calls. The
        # channel declares the same tools in every room, so a call from a room
        # the supervisor was not installed in is refused (RFC §19.7).
        rooms = _VOICE_ROOMS.setdefault(voice_channel, set())
        rooms.add(room_id)
        if not first_install(_VOICE_SERVING, voice_channel):
            return
        voice_channel._tools = [*(voice_channel._tools or []), tool_def]

        # Wrap tool handler for async delegation
        original_handler = voice_channel.tool_handler
        running: set[str] = set()  # rooms whose workers are running

        async def delegate_workers(rid: str, name: str, arguments: dict[str, Any]) -> str:
            if rid not in rooms:
                return json.dumps({"error": "delegate_workers is not available in this room"})
            if rid in running:
                return json.dumps(
                    {"status": "already_running", "message": "Workers are already running."}
                )
            running.add(rid)
            # Launch in the same step as the flag, so a second call of this
            # room's cannot slip between them. If create_task raises (shutdown
            # race), release the room so it isn't stuck in already_running.
            try:
                task = asyncio.create_task(
                    _async_run_and_deliver(
                        kit=kit,
                        room_id=rid,
                        strategy=strategy,
                        workers=workers,
                        task_desc=arguments.get("task", ""),
                        share_channels=share_channels,
                        on_done=lambda **_: running.discard(rid),
                    )
                )
                task.add_done_callback(log_task_exception)
            except BaseException:
                running.discard(rid)
                raise
            return json.dumps(
                {
                    "status": "dispatched",
                    "workers": worker_roles,
                    "message": "Workers are running. Results will be delivered when ready.",
                }
            )

        voice_channel.tool_handler = call_room_handler(
            {"delegate_workers"}, delegate_workers, original_handler
        )
