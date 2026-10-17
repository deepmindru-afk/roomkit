"""The turn executor: builds a channel and runs its turns (RFC §10.6).

An executor never commits: what a turn produces reaches the room through its
owner. The kit owns one in memory, which reads the channels the host
registered; a host runs others with a factory of its own.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core._turn_scope import turns_run_under
from roomkit.core.hooks import HookEngine, HookRegistration, hook_decorator
from roomkit.core.task_utils import cancel_and_wait
from roomkit.models.enums import HookExecution
from roomkit.turns._codec import error_record
from roomkit.turns._executor_rooms import ExecutorRooms
from roomkit.turns._executor_turn import ExecutorTurn
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.wire import ChannelCall, ChannelResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import AsyncHookFn, SyncHookFn
    from roomkit.models.enums import ChannelDirection, ChannelType, EventType, HookTrigger
    from roomkit.turns.base import ExecutorLink
    from roomkit.turns.wire import ChannelControl, TurnRequest

logger = logging.getLogger("roomkit.turns")

ChannelFactory = Callable[[str], "Channel | None | Awaitable[Channel | None]"]
"""Builds, or finds, the channel a turn names by its id."""


class TurnExecutor:
    """Runs the turns of the channels it declares (RFC §10.6).

    Args:
        build_channel: The host's factory: the channel for a channel id, or
            ``None`` when it has none; sync or async.
        hosts: Which channel ids this executor declares; every id when
            omitted.
        executor_id: The id the executor answers with; a random one when
            omitted.
        timeouts: The heartbeat bounds it keeps.
    """

    def __init__(
        self,
        build_channel: ChannelFactory,
        *,
        hosts: Callable[[str], bool] | None = None,
        executor_id: str | None = None,
        timeouts: TurnTimeouts | None = None,
    ) -> None:
        self.executor_id = executor_id or f"executor-{uuid4().hex[:12]}"
        self._build_channel = build_channel
        self._hosts = hosts
        self._timeouts = timeouts or TurnTimeouts()
        self._turns: dict[asyncio.Task[None], ExecutorTurn] = {}
        self._closed = False
        self._hooks = HookEngine(turns=True)
        self._rooms = ExecutorRooms(self.executor_id, self._channel)

    @property
    def hooks(self) -> HookEngine:
        """The hooks of the triggers a turn fires (RFC §10.6.6)."""
        return self._hooks

    def hook(
        self,
        trigger: HookTrigger,
        execution: HookExecution = HookExecution.SYNC,
        priority: int = 0,
        name: str = "",
        timeout: float = 30.0,
        channel_types: set[ChannelType] | None = None,
        channel_ids: set[str] | None = None,
        directions: set[ChannelDirection] | None = None,
        event_types: set[EventType] | None = None,
        fail_closed: bool = False,
    ) -> Callable[..., Any]:
        """Decorator to register a hook on a trigger a turn fires
        (``ON_TOOL_CALL``, ``BEFORE_TOOL_USE``, ``ON_AI_RESPONSE``...), with
        ``RoomKit.hook``'s options. An owner's trigger raises ``ValueError``.

        The kit's own executor (``kit.turn_executor``) fires these hooks for
        the turns it runs and for the tool calls the kit's process serves
        (realtime voice, conference). An executor a host builds runs its turns
        without them for now: the kit wires the channels it registers, and an
        executor that wires its own comes with the distributed backends.
        """
        return hook_decorator(
            self._hooks.register,
            trigger,
            execution=execution,
            priority=priority,
            name=name,
            timeout=timeout,
            channel_types=channel_types,
            channel_ids=channel_ids,
            directions=directions,
            event_types=event_types,
            fail_closed=fail_closed,
        )

    def add_room_hook(
        self,
        room_id: str,
        trigger: HookTrigger,
        execution: HookExecution,
        fn: SyncHookFn | AsyncHookFn,
        priority: int = 0,
        name: str = "",
        *,
        fail_closed: bool = False,
    ) -> None:
        """Add a hook on a trigger a turn fires, for *room_id* alone."""
        self._hooks.add_room_hook(
            room_id,
            HookRegistration(
                trigger=trigger,
                execution=execution,
                fn=fn,
                priority=priority,
                name=name,
                fail_closed=fail_closed,
            ),
        )

    def remove_room_hook(self, room_id: str, name: str) -> bool:
        """Remove a room hook by name."""
        return self._hooks.remove_room_hook(room_id, name)

    def declares(self, channel_id: str) -> bool:
        """Whether this executor takes the turns of *channel_id*."""
        return not self._closed and (self._hosts is None or self._hosts(channel_id))

    def take(self, link: ExecutorLink) -> None:
        """Start the turn *link* brought; it runs on its own task."""
        request = link.request
        stale = self._rooms.enter(
            request.channel_id, request.room_id, request.turn_id, request.previous_turn
        )
        turn = ExecutorTurn(
            link, self.executor_id, self._turn_channel(link.request, stale), self._timeouts
        )
        task = asyncio.get_running_loop().create_task(
            turn.run(),
            context=contextvars.Context(),
            name=f"roomkit-turn-{link.request.turn_id}",
        )
        self._turns[task] = turn
        task.add_done_callback(self._forget)

    async def close(self) -> None:
        """Take no more turns, and cut the ones running, but the turns this
        close runs under: waiting for them would wait for itself."""
        self._closed = True
        spared = set(turns_run_under())
        running = []
        for task, turn in list(self._turns.items()):
            if turn.turn_id in spared:
                turn.spare()
            else:
                running.append(task)
        if running:
            await cancel_and_wait(*running, log_errors_to=logger)

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        """Apply a channel control message of the owner's (RFC §10.6.3): a
        ``channel_call``'s result; nothing for ``room_attached`` and
        ``room_detached``."""
        try:
            channel = await self._channel(message.channel_id)
        except Exception as exc:
            if isinstance(message, ChannelCall):
                return ChannelResult(error=error_record(exc), executor_id=self.executor_id)
            logger.exception("The factory failed to build channel %s", message.channel_id)
            return None
        return await self._rooms.apply(channel, message)

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._turns.pop(task, None)

    async def _turn_channel(self, request: TurnRequest, stale: bool) -> Channel | None:
        """The channel of *request*'s turn, readied for its room (RFC §10.6.2)."""
        channel = await self._channel(request.channel_id)
        if channel is not None:
            await self._rooms.prepare(channel, request, stale=stale)
        return channel

    def _channel(self, channel_id: str) -> Awaitable[Channel | None]:
        """The channel the factory gives for *channel_id*, awaited by the turn."""
        built: Any = self._build_channel(channel_id)
        if inspect.isawaitable(built):
            return built
        future: asyncio.Future[Channel | None] = asyncio.get_running_loop().create_future()
        future.set_result(built)
        return future
