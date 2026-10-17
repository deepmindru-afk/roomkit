"""The turn executor: builds a channel and runs its turns (RFC §10.6).

An executor never commits: what a turn produces reaches the room through its
owner. The kit owns one in memory, which reads the channels the host
registered; a host runs others with a factory of its own.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core._turn_scope import turns_run_under
from roomkit.core.hooks import HookEngine, HookRegistration, hook_decorator
from roomkit.core.task_utils import cancel_and_wait
from roomkit.models.enums import HookExecution
from roomkit.turns._codec import error_record
from roomkit.turns._executor_channels import ChannelFactory, ExecutorChannels
from roomkit.turns._executor_rooms import ExecutorRooms
from roomkit.turns._executor_turn import ExecutorTurn
from roomkit.turns._wiring import TurnWiring
from roomkit.turns._wiring_env import StoreWiringEnv
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.wire import ChannelCall, ChannelResult

if TYPE_CHECKING:
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import AsyncHookFn, SyncHookFn
    from roomkit.models.enums import ChannelDirection, ChannelType, EventType, HookTrigger
    from roomkit.store.base import ConversationStore
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.turns._wiring_env import WiringEnv
    from roomkit.turns.base import ExecutorLink
    from roomkit.turns.wire import ChannelControl, TurnRequest

logger = logging.getLogger("roomkit.turns")


class TurnExecutor:
    """Runs the turns of the channels it declares (RFC §10.6).

    Args:
        build_channel: The host's factory: the channel for a channel id, or
            ``None`` when it has none; sync or async. The executor builds
            each channel once, wires it (RFC §10.6.2) and closes it when it
            closes.
        hosts: Which channel ids this executor declares; every id when
            omitted.
        executor_id: The id the executor answers with; a random one when
            omitted.
        timeouts: The heartbeat bounds it keeps.
        store: The shared store its hooks build a room's context from when
            they fire, and its channels read (RFC §10.6.6, §10.6.9).
        media_resolver: The host's media resolver, given to its AI channels
            as the kit gives the ones it registers (RFC §6.4).
        attached_media: The host's attached-media reader (RFC §6.4).
        telemetry: The telemetry its hooks and channels report to.
        owns_channels: ``False`` when *build_channel* reads a registry whose
            owner wires and closes its channels (the kit's own executor).
    """

    def __init__(
        self,
        build_channel: ChannelFactory,
        *,
        hosts: Callable[[str], bool] | None = None,
        executor_id: str | None = None,
        timeouts: TurnTimeouts | None = None,
        store: ConversationStore | None = None,
        media_resolver: MediaResolver | None = None,
        attached_media: AttachedMedia | None = None,
        telemetry: TelemetryProvider | None = None,
        owns_channels: bool = True,
    ) -> None:
        self.executor_id = executor_id or f"executor-{uuid4().hex[:12]}"
        self._hosts = hosts
        self._timeouts = timeouts or TurnTimeouts()
        self._turns: dict[asyncio.Task[None], ExecutorTurn] = {}
        self._closed = False
        self._hooks = HookEngine(turns=True)
        if telemetry is not None:
            self._hooks._telemetry = telemetry
        self.store = store
        """The shared store the executor was given, ``None`` for none."""
        env = StoreWiringEnv(store, media_resolver=media_resolver, attached_media=attached_media)
        self._wiring = TurnWiring(self._hooks, env)
        self._channels = ExecutorChannels(
            build_channel, owns=owns_channels, wire=self.wire, telemetry=telemetry
        )
        self._rooms = ExecutorRooms(self.executor_id, self._channels.get)

    @property
    def hooks(self) -> HookEngine:
        """The hooks of the triggers a turn fires (RFC §10.6.6)."""
        return self._hooks

    @property
    def wiring(self) -> TurnWiring:
        """What the executor gives the channels it runs (RFC §10.6.2)."""
        return self._wiring

    def use_wiring(self, env: WiringEnv) -> None:
        """Wire channels from *env* in place of what the executor was given:
        what the kit gives its own executor, before any channel is wired."""
        self._wiring = TurnWiring(self._hooks, env)

    def wire(self, channel: Channel) -> None:
        """Give *channel* what its turns fire this executor's hooks through:
        the kit's own executor wires each channel the kit registers, another
        each channel its factory builds."""
        self._wiring.wire(channel)

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

    def room_hook(
        self,
        name: str,
        trigger: HookTrigger,
        execution: HookExecution = HookExecution.SYNC,
        priority: int = 0,
        timeout: float = 30.0,
        *,
        fail_closed: bool = False,
    ) -> Callable[..., Any]:
        """Decorator to define a room hook named *name* on a trigger a turn
        fires (RFC §10.6.6): it runs for a room whose owner enabled the name
        (``kit.enable_room_hook(room_id, name)``), the turn's request naming
        it. Every executor that may run the room's turns defines it; one that
        does not is never given them. A name may carry a hook per trigger.

        ``executor.room_hook("audit", HookTrigger.ON_TOOL_CALL)(fn)``
        """

        def decorator(fn: SyncHookFn | AsyncHookFn) -> SyncHookFn | AsyncHookFn:
            self._hooks.define_room_hook(
                HookRegistration(
                    trigger=trigger,
                    execution=execution,
                    fn=fn,
                    priority=priority,
                    name=name,
                    timeout=timeout,
                    fail_closed=fail_closed,
                )
            )
            return fn

        return decorator

    def remove_room_hook(self, name: str) -> bool:
        """Forget the room hooks defined under *name*."""
        return self._hooks.undefine_room_hook(name)

    def declares(self, channel_id: str) -> bool:
        """Whether this executor takes the turns of *channel_id*."""
        return not self._closed and (self._hosts is None or self._hosts(channel_id))

    def takes(self, channel_id: str, room_hooks: Iterable[str] = ()) -> bool:
        """Whether this executor takes a turn of *channel_id* whose room has
        the hooks named *room_hooks* (RFC §10.6.7)."""
        return self.declares(channel_id) and self._hooks.defines_room_hooks(room_hooks)

    def take(self, link: ExecutorLink) -> None:
        """Start the turn *link* brought; it runs on its own task."""
        request = link.request
        self._hooks.use_room_hooks(request.room_id, request.room_hooks)
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
        await self._channels.close()

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        """Apply a channel control message of the owner's (RFC §10.6.3): a
        ``channel_call``'s result; nothing for ``room_attached`` and
        ``room_detached``."""
        try:
            channel = await self._channels.get(message.channel_id)
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
        channel = await self._channels.get(request.channel_id)
        if channel is not None:
            await self._rooms.prepare(channel, request, stale=stale)
        return channel
