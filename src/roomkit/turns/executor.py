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

from roomkit.core.task_utils import cancel_and_wait
from roomkit.turns._executor_turn import ExecutorTurn, turns_run_under
from roomkit.turns.base import TurnTimeouts

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.turns.base import ExecutorLink

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

    def declares(self, channel_id: str) -> bool:
        """Whether this executor takes the turns of *channel_id*."""
        return not self._closed and (self._hosts is None or self._hosts(channel_id))

    def take(self, link: ExecutorLink) -> None:
        """Start the turn *link* brought; it runs on its own task."""
        turn = ExecutorTurn(
            link, self.executor_id, self._channel(link.request.channel_id), self._timeouts
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

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._turns.pop(task, None)

    def _channel(self, channel_id: str) -> Awaitable[Channel | None]:
        """The channel the factory gives for *channel_id*, awaited by the turn."""
        built: Any = self._build_channel(channel_id)
        if inspect.isawaitable(built):
            return built
        future: asyncio.Future[Channel | None] = asyncio.get_running_loop().create_future()
        future.set_result(built)
        return future
