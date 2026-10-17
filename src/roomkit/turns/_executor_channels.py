"""The channels one executor runs (RFC §10.6.2, §10.6.3).

The kit's own executor finds them in the kit's registry, which wires and
closes them. An executor a host builds builds each through the host's
factory once, wires it, and closes what it built when it closes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.telemetry.base import TelemetryProvider

logger = logging.getLogger("roomkit.turns")

ChannelFactory = Callable[[str], "Channel | None | Awaitable[Channel | None]"]
"""Builds, or finds, the channel a turn names by its id."""


async def _called(factory: ChannelFactory, channel_id: str) -> Channel | None:
    built: Any = factory(channel_id)
    return await built if inspect.isawaitable(built) else built


class ExecutorChannels:
    """The channels of one executor, by id.

    Args:
        factory: The host's factory, or the registry the executor reads.
        owns: Whether the executor builds its channels (a host's factory):
            each is built once, wired and closed with the executor; else
            the factory is a registry read every time, its owner wiring
            and closing what it holds.
        wire: Wires a channel the executor built.
        telemetry: The telemetry a built channel reports to.
    """

    def __init__(
        self,
        factory: ChannelFactory,
        *,
        owns: bool,
        wire: Callable[[Channel], None],
        telemetry: TelemetryProvider | None = None,
    ) -> None:
        self._factory = factory
        self._owns = owns
        self._wire = wire
        self._telemetry = telemetry
        self._built: dict[str, Channel] = {}
        self._building: dict[str, asyncio.Task[Channel | None]] = {}

    async def get(self, channel_id: str) -> Channel | None:
        """The channel for *channel_id*, ``None`` when the factory has none."""
        if not self._owns:
            return await _called(self._factory, channel_id)
        built = self._built.get(channel_id)
        if built is not None:
            return built
        pending = self._building.get(channel_id)
        if pending is None:
            pending = asyncio.get_running_loop().create_task(self._build(channel_id))
            self._building[channel_id] = pending
            pending.add_done_callback(lambda _: self._building.pop(channel_id, None))
        # Shielded: a turn cut while the channel is built does not cut the
        # build another turn waits for.
        return await asyncio.shield(pending)

    async def _build(self, channel_id: str) -> Channel | None:
        channel = await _called(self._factory, channel_id)
        if channel is None:
            return None
        self._wire(channel)
        if self._telemetry is not None:
            channel._telemetry = self._telemetry  # ty: ignore[unresolved-attribute]
            if hasattr(channel, "_propagate_telemetry"):
                channel._propagate_telemetry()  # ty: ignore[call-non-callable]
        self._built[channel_id] = channel
        return channel

    async def close(self) -> None:
        """Close what the executor built; what a registry holds is its owner's."""
        built, self._built = list(self._built.values()), {}
        for channel in built:
            try:
                await channel.close()
            except Exception:
                logger.exception("Closing channel %s failed", channel.channel_id)
