"""A channel the room's owner holds by its declaration alone (RFC §10.6.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.channels.base import Channel
from roomkit.models.channel import (
    ChannelBinding,
    ChannelCapabilities,
    ChannelDeclaration,
    ChannelOutput,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import RoomEvent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit


class DeclaredChannel(Channel):
    """An intelligence channel the kit knows by its declaration, with no
    object of it in this process.

    The kit finds it wherever it looks a channel up: a binding takes its
    capabilities, routing plans its turns, a delegation names it. Its turns
    run on an executor whose host builds the channel; the kit's own executor
    never builds it, so with no such executor they fail with ``no_executor``.
    Nothing here runs it: a message, a delivery or a turn handed to it is
    refused.

    Raises:
        ValueError: *declaration* is not an intelligence channel's.
    """

    def __init__(self, declaration: ChannelDeclaration) -> None:
        if declaration.category is not ChannelCategory.INTELLIGENCE:
            raise ValueError(
                f"Channel {declaration.channel_id!r} is declared as a "
                f"{declaration.category.value} channel: only an intelligence channel "
                "can be registered by its declaration alone"
            )
        super().__init__(declaration.channel_id)
        self.channel_type = declaration.channel_type
        self.category = declaration.category
        self._declaration = declaration

    def declaration(self) -> ChannelDeclaration:
        return self._declaration

    def capabilities(self) -> ChannelCapabilities:
        # A binding keeps what it is given: each one gets its own copy.
        return self._declaration.capabilities.model_copy(deep=True)

    @property
    def recent_events_window(self) -> int:
        return self._declaration.recent_events_window

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise TypeError(self._absent("take a message"))

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        raise TypeError(self._absent("be delivered to"))

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        raise TypeError(self._absent("run a turn"))

    def _absent(self, what: str) -> str:
        return (
            f"Channel {self.channel_id!r} is registered by its declaration alone: it "
            f"cannot {what} in this process; an executor whose host builds it runs it"
        )


def register_strategy_agent(kit: RoomKit, agent: Channel) -> None:
    """Register *agent*, a strategy's, on *kit* unless the kit holds its id
    already.

    Raises:
        ValueError: the kit holds the id by its declaration alone: the
            strategy would set its turns and tools up on an object the kit
            never reads.
    """
    held = kit.channels.get(agent.channel_id)
    if held is None:
        kit.register_channel(agent)
    elif isinstance(held, DeclaredChannel):
        raise ValueError(
            f"Channel {agent.channel_id!r} is registered by its declaration alone: "
            "a strategy needs the agent itself registered on the kit"
        )
