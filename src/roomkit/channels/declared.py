"""A channel the room's owner holds by its declaration alone (RFC §10.6.2)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from roomkit.channels.base import Channel
from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.models.channel import (
    AgentIdentity,
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


_NO_IDENTITY = AgentIdentity(auto_greet=False)


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

    # -- What a strategy reads of an agent, as it reads it of a live one --

    @property
    def thinks_while_listening(self) -> bool:
        return self._declaration.thinks_while_listening

    @property
    def is_config_only(self) -> bool:
        """Never: an executor builds it with its provider (RFC §19.1)."""
        return False

    @property
    def name(self) -> str | None:
        return self._identity.name

    @property
    def role(self) -> str | None:
        return self._identity.role

    @property
    def description(self) -> str | None:
        return self._identity.description

    @property
    def scope(self) -> str | None:
        return self._identity.scope

    @property
    def greeting(self) -> str | None:
        return self._identity.greeting

    @property
    def language(self) -> str | None:
        return self._identity.language

    @property
    def purpose(self) -> str | None:
        return self._identity.purpose

    @property
    def voice(self) -> str | None:
        return self._identity.voice

    @property
    def auto_greet(self) -> bool:
        return self._identity.auto_greet

    @property
    def _identity(self) -> AgentIdentity:
        return self._declaration.identity or _NO_IDENTITY

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


def id_of(agent: Channel | str) -> str:
    """The channel id of a strategy's *agent*: the agent itself or its id."""
    return agent if isinstance(agent, str) else agent.channel_id


def held_agent[C: Channel](kit: RoomKit, agent: C | str) -> C | Channel:
    """A strategy's *agent*: itself, or the intelligence channel *kit* holds
    under the id it is named by, a declared one included (RFC §10.6.2).

    Raises:
        ChannelNotRegisteredError: *kit* holds no channel under the id.
        ValueError: the channel under the id is not an intelligence channel.
    """
    if not isinstance(agent, str):
        return agent
    channel = kit.channels.get(agent)
    if channel is None:
        raise ChannelNotRegisteredError(
            f"Channel {agent!r} is not registered: a strategy names by its id an agent "
            "the kit holds, live or by its declaration"
        )
    if channel.category is not ChannelCategory.INTELLIGENCE:
        raise ValueError(f"Channel {agent!r} is not an intelligence channel")
    return channel


def register_strategy_agents(kit: RoomKit, agents: Iterable[Channel | str]) -> None:
    """Register on *kit* each of a strategy's *agents* given live that it does
    not hold yet: all of them, or none when one is refused. An agent named by
    its id is one the kit holds already.

    Raises:
        ChannelNotRegisteredError: an id names no channel of the kit.
        ValueError: an id names no intelligence channel, or an agent is given
            live under an id the kit holds by its declaration alone: that
            object would never run.
    """
    agents = list(agents)
    for agent in agents:
        held = kit.channels.get(id_of(agent))
        if isinstance(agent, str):
            held_agent(kit, agent)
        elif isinstance(held, DeclaredChannel) and held is not agent:
            raise ValueError(
                f"Channel {agent.channel_id!r} is registered by its declaration alone: "
                f"give the strategy its id, {agent.channel_id!r}, not an agent object"
            )
    for agent in agents:
        if not isinstance(agent, str) and agent.channel_id not in kit.channels:
            kit.register_channel(agent)


def named_agent[C: Channel](agent: C | str) -> C | DeclaredChannel:
    """A strategy's *agent* until its install reads the kit's: itself, or for
    one named by its id a bare declaration under that id."""
    if isinstance(agent, str):
        return DeclaredChannel(ChannelDeclaration(channel_id=agent))
    return agent
