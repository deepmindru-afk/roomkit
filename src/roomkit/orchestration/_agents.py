"""A strategy's agents, given live or named by id (RFC §10.6.2).

A supervisor, its workers, a loop's producer and reviewers are each an
``Agent`` the host gives the strategy, or the channel id of one the kit
holds, live or by its declaration alone. The strategy reads the kit's at its
install; until then an id stands as a bare declaration under that id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.channels.agent import Agent
from roomkit.channels.declared import DeclaredChannel, held_agent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

type HeldAgent = Agent | DeclaredChannel
"""A strategy's agent as it reads it: the agent given live, or the channel the
kit holds under the id it was named by."""


def held(kit: RoomKit, agent: Agent | str) -> HeldAgent:
    """*agent* as given, or the agent *kit* holds under the id it is named by.

    Raises:
        ChannelNotRegisteredError: *kit* holds no channel under the id.
        ValueError: the id names no intelligence channel.
        TypeError: the id names an intelligence channel that is no agent (an
            ``AIChannel``, an ACP agent): give the strategy the channel itself.
    """
    if not isinstance(agent, str):
        return agent
    channel = held_agent(kit, agent)
    if not isinstance(channel, Agent | DeclaredChannel):
        raise TypeError(
            f"Channel {agent!r} is not an Agent nor a declared one: a strategy takes it "
            "by its id only then"
        )
    return channel
