"""HostChannel — the host application's own voice in a room (RFC §10.5).

The application that embeds RoomKit writes to a room's people in its own name:
a notice, a workflow's question, a reminder. :meth:`RoomKit.post` speaks
through this channel; ``send_event`` from it does the same, one step lower.
"""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelDirection, ChannelType
from roomkit.models.event import RoomEvent

HOST_CHANNEL_ID = "host"
"""The id :meth:`RoomKit.post` registers its host channel under on first use."""


class HostChannel(Channel):
    """A transport of type ``HOST`` that brings the host's words into rooms.

    Its events are ``INBOUND``, as every transport's: the room reads them as a
    party's message, broadcasts them to its other transports, and they solicit
    agents as a message does unless addressed otherwise (RFC §19.3). Its
    direction is ``INBOUND``: it is never delivered to, and it receives no
    external payload. Who speaks is a participant of the room, named by the
    ``participant_id`` of the post.

    It is not an intelligence channel, which would make the host's words an
    agent's output, nor a ``SYSTEM`` source, which marks what the framework
    writes itself (RFC §6.2).
    """

    channel_type = ChannelType.HOST
    category = ChannelCategory.TRANSPORT
    direction = ChannelDirection.INBOUND

    sender_is_participant = True
    """Who speaks is named by the room (``participant_id``), not an address."""

    def __init__(self, channel_id: str = HOST_CHANNEL_ID) -> None:
        super().__init__(channel_id)

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        msg = (
            f"Channel {self.channel_id!r} receives no external payload: the host speaks "
            "through RoomKit.post() or send_event()"
        )
        raise TypeError(msg)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        # Never called: an INBOUND transport is not delivered to (RFC §4.2).
        return ChannelOutput.empty()
