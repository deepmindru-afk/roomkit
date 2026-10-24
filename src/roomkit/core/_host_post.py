"""``RoomKit.post()``: the host's own voice in a room (RFC §10.5).

The host speaks through a transport of type ``HOST``: its words reach the
room's people as a party's message, and solicit no agent unless they name
one. The room is given the host channel on its first post.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from roomkit.channels.host import HOST_CHANNEL_ID, HostChannel
from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventContent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.event import RoomEvent


async def post(
    kit: RoomKit,
    room_id: str,
    content: str | EventContent,
    *,
    participant_id: str | None,
    addressed_to: Sequence[str],
    metadata: dict[str, Any] | None,
    channel_id: str | None,
    organization_id: str | None,
) -> RoomEvent:
    """Inject *content* from the host channel, attaching it first if unbound."""
    host_id = _host_channel_id(kit, channel_id)
    # An ordinary attach over a live binding is allowed and would rewrite it:
    # a binding the integrator set up keeps its settings.
    if await kit.store.get_binding(room_id, host_id) is None:
        await kit.attach_channel(room_id, host_id, organization_id=organization_id)
    body = TextContent(body=content) if isinstance(content, str) else content
    return await kit.send_event(
        room_id,
        host_id,
        body,
        participant_id=participant_id,
        addressed_to=list(addressed_to),
        metadata=metadata,
        organization_id=organization_id,
    )


def _host_channel_id(kit: RoomKit, channel_id: str | None) -> str:
    """The host channel the post speaks from, registered on first use when unnamed."""
    if channel_id is not None:
        channel = kit.get_channel(channel_id)
        if channel is None:
            raise ChannelNotRegisteredError(f"Channel {channel_id} not registered")
        if channel.channel_type is not ChannelType.HOST:
            msg = f"post() speaks from a channel of type HOST; {channel_id!r} is not one"
            raise ValueError(msg)
        return channel_id
    channel = kit.get_channel(HOST_CHANNEL_ID)
    if channel is None:
        kit.register_channel(HostChannel())
    elif channel.channel_type is not ChannelType.HOST:
        msg = (
            f"Channel {HOST_CHANNEL_ID!r} is registered and is not a host channel: "
            "name the HostChannel post() speaks from with channel_id"
        )
        raise ValueError(msg)
    return HOST_CHANNEL_ID
