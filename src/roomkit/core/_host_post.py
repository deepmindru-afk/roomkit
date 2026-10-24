"""``RoomKit.post()``: the host's own voice in a room (RFC §10.5).

The host speaks through a transport of type ``HOST``: its words reach the
room's people as a party's message, and solicit no agent unless they name
one. A room that never bound the host channel is given it on its first post.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from roomkit.channels.host import HOST_CHANNEL_ID, HostChannel
from roomkit.core._authors import People
from roomkit.core._publication_keys import refuse_reserved
from roomkit.core.exceptions import (
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    ParticipantNotFoundError,
    RoomClosedError,
    RoomNotFoundError,
)
from roomkit.core.mixins.helpers import _REFUSING_STATUSES
from roomkit.core.task_utils import run_to_its_end
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventContent, TextContent, refuse_too_deep

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
    idempotency_key: str | None,
    chain_depth: int,
) -> RoomEvent:
    """Inject *content* from the host channel, attaching it first to a room
    that never bound it, within one ``process_timeout`` from the call's start
    to its commit (RFC §13.6)."""
    deadline = asyncio.get_running_loop().time() + kit._process_timeout
    body = TextContent(body=content) if isinstance(content, str) else content
    # What send_event() refuses of its arguments, refused before the attach
    # can write anything.
    refuse_too_deep(body)
    refuse_reserved(idempotency_key)
    if chain_depth < 0:
        raise ValueError(f"chain_depth is a depth in a chain, never negative: {chain_depth}")
    host_id = _host_channel_id(kit, channel_id)
    if participant_id:
        await kit._by_deadline(
            _refuse_unreached_speaker(kit, room_id, host_id, participant_id, organization_id),
            room_id,
            host_id,
            deadline,
        )
    # A room that binds the channel is never written to here: this read
    # decides nothing the locked section would decide otherwise, and spares
    # each later post the lock.
    bound = await kit._by_deadline(
        kit.store.get_binding(room_id, host_id), room_id, host_id, deadline
    )
    if bound is None:
        await _attach_on_first_post(kit, room_id, host_id, organization_id, deadline)
    # The rest of the call's budget, not a fresh one (RFC §13.6).
    return await kit._send_event(
        room_id,
        host_id,
        body,
        participant_id=participant_id,
        addressed_to=list(addressed_to),
        metadata=metadata,
        idempotency_key=idempotency_key,
        organization_id=organization_id,
        chain_depth=chain_depth,
        deadline=deadline,
    )


async def _refuse_unreached_speaker(
    kit: RoomKit, room_id: str, host_id: str, participant_id: str, organization_id: str | None
) -> None:
    """Refuse a speaker *host_id* does not reach, before anything is written
    (RFC §10.5, §5.5): *participant_id* names a participant of the room, by
    their id or their identity's, whose ``connected_via`` holds the host
    channel, as :class:`People` reads a turn's sender. The room is read
    scoped first, so that another organization's room is not found whoever
    the id names.

    Raises:
        RoomNotFoundError: the room is missing, or another organization's.
        ParticipantNotFoundError: the id names nobody on the host channel.
    """
    await kit.get_room(room_id, organization_id=organization_id)
    people = People(await kit.store.list_participants(room_id))
    if people.named_on(participant_id, host_id) is None:
        raise ParticipantNotFoundError(
            f"post() speaks as a participant of room {room_id} reached through the host "
            f"channel {host_id!r}, and {participant_id!r} names none: add them with "
            f"kit.add_member({room_id!r}, {host_id!r}, {participant_id!r})"
        )


async def _attach_on_first_post(
    kit: RoomKit, room_id: str, host_id: str, organization_id: str | None, deadline: float
) -> None:
    """Attach *host_id* to a room that never bound it, in one section under
    the room lock, where every binding change and status change runs (RFC
    §10.5), as step 2's attach does (§10.1): the wait for the lock is
    bounded by the call's *deadline*, and the section it opens is not, nor is
    it cut by its caller (§13.6). Past the deadline the post is refused,
    nothing written; once the lock is held, the attach runs to its end, its
    record and its ON_CHANNEL_ATTACHED hooks included.

    Raises:
        ProcessTimeoutError: the deadline passed before the lock was held.
        RoomNotFoundError: the room is missing, or another organization's.
        RoomClosedError: the room's status refuses new events (RFC §5.1).
        ChannelNotFoundError: the host channel was detached from the room.
    """
    async with AsyncExitStack() as stack:
        await kit._by_deadline(
            stack.enter_async_context(kit._lock_manager.locked(room_id)),
            room_id,
            host_id,
            deadline,
        )
        await run_to_its_end(
            _attach_locked(kit, room_id, host_id, organization_id),
            held=kit._decided_writes,
            name=f"roomkit-decided-write-{room_id}",
            expected=(RoomNotFoundError, RoomClosedError, ChannelNotFoundError),
        )


async def _attach_locked(
    kit: RoomKit, room_id: str, host_id: str, organization_id: str | None
) -> None:
    """:func:`_attach_on_first_post`'s section, under the room lock its
    caller holds: the room read with its scope and its due timers applied,
    then its binding. A binding made by then, the integrator's above all, is
    kept as it is. Nothing is written for a room that refuses events, nor
    for one the host channel was detached from (§7.5 rule 7), which only an
    explicit attach grants it again."""
    room = await kit.get_room(room_id, organization_id=organization_id)
    room = await kit._room_timers.apply_due(room)
    if room.status in _REFUSING_STATUSES:
        raise RoomClosedError(f"Room {room_id} does not accept new events")
    if await kit._attach_unbound(room, host_id) is None:
        raise ChannelNotFoundError(
            f"Channel {host_id} not in room {room_id}: it was detached, and post() "
            "does not attach it again"
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
