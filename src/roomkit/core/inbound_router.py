"""Inbound room router — determines which room an inbound message belongs to."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.delivery import SYSTEM_SENDER_ID
from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.models.identity import Identity
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter, RoomFilter
from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

# Page size of the scan for another sender's message (RFC §10.4 step 3). The
# scan stops at the first one, so a room with any history answers on its
# first page; only a room holding nothing but the sender's own words is read
# to the end.
_HISTORY_PAGE = 200

# The statuses that refuse new events (RFC §5.1): a room in one is no longer
# a routed message's.
_REFUSING = frozenset({RoomStatus.CLOSED, RoomStatus.ARCHIVED})


class InboundRoomRouter(ABC):
    """Route an inbound message to a room (or ``None`` for auto-create)."""

    @abstractmethod
    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
        identity: Identity | None = None,
    ) -> str | None:
        """Return room_id for the message, or ``None`` to create a new room.

        *participant_id* is the address naming the message's conversation
        (``Channel.conversation_address``): the sender's, or on a chat
        channel the chat it was posted in. *channel_data* is the message's
        metadata.

        *organization_id* is the organization the caller acts for (RFC
        §17.2), ``None`` when the call names none. *identity* is the one the
        store resolves *participant_id* to within that organization
        (``resolve_identity``), read by the framework once per message
        before the first route, or ``None``: the router and the check made
        when the sender is recorded name the sender alike (RFC §10.4).

        The answer is held to RFC §10.1 step 2's rules whoever gave it: a
        room of another organization than *organization_id* fails the
        message (``RoomNotFoundError``), and a room id the store lacks is
        created under that id. An override must accept both keywords; the
        framework passes them on every call.
        """
        ...


class DefaultInboundRoomRouter(InboundRoomRouter):
    """Default router (RFC §10.4): the sender's own room, then a channel bound
    to a single conversation that is open to the sender, else ``None``.

    Returns ``None`` rather than choosing when the message could belong to
    more than one room. A new room is recoverable; a message delivered into
    someone else's conversation is not — it is stored there, broadcast to that
    room's channels, and read back as context by that room's agent.
    """

    def __init__(
        self,
        store: ConversationStore,
        *,
        recipient_of: Callable[[ChannelBinding], str | None] | None = None,
    ) -> None:
        self._store = store
        # The address a binding delivers to, normalized (the kit passes its
        # channels' Channel.recipient_address); it names the correspondent.
        self._recipient_of = recipient_of

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
        identity: Identity | None = None,
    ) -> str | None:
        # The organization's rooms only, every room when the call names none
        # (RFC §10.4, §17.2): another tenant's room is no candidate, so it
        # neither wins nor makes the organization's one room ambiguous.
        own = known_as(participant_id, identity)
        # Step 1 (RFC §10.4): the sender's own room. Tried first because it
        # identifies the conversation, where a binding of the channel to some
        # room only identifies the pipe.
        if participant_id:
            identity_id = identity.id if identity is not None else None
            room_id = await self._senders_room(
                channel_id, participant_id, identity_id, organization_id
            )
            if room_id is not None:
                return room_id

        # Step 3: a channel dedicated to one conversation. Only when the
        # channel is bound to exactly one active room — a channel shared across
        # rooms (delegation, a room re-created after its predecessor closed, a
        # number shared by many correspondents) makes this ambiguous, and the
        # framework creating a fresh room is the safe answer — and only when
        # that room is open to this sender: on a shared number, the one room
        # bound to it is the first correspondent's conversation.
        candidates = await self._store.find_room_ids_by_channel(
            channel_id,
            RoomFilter(statuses=[RoomStatus.ACTIVE], organization_id=organization_id),
            limit=2,
        )
        if len(candidates) == 1:
            if await self._admits(candidates[0], channel_id, participant_id or None, own):
                return candidates[0]
            return None
        if len(candidates) > 1:
            # Debug, not a warning: on a number shared by many correspondents,
            # several rooms bound to the channel is the ordinary state, and
            # every new correspondent passes through here.
            logger.debug(
                "Channel %s is bound to %d active rooms — refusing to guess which one "
                "this message belongs to. Pass room_id explicitly, or install a custom "
                "InboundRoomRouter.",
                channel_id,
                len(candidates),
            )

        return None

    async def _senders_room(
        self,
        channel_id: str,
        sender: str,
        identity_id: str | None,
        organization_id: str | None,
    ) -> str | None:
        """The newest active room of *organization_id* that is *sender*'s own
        (RFC §10.4 step 1); *identity_id* is the identity the framework
        resolved the address to.

        First the room whose binding of this very channel names the sender: a
        binding speaks for its own channel, so a correspondent of one number
        is not taken to that room when they write to another. Otherwise the
        room where the sender is a participant reached through this channel:
        an id names its participant only on a channel they are reached
        through (RFC §5.5), so a number linked to someone's identity, or a
        sender id equal to a participant's on another channel, does not land
        in a room they joined elsewhere, which would then carry that room's
        messages to this sender. Within each lookup the sender is named by
        their address or their identity (``link_address``), and the newest
        room either name finds wins: an older room found by the address does
        not hide a newer one found by the identity.
        """
        step_one = RoomFilter(statuses=[RoomStatus.ACTIVE], organization_id=organization_id)
        room_id = await self._store.find_room_id_by_binding(
            channel_id, sender, identity_id, step_one
        )
        if room_id is not None:
            return room_id
        return await self._store.find_room_id_by_participant(
            channel_id, sender, identity_id, step_one
        )

    async def _admits(
        self, room_id: str, channel_id: str, sender: str | None, own: frozenset[str]
    ) -> bool:
        """Whether the one room bound to *channel_id* is open to *sender* (RFC §10.4).

        A group binding is open to everyone. Otherwise the room is another
        correspondent's conversation as soon as anything names one: the
        binding, a participant who joined through the channel, or a message
        the room received on it.
        """
        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            return False
        recipient = self._recipient_of(binding) if self._recipient_of is not None else None
        return await step_three_admits(self._store, binding, sender, own, recipient)


async def step_three_admits(
    store: ConversationStore,
    binding: ChannelBinding,
    sender: str | None,
    own: frozenset[str],
    recipient: str | None,
) -> bool:
    """Whether the room of *binding*, its binding of the channel, admits
    *sender* by RFC §10.4 step 3's conditions.

    *own* holds the ids the sender is known by (:func:`known_as`), and
    *recipient* the address the binding delivers to, in its channel's form.
    The default router asks it of the one room bound to the channel, and the
    framework again under the room lock before it records the sender there
    (§10.1 step 2): one predicate, so the two never drift apart.
    """
    if binding.group:
        return True
    if sender == SYSTEM_SENDER_ID:
        # The framework names the room it writes to, and nothing it writes
        # counts as a correspondent's: a sender borrowing the name, on a
        # channel whose senders choose their id, gets a room of its own.
        return False
    if not binding_admits(binding, own):
        return False
    if recipient is not None and recipient not in own:
        # The room delivers to someone else: it is their conversation.
        return False
    for participant in await store.list_participants(binding.room_id):
        if binding.channel_id in participant.connected_via and not own & {
            participant.id,
            participant.identity_id,
        }:
            return False
    return not await _heard_from_other(store, binding.room_id, binding.channel_id, own)


async def still_the_messages(
    store: ConversationStore,
    room: Room,
    channel_id: str,
    binding: ChannelBinding | None,
    sender: str | None,
    own: frozenset[str],
    recipient: str | None,
) -> bool:
    """Whether a routed message's *room*, read under the room lock, is still
    the message's (RFC §10.1 step 2, §10.4), whichever strategy chose it.

    *binding* is the room's binding of *channel_id*, *own* the ids the sender
    is known by and *recipient* the address the binding delivers to. Not the
    message's: a room whose status refuses events; a PAUSED room that is not
    the sender's conversation by step 1 (the binding names them, or a
    participant they are was reached through the channel) and whose binding
    is not ``group``; a binding, not ``group``, naming a correspondent other
    than the sender (its participant, else its recipient); a binding naming
    no one that the room no longer opens to the sender by step 3's
    conditions, unless the sender is a participant reached through the
    channel. The framework's own sender and a message with no address are
    no one's correspondent, and are recorded on no binding.
    """
    if room.status in _REFUSING:
        return False
    group = binding is not None and binding.group
    correspondent = None if binding is None or group else binding.participant_id or recipient
    if room.status != RoomStatus.PAUSED and (
        binding is None or group or (correspondent is not None and correspondent in own)
    ):
        return True
    members = [
        participant
        for participant in await store.list_participants(room.id)
        if channel_id in participant.connected_via
        and own & {participant.id, participant.identity_id}
    ]
    # A participant who is the sender names them by its id and its identity
    # too: a binding naming that participant names the sender.
    names = own | {name for member in members for name in (member.id, member.identity_id) if name}
    named = correspondent is not None and correspondent in names
    if room.status == RoomStatus.PAUSED and not (group or members or named):
        return False
    if binding is None or group:
        return True
    if correspondent is not None:
        return named
    if not recordable_sender(sender) or members:
        return True
    return await step_three_admits(store, binding, sender, own, recipient)


def known_as(sender: str | None, identity: Identity | None) -> frozenset[str]:
    """The ids a room may know *sender* by: the address, and its identity.

    Identity resolution names a participant and stamps an event with the
    identity's id, not the address (RFC §11), so the identity the store
    resolves the address to, within the caller's organization (§17.2), is
    the sender's own too.
    """
    if not sender:
        return frozenset()
    return frozenset({sender} if identity is None else {sender, identity.id})


def recordable_sender(sender: str | None) -> bool:
    """Whether *sender* can be a binding's correspondent (RFC §10.4).

    Not an empty address, and not the framework's own sender: what the host
    writes through ``deliver()`` names no correspondent.
    """
    return bool(sender) and sender != SYSTEM_SENDER_ID


def binding_names_no_one(binding: ChannelBinding) -> bool:
    """Whether *binding* carries one conversation and no correspondent yet."""
    return not binding.group and binding.participant_id is None


def binding_admits(binding: ChannelBinding, known_as: frozenset[str]) -> bool:
    """Whether *binding* lets a sender known by *known_as* in (RFC §10.4).

    A group binding lets everyone in; any other, the correspondent it names,
    or anyone while it names no one.
    """
    return binding.group or binding.participant_id is None or binding.participant_id in known_as


async def _heard_from_other(
    store: ConversationStore, room_id: str, channel_id: str, own: frozenset[str]
) -> bool:
    """Whether the room received a message on *channel_id* naming another sender.

    Received means stored and not ``BLOCKED`` (the default page skips those).
    An event whose source names no participant, or the framework's own sender
    (``SYSTEM_SENDER_ID``: the host speaking through ``deliver()``), is not a
    correspondent's; *own* holds the ids the sender is known by.
    """
    received = EventFilter(source_channel_id=channel_id)
    offset = 0
    while True:
        page = await store.list_events(
            room_id, offset=offset, limit=_HISTORY_PAGE, event_filter=received
        )
        for event in page:
            named = event.source.participant_id
            if named not in (None, SYSTEM_SENDER_ID) and named not in own:
                return True
        if len(page) < _HISTORY_PAGE:
            return False
        offset += len(page)
