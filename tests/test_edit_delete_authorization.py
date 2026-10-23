"""Authorization on inbound EDIT/DELETE (RFC §10.3).

The framework, not the payload, decides who may rewrite a room's history. A
channel that lets a remote party choose ``edit_source`` or ``delete_type``
must not thereby let them choose their own authority.
"""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelType,
    DeleteType,
    EventDirection,
    EventStatus,
    EventType,
    HookTrigger,
    ParticipantRole,
    ParticipantStatus,
)
from roomkit.models.event import (
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.hook import HookResult


class _Transport(Channel):
    """Transport that forwards ``event_type``/``content`` verbatim.

    This is the realistic shape of a WebSocket or transport channel: the
    remote party controls the payload, so it also controls ``edit_source``
    and ``delete_type``.
    """

    channel_type = ChannelType.WEBSOCKET

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                # A provider's echo of the channel's own output travels outbound.
                direction=message.metadata.get("direction", EventDirection.INBOUND),
                participant_id=message.sender_id or None,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _SystemTransport(_Transport):
    channel_type = ChannelType.SYSTEM


class _Editable(_Transport):
    """A transport that applies edits and deletes natively, and records what
    it is handed."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(supports_edit=True, supports_delete=True)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


async def _setup() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(_Transport("src"))
    kit.register_channel(_SystemTransport("sys"))
    await kit.create_room(room_id="r1")
    # Author and non-author write on one channel: a group binding (RFC §10.4).
    await kit.attach_channel("r1", "src", group=True)
    await kit.attach_channel("r1", "sys")
    return kit


async def _post(kit: RoomKit, body: str, sender_id: str) -> RoomEvent:
    result = await kit.process_inbound(
        InboundMessage(channel_id="src", sender_id=sender_id, content=TextContent(body=body))
    )
    assert result.event is not None
    return result.event


async def _edit(
    kit: RoomKit,
    target: RoomEvent,
    *,
    sender_id: str,
    edit_source: str | None,
    channel_id: str = "src",
):
    return await kit.process_inbound(
        InboundMessage(
            channel_id=channel_id,
            sender_id=sender_id,
            event_type=EventType.EDIT,
            content=EditContent(
                target_event_id=target.id,
                new_content=TextContent(body="rewritten"),
                edit_source=edit_source,
            ),
        )
    )


async def _delete(
    kit: RoomKit,
    target: RoomEvent,
    *,
    sender_id: str,
    delete_type: DeleteType,
    channel_id: str = "src",
    direction: EventDirection = EventDirection.INBOUND,
):
    return await kit.process_inbound(
        InboundMessage(
            channel_id=channel_id,
            sender_id=sender_id,
            event_type=EventType.DELETE,
            content=DeleteContent(target_event_id=target.id, delete_type=delete_type),
            metadata={"direction": direction},
        )
    )


class TestEditAuthorization:
    async def test_author_may_edit_own_message(self) -> None:
        """The ordinary path must keep working."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(kit, original, sender_id="alice", edit_source="sender")

        assert result.blocked is False
        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content.body == "rewritten"

    async def test_non_author_cannot_edit(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(kit, original, sender_id="mallory", edit_source="sender")

        assert result.blocked is True
        assert result.reason == "not_original_author"

    async def test_unknown_edit_source_does_not_skip_the_author_check(self) -> None:
        """An ``edit_source`` outside the RFC vocabulary is not a privilege."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        for forged in ("admin", "system_", "Sender", "alice", ""):
            result = await _edit(kit, original, sender_id="mallory", edit_source=forged)

            assert result.blocked is True, f"edit_source={forged!r} bypassed authorization"
            assert result.reason == "not_original_author"

        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content.body == "v1"

    async def test_participant_id_as_edit_source_still_requires_authorship(self) -> None:
        """The pattern the examples used to show must not grant authority."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        # Alice editing her own message with her id as edit_source: allowed.
        mine = await _edit(kit, original, sender_id="alice", edit_source="alice")
        assert mine.blocked is False
        # Mallory doing the same on Alice's message: refused.
        theirs = await _edit(kit, original, sender_id="mallory", edit_source="mallory")
        assert theirs.blocked is True

    async def test_system_edit_source_requires_a_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(kit, original, sender_id="mallory", edit_source="system")

        assert result.blocked is True
        assert result.reason == "not_authorized"

    async def test_system_edit_source_allowed_from_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(
            kit, original, sender_id="moderator", edit_source="system", channel_id="sys"
        )

        assert result.blocked is False


class TestDeleteAuthorization:
    async def test_author_may_delete_own_message(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)

        assert result.blocked is False

    async def test_non_author_cannot_delete_as_sender(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="mallory", delete_type=DeleteType.SENDER)

        assert result.blocked is True
        assert result.reason == "not_original_author"

    async def test_admin_delete_requires_verified_authority(self) -> None:
        """Claiming ADMIN is not the same as holding it."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="mallory", delete_type=DeleteType.ADMIN)

        assert result.blocked is True
        assert result.reason == "not_authorized"
        stored = await kit._store.get_event(original.id)
        assert stored is not None
        assert stored.metadata.get("deleted") is not True

    async def test_admin_delete_allowed_for_room_owner(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        await kit.add_member("r1", "src", "owner-1", role=ParticipantRole.OWNER)

        result = await _delete(kit, original, sender_id="owner-1", delete_type=DeleteType.ADMIN)

        assert result.blocked is False

    async def test_system_delete_requires_a_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="mallory", delete_type=DeleteType.SYSTEM)

        assert result.blocked is True
        assert result.reason == "not_authorized"

    async def test_system_delete_allowed_from_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(
            kit, original, sender_id="janitor", delete_type=DeleteType.SYSTEM, channel_id="sys"
        )

        assert result.blocked is False


def _hidden_changes(target_id: str) -> dict[str, tuple[EventType, EventContent]]:
    """Events holding a change they do not make (RFC §10.3 rule 1), by name."""
    edit = EditContent(target_event_id=target_id, new_content=TextContent(body="rewritten"))
    delete = DeleteContent(target_event_id=target_id)
    return {
        "a message carrying an edit": (EventType.MESSAGE, edit),
        "a message carrying a delete": (EventType.MESSAGE, delete),
        "an edit carrying text": (EventType.EDIT, TextContent(body="rewritten")),
        "a delete carrying an edit": (EventType.DELETE, edit),
        "an edit carrying a delete": (EventType.EDIT, delete),
        "a composite part": (
            EventType.MESSAGE,
            CompositeContent(parts=[TextContent(body="hi"), edit]),
        ),
        "an edit's new content": (EventType.EDIT, edit.model_copy(update={"new_content": delete})),
        "a composite in an edit's new content": (
            EventType.EDIT,
            edit.model_copy(update={"new_content": CompositeContent(parts=[delete])}),
        ),
    }


class TestWhatAChangeIs:
    """RFC §10.3 rule 1: an event is an edit when its type is EDIT and its
    content an EditContent, a delete when DELETE and DeleteContent; any other
    event holding a change, at any depth, is refused before a channel can
    apply it natively."""

    @pytest.mark.parametrize("case", list(_hidden_changes("")))
    async def test_a_change_held_where_it_is_not_made_is_refused(self, case: str) -> None:
        kit = await _setup()
        peer = _Editable("peer")
        kit.register_channel(peer)
        await kit.attach_channel("r1", "peer")
        original = await _post(kit, "v1", sender_id="alice")
        event_type, content = _hidden_changes(original.id)[case]

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="src", sender_id="alice", event_type=event_type, content=content
            )
        )

        assert (result.blocked, result.reason) == (True, "target_event_not_found")
        assert result.event is None
        assert [e.id for e in peer.delivered] == [original.id]
        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content == TextContent(body="v1")
        assert not stored.metadata
        await kit.close()


class TestWhatAChangeMayTarget:
    """RFC §10.3 rule 2: the target, read under the room lock, is an event of
    the room that is not BLOCKED, not deleted and not itself a change."""

    async def test_a_refused_message_is_no_target(self) -> None:
        kit = await _setup()

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def no_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and event.content.body == "spam":
                return HookResult.block(reason="spam")
            return HookResult.allow()

        refused = (
            await kit.process_inbound(
                InboundMessage(
                    channel_id="src", sender_id="alice", content=TextContent(body="spam")
                )
            )
        ).event
        assert refused is not None and refused.status == EventStatus.BLOCKED

        edit = await _edit(kit, refused, sender_id="alice", edit_source="sender")
        delete = await _delete(kit, refused, sender_id="alice", delete_type=DeleteType.SENDER)

        assert (edit.reason, delete.reason) == ("target_event_not_found",) * 2
        stored = await kit._store.get_event(refused.id)
        assert stored is not None and not stored.metadata
        await kit.close()

    @pytest.mark.parametrize("change", ["edit", "delete"])
    async def test_a_deleted_message_is_no_target(self, change: str) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        first = await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)
        assert first.blocked is False

        if change == "edit":
            result = await _edit(kit, original, sender_id="alice", edit_source="sender")
        else:
            result = await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)

        assert (result.blocked, result.reason) == (True, "target_event_not_found")
        assert result.event is None
        await kit.close()

    @pytest.mark.parametrize("change", ["edit", "delete"])
    async def test_a_change_is_no_target(self, change: str) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        first = await _edit(kit, original, sender_id="alice", edit_source="sender")
        assert first.event is not None

        if change == "edit":
            result = await _edit(kit, first.event, sender_id="alice", edit_source="sender")
        else:
            result = await _delete(
                kit, first.event, sender_id="alice", delete_type=DeleteType.SENDER
            )

        assert (result.blocked, result.reason) == (True, "target_event_not_found")
        await kit.close()

    async def test_an_edited_message_may_be_edited_again(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        await _edit(kit, original, sender_id="alice", edit_source="sender")

        again = await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="alice",
                event_type=EventType.EDIT,
                content=EditContent(
                    target_event_id=original.id, new_content=TextContent(body="third")
                ),
            )
        )

        assert again.blocked is False
        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content == TextContent(body="third")
        await kit.close()

    async def test_a_retried_delete_is_answered_as_the_first(self) -> None:
        """A client that retries a delete gives it a key (RFC §13.4)."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        retry = InboundMessage(
            channel_id="src",
            sender_id="alice",
            event_type=EventType.DELETE,
            content=DeleteContent(target_event_id=original.id),
            idempotency_key="delete-1",
        )

        first = await kit.process_inbound(retry)
        second = await kit.process_inbound(retry)

        assert first.event is not None and first.blocked is False
        assert second.duplicate is True and second.event is not None
        assert second.event.id == first.event.id
        await kit.close()


class TestWhoMayChange:
    """RFC §10.3 rules 3 to 6: who authored an event, and who holds the
    authority a change claims."""

    @pytest.mark.parametrize(
        ("role", "status", "allowed"),
        [
            (ParticipantRole.OWNER, ParticipantStatus.ACTIVE, True),
            (ParticipantRole.AGENT, ParticipantStatus.ACTIVE, True),
            (ParticipantRole.MEMBER, ParticipantStatus.ACTIVE, False),
            (ParticipantRole.OWNER, ParticipantStatus.LEFT, False),
            (ParticipantRole.OWNER, ParticipantStatus.BANNED, False),
            (ParticipantRole.AGENT, ParticipantStatus.BANNED, False),
        ],
    )
    async def test_an_admin_delete_is_an_active_owner_s_or_agent_s(
        self, role: ParticipantRole, status: ParticipantStatus, allowed: bool
    ) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        moderator = await kit.add_member("r1", "src", "mod-1", role=role)
        await kit._store.update_participant(moderator.model_copy(update={"status": status}))

        result = await _delete(kit, original, sender_id="mod-1", delete_type=DeleteType.ADMIN)

        assert (result.blocked, result.reason) == (
            (False, None) if allowed else (True, "not_authorized")
        )
        await kit.close()

    async def test_an_admin_delete_from_no_one_needs_an_identity(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="", delete_type=DeleteType.ADMIN)

        assert (result.blocked, result.reason) == (True, "identity_required_for_edit")
        await kit.close()

    @pytest.mark.parametrize(
        ("delete_type", "sender_id", "reason"),
        [(DeleteType.ADMIN, "owner-1", "not_authorized"), (DeleteType.SENDER, "alice", None)],
        ids=["an owner's authority", "a participant's authorship"],
    )
    async def test_a_channel_s_own_output_speaks_as_no_participant(
        self, delete_type: DeleteType, sender_id: str, reason: str | None
    ) -> None:
        """An event that travelled outbound is its channel's, whatever
        participant it names (RFC §10.3 rule 3): it neither holds that
        participant's authority nor is the author of their messages."""
        kit = await _setup()
        await kit.add_member("r1", "src", "owner-1", role=ParticipantRole.OWNER)
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(
            kit,
            original,
            sender_id=sender_id,
            delete_type=delete_type,
            direction=EventDirection.OUTBOUND,
        )

        assert (result.blocked, result.reason) == (True, reason or "not_original_author")
        stored = await kit._store.get_event(original.id)
        assert stored is not None and not stored.metadata
        await kit.close()

    async def test_a_system_change_needs_no_author(self) -> None:
        """A SYSTEM change claims the system's authority (rule 6): neither it
        nor its target needs an author, as a sender's change does (rule 4)."""
        kit = await _setup()
        notice = (
            await kit.process_inbound(
                InboundMessage(channel_id="sys", sender_id="", content=TextContent(body="notice"))
            )
        ).event
        assert notice is not None and notice.source.participant_id is None

        result = await _delete(
            kit, notice, sender_id="", delete_type=DeleteType.SYSTEM, channel_id="sys"
        )

        assert (result.blocked, result.reason) == (False, None)
        await kit.close()
