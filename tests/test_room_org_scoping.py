"""Room operations can be scoped to one organization (RFC §17.2).

A library has no caller or auth context of its own, so the scope has to come
from the caller: the operations below take an optional `organization_id` and,
when given one, refuse a room belonging to anyone else.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit import Access, AgentResponsePolicy, RoomKit, RoomTimers
from roomkit.channels import SMSChannel
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import TextContent
from roomkit.models.identity import Identity
from roomkit.providers.sms.mock import MockSMSProvider
from tests.test_framework import SimpleChannel


async def _two_tenants() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="acme-room", organization_id="acme")
    await kit.create_room(room_id="globex-room", organization_id="globex")
    return kit


class TestScopedReads:
    async def test_matching_organization_reads_the_room(self) -> None:
        kit = await _two_tenants()
        room = await kit.get_room("acme-room", organization_id="acme")
        assert room.id == "acme-room"

    async def test_other_organization_reads_as_not_found(self) -> None:
        """Reported as missing, not as "wrong organization": a distinct error
        would let a caller probe which room ids exist outside its tenant."""
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.get_room("globex-room", organization_id="acme")

    async def test_unscoped_read_is_unchanged(self) -> None:
        kit = await _two_tenants()
        assert (await kit.get_room("globex-room")).id == "globex-room"

    async def test_timeline_is_scoped(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.get_timeline("globex-room", organization_id="acme")
        assert await kit.get_timeline("globex-room", organization_id="globex") is not None


class TestScopedMutations:
    async def test_attach_channel_is_scoped(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.attach_channel("globex-room", "sms1", organization_id="acme")
        # The refusal left nothing behind.
        assert await kit.store.get_binding("globex-room", "sms1") is None

        binding = await kit.attach_channel("globex-room", "sms1", organization_id="globex")
        assert binding.channel_id == "sms1"

    async def test_detach_channel_is_scoped(self) -> None:
        kit = await _two_tenants()
        await kit.attach_channel("globex-room", "sms1")

        with pytest.raises(RoomNotFoundError):
            await kit.detach_channel("globex-room", "sms1", organization_id="acme")
        assert await kit.store.get_binding("globex-room", "sms1") is not None

        assert await kit.detach_channel("globex-room", "sms1", organization_id="globex") is True

    async def test_close_and_archive_are_scoped(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.close_room("globex-room", organization_id="acme")
        with pytest.raises(RoomNotFoundError):
            await kit.archive_room("globex-room", organization_id="acme")

        assert (await kit.get_room("globex-room")).status.value == "active"

    async def test_send_event_is_scoped(self) -> None:
        kit = await _two_tenants()
        await kit.attach_channel("globex-room", "sms1")

        with pytest.raises(RoomNotFoundError):
            await kit.send_event(
                "globex-room",
                "sms1",
                TextContent(body="from the wrong tenant"),
                organization_id="acme",
            )
        assert await kit.get_timeline("globex-room") == await kit.get_timeline("globex-room")

        event = await kit.send_event(
            "globex-room",
            "sms1",
            TextContent(body="from the right tenant"),
            organization_id="globex",
        )
        assert event.id

    async def test_binding_operations_share_the_room_scope(self) -> None:
        kit = await _two_tenants()
        await kit.attach_channel("globex-room", "sms1")

        operations = [
            kit.mute("globex-room", "sms1", organization_id="acme"),
            kit.unmute("globex-room", "sms1", organization_id="acme"),
            kit.mute_output("globex-room", "sms1", organization_id="acme"),
            kit.unmute_output("globex-room", "sms1", organization_id="acme"),
            kit.set_visibility("globex-room", "sms1", "none", organization_id="acme"),
            kit.set_access("globex-room", "sms1", Access.NONE, organization_id="acme"),
            kit.update_binding_metadata(
                "globex-room", "sms1", {"leaked": True}, organization_id="acme"
            ),
            kit.get_binding("globex-room", "sms1", organization_id="acme"),
            kit.list_bindings("globex-room", organization_id="acme"),
        ]
        for operation in operations:
            with pytest.raises(RoomNotFoundError):
                await operation

        binding = await kit.get_binding("globex-room", "sms1")
        assert binding.access is Access.READ_WRITE
        assert binding.muted is False
        assert binding.output_muted is False
        assert "leaked" not in binding.metadata

    async def test_lifecycle_and_participant_operations_share_the_room_scope(self) -> None:
        kit = await _two_tenants()

        operations = [
            kit.set_room_timers(
                "globex-room", RoomTimers(closed_after_seconds=10), organization_id="acme"
            ),
            kit.set_agent_response_policy(
                "globex-room",
                AgentResponsePolicy.ADDRESSED_ONLY,
                organization_id="acme",
            ),
            kit.check_room_timers("globex-room", organization_id="acme"),
            kit.update_room_metadata("globex-room", {"leaked": True}, organization_id="acme"),
            kit.ensure_participant("globex-room", "sms1", "p1", organization_id="acme"),
            kit.resolve_participant("globex-room", "p1", "identity-1", organization_id="acme"),
        ]
        for operation in operations:
            with pytest.raises(RoomNotFoundError):
                await operation

        room = await kit.get_room("globex-room")
        assert "leaked" not in room.metadata
        assert await kit.store.get_participant("globex-room", "p1") is None

    async def test_query_and_read_tracking_operations_share_the_room_scope(self) -> None:
        kit = await _two_tenants()

        operations = [
            kit.list_tasks("globex-room", organization_id="acme"),
            kit.list_observations("globex-room", organization_id="acme"),
            kit.mark_read("globex-room", "sms1", "event-1", organization_id="acme"),
            kit.mark_all_read("globex-room", "sms1", organization_id="acme"),
            kit.list_read_markers("globex-room", organization_id="acme"),
        ]
        for operation in operations:
            with pytest.raises(RoomNotFoundError):
                await operation

        assert await kit.list_read_markers("globex-room") == {}

    async def test_check_all_timers_can_be_limited_to_one_tenant(self) -> None:
        kit = await _two_tenants()
        expired = RoomTimers(
            inactive_after_seconds=1,
            last_activity_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        await kit.set_room_timers("globex-room", expired)

        transitioned = await kit.check_all_timers(organization_id="acme")

        assert transitioned == []
        assert (await kit.get_room("globex-room")).status.value == "active"


class TestUnscopedRooms:
    async def test_a_room_without_an_organization_is_reachable_unscoped_only(self) -> None:
        """A room created with no organization belongs to no tenant, so a
        scoped caller does not reach it."""
        kit = RoomKit()
        await kit.create_room(room_id="solo")

        assert (await kit.get_room("solo")).id == "solo"
        with pytest.raises(RoomNotFoundError):
            await kit.get_room("solo", organization_id="acme")


def _message() -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="+15550001", content=TextContent(body="hi"))


class _AnswersWith(InboundRoomRouter):
    """A custom router giving one answer, recording the organization and the
    identity it was given."""

    def __init__(self, room_id: str | None) -> None:
        self.room_id = room_id
        self.asked: list[tuple[str | None, str | None]] = []

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
        self.asked.append((organization_id, identity.id if identity is not None else None))
        return self.room_id


class TestScopedInbound:
    """``process_inbound`` scopes the room a message lands in: routed or
    explicit, a room of another organization is not found to the caller, before
    any event is committed or any channel auto-attached."""

    async def test_an_explicit_room_of_another_organization_is_not_found(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.process_inbound(_message(), room_id="globex-room", organization_id="acme")
        # The refusal left nothing behind: no event, and no auto-attach either.
        assert await kit.store.list_events("globex-room") == []
        assert await kit.store.get_binding("globex-room", "sms1") is None

    async def test_a_matching_organization_processes_the_message(self) -> None:
        kit = await _two_tenants()
        result = await kit.process_inbound(_message(), room_id="acme-room", organization_id="acme")
        assert result.event is not None and result.event.room_id == "acme-room"

    async def test_a_room_that_does_not_exist_is_created_under_the_caller(self) -> None:
        kit = await _two_tenants()
        await kit.process_inbound(_message(), room_id="fresh-room", organization_id="acme")
        assert (await kit.get_room("fresh-room")).organization_id == "acme"

    @pytest.mark.parametrize("names_the_sender", [True, False], ids=["step-1", "step-3"])
    async def test_the_default_router_never_returns_another_organizations_room(
        self, names_the_sender: bool
    ) -> None:
        """RFC §10.4: the default strategy searches the caller's organization
        only, so the sender's room at another tenant, or the one room of
        another tenant bound to the channel, is not a candidate: the message
        gets a room of its own organization."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="globex-room", organization_id="globex")
        sender = _message().sender_id if names_the_sender else None
        await kit.attach_channel("globex-room", "sms1", participant_id=sender)
        before = await kit.store.list_events("globex-room")

        result = await kit.process_inbound(_message(), organization_id="acme")

        assert result.event is not None and result.event.room_id != "globex-room"
        assert (await kit.get_room(result.event.room_id)).organization_id == "acme"
        assert await kit.store.list_events("globex-room") == before

    async def test_a_router_answering_another_organizations_room_fails_the_message(
        self,
    ) -> None:
        """The answer is held to step 2's rules whoever gave it (RFC §10.1)."""
        router = _AnswersWith("globex-room")
        kit = RoomKit(inbound_router=router)
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="globex-room", organization_id="globex")
        await kit.attach_channel("globex-room", "sms1")
        before = await kit.store.list_events("globex-room")

        with pytest.raises(RoomNotFoundError):
            await kit.process_inbound(_message(), organization_id="acme")

        assert await kit.store.list_events("globex-room") == before
        binding = await kit.store.get_binding("globex-room", "sms1")
        assert binding is not None and binding.participant_id is None

    async def test_a_room_a_router_answers_that_does_not_exist_is_created_with_its_id(
        self,
    ) -> None:
        kit = RoomKit(inbound_router=_AnswersWith("fresh-room"))
        kit.register_channel(SimpleChannel("sms1"))

        result = await kit.process_inbound(_message(), organization_id="acme")

        assert result.event is not None and result.event.room_id == "fresh-room"
        assert (await kit.get_room("fresh-room")).organization_id == "acme"
        binding = await kit.store.get_binding("fresh-room", "sms1")
        assert binding is not None and binding.participant_id == _message().sender_id

    async def test_the_router_is_given_the_organization_and_the_identity(self) -> None:
        """Read once, scoped to the caller's organization (RFC §10.4, §17.2):
        an identity of an organization is no one's to an unscoped call."""
        router = _AnswersWith(None)
        kit = RoomKit(inbound_router=router)
        kit.register_channel(SimpleChannel("sms1"))
        await kit.store.create_identity(Identity(id="id-alice", organization_id="acme"))
        await kit.store.link_address("id-alice", "sms", _message().sender_id, "acme")

        await kit.process_inbound(_message(), organization_id="acme")
        await kit.process_inbound(_message())

        assert router.asked == [("acme", "id-alice"), (None, None)]

    async def test_an_address_linked_within_an_organization_finds_its_persons_room(
        self,
    ) -> None:
        """The host added alice under her identity, then linked her number in
        its organization: her first text from that number lands in her room,
        which only step 1, by the identity, can find (two rooms share the
        number)."""
        kit = RoomKit()
        kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
        for room_id in ("alice-room", "carol-room"):
            await kit.create_room(room_id=room_id, organization_id="acme")
        await kit.attach_channel("carol-room", "sms", participant_id="+15550000003")
        await kit.attach_channel("alice-room", "sms")
        await kit.add_member("alice-room", "sms", "id-alice")
        await kit.store.create_identity(Identity(id="id-alice", organization_id="acme"))
        await kit.store.link_address("id-alice", str(ChannelType.SMS), "+15550000001", "acme")

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")
            ),
            organization_id="acme",
        )

        assert result.event is not None and result.event.room_id == "alice-room"

    async def test_unscoped_processing_is_unchanged(self) -> None:
        kit = await _two_tenants()
        result = await kit.process_inbound(_message(), room_id="globex-room")
        assert result.event is not None and result.event.room_id == "globex-room"

    async def test_a_routed_miss_creates_the_room_under_the_caller(self) -> None:
        """No room routes for this channel: the auto-created one belongs to
        the caller's organization."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        result = await kit.process_inbound(_message(), organization_id="acme")
        assert (await kit.get_room(result.event.room_id)).organization_id == "acme"

    @pytest.mark.parametrize("organization_id", [None, "acme", "globex"])
    async def test_a_named_room_created_meanwhile_is_read_again(
        self, monkeypatch: pytest.MonkeyPatch, organization_id: str | None
    ) -> None:
        """RFC §10.1 step 2, §14.3: a creation the store refuses because
        another call created the room meanwhile reads the room again, as one
        that exists, checked for the caller's organization; the room it
        holds is never replaced."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="shared", organization_id="acme", metadata={"k": "v"})
        # The existence read is stale: room_exists unscoped, get_room scoped.
        reads = {"stale": organization_id is not None}
        real_get_room = kit.store.get_room

        async def not_yet(room_id: str) -> bool:
            return False

        async def stale_once(room_id: str) -> Any:
            if reads.pop("stale", False):
                return None
            return await real_get_room(room_id)

        monkeypatch.setattr(kit.store, "room_exists", not_yet)
        monkeypatch.setattr(kit.store, "get_room", stale_once)

        if organization_id == "globex":
            with pytest.raises(RoomNotFoundError):
                await kit.process_inbound(
                    _message(), room_id="shared", organization_id=organization_id
                )
        else:
            result = await kit.process_inbound(
                _message(), room_id="shared", organization_id=organization_id
            )
            assert result.event is not None and result.event.room_id == "shared"
        room = await real_get_room("shared")
        assert room is not None and (room.organization_id, room.metadata) == ("acme", {"k": "v"})
        await kit.close()

    async def test_a_deferred_send_to_another_organization_is_refused_synchronously(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.process_inbound(
                _message(), room_id="globex-room", organization_id="acme", defer_delivery=True
            )
        assert await kit.store.list_events("globex-room") == []

    async def test_a_matching_organization_still_auto_attaches_the_channel(self) -> None:
        kit = await _two_tenants()
        assert await kit.store.get_binding("acme-room", "sms1") is None
        await kit.process_inbound(_message(), room_id="acme-room", organization_id="acme")
        assert await kit.store.get_binding("acme-room", "sms1") is not None
