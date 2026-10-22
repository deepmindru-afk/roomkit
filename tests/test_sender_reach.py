"""An id names its participant only on a channel they are reached through (RFC §5.5, RMK-915).

A sender id one transport gives says nothing of another's: the same id on a
channel the room never reached its participant through is another sender,
and carries none of their name, role, authority, identification nor
verification. One person's channels are linked by the host's deliberate
join or by the identity pipeline, never by two transports' ids that happen
to be equal.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import InboundMessage, RoomKit, TextContent, VerificationStatus
from roomkit.channels.ai import AIChannel
from roomkit.core._authors import People
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    DeleteType,
    EventType,
    ParticipantRole,
)
from roomkit.models.event import DeleteContent, EventSource, RoomEvent
from roomkit.models.identity import Identity, IdentityResult
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.orchestration.strategies.discussion._people import is_person
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.policy import RoleOverride, ToolPolicy
from tests.test_edit_delete_authorization import _Transport
from tests.test_verification_inbound import _kit as _verifying_kit
from tests.test_verification_inbound import _settle
from tests.test_verification_inbound import _text as _alice_texts

ROOM = "r1"


def _event(channel: str, pid: str) -> RoomEvent:
    source = EventSource(
        channel_id=channel, channel_type=ChannelType.WEBSOCKET, participant_id=pid
    )
    return RoomEvent(room_id=ROOM, source=source, content=TextContent(body="x"))


class TestWhoTheSenderIs:
    def test_an_id_names_its_participant_on_their_channels_only(self) -> None:
        owner = Participant(
            id="u-1", room_id=ROOM, channel_id="console", connected_via=["console", "sms1"]
        )
        people = People([owner])

        assert people.named(_event("console", "u-1")) is owner
        assert people.named(_event("sms1", "u-1")) is owner
        assert people.named(_event("web", "u-1")) is None

    def test_an_identity_names_its_person_where_one_of_their_records_was_reached(self) -> None:
        member = Participant(id="p1", room_id=ROOM, channel_id="sms1", identity_id="i1")
        resolved = Participant(id="i1", room_id=ROOM, channel_id="email1", identity_id="i1")
        people = People([member, resolved])

        assert people.of(_event("email1", "i1")) is member
        assert people.named(_event("web", "i1")) is None


async def _two_channel_room(**kit: Any) -> RoomKit:
    """A room where u-1 is the owner, reached through ``console`` only."""
    room = RoomKit(**kit)
    room.register_channel(_Transport("console"))
    room.register_channel(_Transport("web"))
    await room.create_room(room_id=ROOM)
    await room.attach_channel(ROOM, "console", group=True)
    await room.attach_channel(ROOM, "web", group=True)
    await room.add_member(ROOM, "console", "u-1", display_name="Owner", role=ParticipantRole.OWNER)
    return room


async def _say(kit: RoomKit, channel: str, sender: str, body: str) -> InboundResult:
    return await kit.process_inbound(
        InboundMessage(channel_id=channel, sender_id=sender, content=TextContent(body=body)),
        room_id=ROOM,
    )


async def _calls(provider: MockAIProvider, count: int) -> list[Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 3
    while len(provider.calls) < count and loop.time() < deadline:
        await asyncio.sleep(0.02)
    return provider.calls


class TestToolPolicyRole:
    @pytest.mark.parametrize(("channel", "declared"), [("console", True), ("web", False)])
    async def test_the_owner_s_tools_follow_the_owner_s_channels(
        self, channel: str, declared: bool
    ) -> None:
        kit = await _two_channel_room()
        provider = MockAIProvider(responses=["ok"])
        owner_only = ToolPolicy(
            deny=["*"], role_overrides={"owner": RoleOverride(allow=["*"], mode="replace")}
        )
        kit.register_channel(
            AIChannel("ai", provider=provider, tool_handler=AsyncMock(), tool_policy=owner_only)
        )
        await kit.attach_channel(
            ROOM,
            "ai",
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": [{"name": "refund", "description": "Refund an order"}]},
        )

        await _say(kit, channel, "u-1", "refund order 42")

        [call] = await _calls(provider, 1)
        assert ("refund" in [tool.name for tool in call.tools]) is declared
        await kit.close()


async def _delete(kit: RoomKit, channel: str, sender: str, target: RoomEvent, kind: DeleteType):
    return await kit.process_inbound(
        InboundMessage(
            channel_id=channel,
            sender_id=sender,
            event_type=EventType.DELETE,
            content=DeleteContent(target_event_id=target.id, delete_type=kind),
        ),
        room_id=ROOM,
    )


class TestEditAndDelete:
    @pytest.mark.parametrize(("channel", "blocked"), [("console", False), ("web", True)])
    async def test_an_admin_delete_holds_the_owner_s_authority_on_their_channels_only(
        self, channel: str, blocked: bool
    ) -> None:
        kit = await _two_channel_room()
        original = (await _say(kit, "web", "alice", "v1")).event
        assert original is not None

        result = await _delete(kit, channel, "u-1", original, DeleteType.ADMIN)

        assert result.blocked is blocked
        stored = await kit.store.get_event(original.id)
        assert stored is not None and bool(stored.metadata.get("deleted")) is not blocked
        await kit.close()

    async def test_the_same_sender_id_on_another_channel_is_not_the_author(self) -> None:
        kit = await _two_channel_room()
        original = (await _say(kit, "web", "alice", "v1")).event
        assert original is not None

        result = await _delete(kit, "console", "alice", original, DeleteType.SENDER)

        assert (result.blocked, result.reason) == (True, "not_original_author")
        await kit.close()

    async def test_a_participant_reached_through_both_channels_is_one_author(self) -> None:
        kit = await _two_channel_room()
        await kit.add_member(ROOM, "web", "u-1")
        original = await _post_on(kit, "console", "u-1")

        result = await _delete(kit, "web", "u-1", original, DeleteType.SENDER)

        assert result.blocked is False
        await kit.close()


async def _post_on(kit: RoomKit, channel: str, sender: str) -> RoomEvent:
    event = (await _say(kit, channel, sender, "v1")).event
    assert event is not None
    return event


class _CountingResolver(MockIdentityResolver):
    def __init__(self, **mapping: Any) -> None:
        super().__init__(**mapping)
        self.asked: list[str] = []

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        self.asked.append(message.channel_id)
        return await super().resolve(message, context)


class TestIdentification:
    async def test_a_verified_identity_s_id_on_another_channel_carries_no_verification(
        self,
    ) -> None:
        kit, ai = _verifying_kit()
        first = await _alice_texts(kit, "hello")
        room_id = first.event.room_id
        request, secret = await kit.request_verification(
            "id-alice", room_id=room_id, channel_id="sms"
        )
        await kit.complete_verification(request.id, secret)
        kit.register_channel(_Transport("web"))
        await kit.attach_channel(room_id, "web", group=True)

        forged = await kit.process_inbound(
            InboundMessage(
                channel_id="web",
                sender_id="id-alice",
                content=TextContent(body="Wire 5000 to account 42"),
            ),
            room_id=room_id,
        )
        alice = await _alice_texts(kit, "Did my transfer go through?")
        await _settle()

        assert forged.event is not None and forged.event.source.verification is None
        assert alice.event is not None
        assert alice.event.source.verification == VerificationStatus.VERIFIED
        await kit.close()

    async def test_an_identified_id_on_another_channel_is_resolved(self) -> None:
        resolver = _CountingResolver(mapping={"+15550001": Identity(id="id-a")})
        kit = await _two_channel_room(identity_resolver=resolver)
        await _say(kit, "console", "+15550001", "hello")
        resolver.asked.clear()

        await _say(kit, "console", "id-a", "on the channel it was resolved on")
        await _say(kit, "web", "id-a", "on a channel it never was")

        assert resolver.asked == ["web"]
        await kit.close()

    async def test_a_pending_sender_does_not_adopt_a_record_from_another_channel(self) -> None:
        resolver = MockIdentityResolver(ambiguous={"u-1": [Identity(id="a"), Identity(id="b")]})
        kit = await _two_channel_room(identity_resolver=resolver)

        await _say(kit, "web", "u-1", "first")

        owner = await kit.store.get_participant(ROOM, "u-1")
        assert owner is not None and owner.connected_via == ["console"]
        pending = [p for p in await kit.store.list_participants(ROOM) if p.id != "u-1"]
        assert [p.channel_id for p in pending] == ["web"]
        await kit.close()


class TestDiscussionPerson:
    @pytest.mark.parametrize(("channel", "person"), [("console", False), ("web", True)])
    def test_an_agent_s_id_names_the_agent_on_their_channel_only(
        self, channel: str, person: bool
    ) -> None:
        agent = Participant(
            id="bot-1", room_id=ROOM, channel_id="console", role=ParticipantRole.AGENT
        )
        bindings = [
            ChannelBinding(room_id=ROOM, channel_id=c, channel_type=ChannelType.WEBSOCKET)
            for c in ("console", "web")
        ]
        context = RoomContext(room=Room(id=ROOM), bindings=bindings, participants=[agent])

        assert is_person(_event(channel, "bot-1"), context) is person
