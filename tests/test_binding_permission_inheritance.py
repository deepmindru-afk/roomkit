"""A binding is never widened implicitly (RFC §7.5-6, §7.5-7).

Two ways the framework can hand out more access than an integrator granted:
by deriving a binding from another and letting the gaps fall back to defaults,
and by re-attaching a channel that was deliberately detached.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import Access, ChannelType, HookTrigger, RoomStatus, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import RoomTimers
from roomkit.store.memory import InMemoryStore


class _Transport(Channel):
    channel_type = ChannelType.WEBSOCKET

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class TestSharedChannelInheritsPermissions:
    """Sharing a channel into a delegated room must not promote it."""

    async def _kit(self) -> RoomKit:
        kit = RoomKit()
        kit.register_channel(_Transport("observer"))
        await kit.create_room(room_id="parent")
        return kit

    async def test_read_only_stays_read_only_in_the_child_room(self) -> None:
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", access=Access.READ_ONLY)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.access == Access.READ_ONLY

    async def test_restricted_visibility_is_carried_over(self) -> None:
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", visibility=Visibility.NONE)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.visibility == Visibility.NONE

    async def test_a_muted_channel_stays_muted(self) -> None:
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", muted=True)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.muted is True

    async def test_a_full_participant_is_still_shared_as_one(self) -> None:
        """Narrowing is not the goal — faithfulness is."""
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", access=Access.READ_WRITE)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.access == Access.READ_WRITE
        assert child.muted is False


class TestDetachIsNotUndoneByAutoAttach:
    """Detaching is how access is revoked; a message must not restore it."""

    async def _kit(self) -> RoomKit:
        kit = RoomKit()
        kit.register_channel(_Transport("ws"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws")
        return kit

    async def test_a_detached_channel_is_not_re_attached_by_a_message(self) -> None:
        kit = await self._kit()
        await kit.detach_channel("r1", "ws")

        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="mallory", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert await kit._store.get_binding("r1", "ws") is None

    async def test_a_metadata_write_cannot_clear_the_revocation(self) -> None:
        """RFC §5.1, §7.5 rule 7 (RMK-978): the revocation is the framework's
        key. An integrator's metadata write that names it has it dropped, a
        read-modify-write of the room's metadata included, so the next
        message does not attach the channel again: only an explicit attach
        grants access back."""
        kit = await self._kit()
        await kit.detach_channel("r1", "ws")
        room = await kit.get_room("r1")
        await kit.update_room_metadata("r1", {**room.metadata, "stage": "won"})
        await kit.update_room_metadata("r1", {"roomkit:detached_channels": []})

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="mallory", content=TextContent(body="hi")),
            room_id="r1",
        )

        room = await kit.get_room("r1")
        bound = await kit._store.get_binding("r1", "ws")
        assert (room.metadata, bound, result.reason) == (
            {"roomkit:detached_channels": ["ws"], "stage": "won"},
            None,
            "source_unbound",
        )
        await kit.close()

    async def test_an_explicit_attach_re_grants_access(self) -> None:
        """Revocation is not permanent — it is the integrator's to undo."""
        kit = await self._kit()
        await kit.detach_channel("r1", "ws")
        await kit.attach_channel("r1", "ws", access=Access.READ_ONLY)

        binding = await kit._store.get_binding("r1", "ws")
        assert binding is not None
        assert binding.access == Access.READ_ONLY

        # And the auto-attach path leaves that binding alone afterwards.
        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )
        binding = await kit._store.get_binding("r1", "ws")
        assert binding is not None
        assert binding.access == Access.READ_ONLY

    async def test_a_never_attached_channel_is_still_auto_attached(self) -> None:
        """The convenience this guard narrows must keep working."""
        kit = RoomKit()
        kit.register_channel(_Transport("ws"))
        await kit.create_room(room_id="r1")

        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert await kit._store.get_binding("r1", "ws") is not None

    async def test_a_detach_racing_an_inbound_message_is_final(self) -> None:
        """Both serial orders end detached; there is no third outcome.

        Auto-attach first → the detach removes the binding after it. Detach
        first → the tombstone blocks the auto-attach. The bug this guards
        against is the non-serial interleaving: the message re-attaching a
        channel whose detach landed mid-flight, because the revocation was
        read from a snapshot taken before the detach committed.
        """
        for round_no in range(20):
            kit = RoomKit()
            kit.register_channel(_Transport("ws"))
            await kit.create_room(room_id="r1")
            await kit.attach_channel("r1", "ws")

            await asyncio.gather(
                kit.detach_channel("r1", "ws"),
                kit.process_inbound(
                    InboundMessage(
                        channel_id="ws", sender_id="mallory", content=TextContent(body="hi")
                    ),
                    room_id="r1",
                ),
                return_exceptions=True,
            )
            binding = await kit._store.get_binding("r1", "ws")
            assert binding is None, f"round {round_no}: the message undid the detach"


class TestDetachSurvivesTheProcess:
    """The revocation lives in the store, not in framework memory (RFC §7.5-7).

    A restart — or another worker sharing the store — must see an explicit
    detach exactly as the process that performed it does. These tests share
    one store between two RoomKit instances: the second instance stands in
    for a restarted process or a sibling worker.
    """

    async def test_detach_survives_a_restart(self) -> None:
        store = InMemoryStore()
        kit1 = RoomKit(store=store)
        kit1.register_channel(_Transport("ws"))
        await kit1.create_room(room_id="r1")
        await kit1.attach_channel("r1", "ws")
        await kit1.detach_channel("r1", "ws")

        # "Restart": a fresh framework over the same store.
        kit2 = RoomKit(store=store)
        kit2.register_channel(_Transport("ws"))
        await kit2.process_inbound(
            InboundMessage(channel_id="ws", sender_id="mallory", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert await store.get_binding("r1", "ws") is None

    async def test_detach_on_one_worker_is_seen_by_another(self) -> None:
        store = InMemoryStore()
        kit1 = RoomKit(store=store)
        kit2 = RoomKit(store=store)
        for kit in (kit1, kit2):
            kit.register_channel(_Transport("ws"))
        await kit1.create_room(room_id="r1")
        await kit1.attach_channel("r1", "ws")

        await kit1.detach_channel("r1", "ws")

        # The other worker's inbound path must not re-grant the access.
        await kit2.process_inbound(
            InboundMessage(channel_id="ws", sender_id="mallory", content=TextContent(body="hi")),
            room_id="r1",
        )
        assert await store.get_binding("r1", "ws") is None

    async def test_explicit_reattach_after_a_restart_re_grants(self) -> None:
        store = InMemoryStore()
        kit1 = RoomKit(store=store)
        kit1.register_channel(_Transport("ws"))
        await kit1.create_room(room_id="r1")
        await kit1.attach_channel("r1", "ws")
        await kit1.detach_channel("r1", "ws")

        kit2 = RoomKit(store=store)
        kit2.register_channel(_Transport("ws"))
        await kit2.attach_channel("r1", "ws")

        assert await store.get_binding("r1", "ws") is not None
        # And auto-attach works again for it after another detach-less inbound.
        await kit2.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )
        assert await store.get_binding("r1", "ws") is not None

    async def test_the_tombstone_leaves_no_trace_once_cleared(self) -> None:
        """Metadata carries the mechanism only while a revocation stands."""
        from roomkit.core.mixins.channel_ops import DETACHED_CHANNELS_KEY

        store = InMemoryStore()
        kit = RoomKit(store=store)
        kit.register_channel(_Transport("ws"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws")

        await kit.detach_channel("r1", "ws")
        room = await kit.get_room("r1")
        assert room.metadata.get(DETACHED_CHANNELS_KEY) == ["ws"]

        await kit.attach_channel("r1", "ws")
        room = await kit.get_room("r1")
        assert DETACHED_CHANNELS_KEY not in room.metadata

    async def test_detaching_a_never_attached_channel_leaves_no_tombstone(self) -> None:
        """A no-op detach must not block future auto-attach."""
        from roomkit.core.mixins.channel_ops import DETACHED_CHANNELS_KEY

        kit = RoomKit()
        kit.register_channel(_Transport("ws"))
        await kit.create_room(room_id="r1")

        assert await kit.detach_channel("r1", "ws") is False
        room = await kit.get_room("r1")
        assert DETACHED_CHANNELS_KEY not in room.metadata

        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )
        assert await kit._store.get_binding("r1", "ws") is not None


class TestTheAttachOfStepTwo:
    """RFC §10.1 step 2: a room in hand that does not bind the message's
    channel has it attached under the room lock, its due timers applied
    first, at the declaration's defaults; none is made to a room whose
    status refuses events. A room created for the message has the channel
    attached unless a binding of it exists by then, a hook's included."""

    @pytest.mark.parametrize("refusing", ["closed", "archived", "due-to-close"])
    async def test_a_room_that_refuses_events_gets_no_attach(self, refusing: str) -> None:
        kit = RoomKit()
        kit.register_channel(_Transport("ws"))
        room = await kit.create_room(room_id="r1")
        if refusing == "due-to-close":
            gone = datetime.now(UTC) - timedelta(minutes=5)
            room = room.model_copy(
                update={
                    "status": RoomStatus.PAUSED,
                    "paused_at": gone,
                    "timers": RoomTimers(closed_after_seconds=1),
                }
            )
        else:
            room = room.model_copy(update={"status": RoomStatus(refusing)})
        await kit._store.update_room(room)

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert (result.blocked, result.reason) == (True, "room_closed")
        assert await kit._store.get_binding("r1", "ws") is None

    async def test_a_room_its_creation_hook_closed_gets_no_attach(self) -> None:
        """The room created for the message, closed by an ``ON_ROOM_CREATED``
        hook, gets no binding, as a room in hand whose status refuses events:
        step 6 refuses the message ``room_closed``, not routed again."""
        kit = RoomKit()
        kit.register_channel(_Transport("ws"))
        created: list[str] = []

        @kit.hook(HookTrigger.ON_ROOM_CREATED)
        async def close_at_once(event: object, ctx: RoomContext) -> None:
            created.append(ctx.room.id)
            await kit.close_room(ctx.room.id)

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi"))
        )

        assert (result.blocked, result.reason) == (True, "room_closed")
        assert len(created) == 1
        assert await kit._store.get_binding(created[0], "ws") is None
        await kit.close()

    async def test_a_binding_a_creation_hook_made_is_kept(self) -> None:
        kit = RoomKit()
        kit.register_channel(_Transport("ws"))

        @kit.hook(HookTrigger.ON_ROOM_CREATED)
        async def bind_read_only(event: object, ctx: RoomContext) -> None:
            await kit.attach_channel(ctx.room.id, "ws", access=Access.READ_ONLY)

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi"))
        )

        assert result.event is not None
        binding = await kit._store.get_binding(result.event.room_id, "ws")
        assert binding is not None and binding.access == Access.READ_ONLY
        assert (result.blocked, result.reason) == (True, "source_read_only")
