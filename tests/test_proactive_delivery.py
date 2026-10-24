"""External events preserve addressing, publication identity and real outcomes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from roomkit import Access, DeliveryOutcome, HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.worker import execute_delivery
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from tests.test_addressing import _RecordingAgent, _room
from tests.test_framework import SimpleChannel


@pytest.mark.parametrize(
    ("address", "expected", "status"),
    [
        (["a"], ["a"], "sent"),
        (None, ["b"], "sent"),
        ([], [], "sent"),
        (["absent"], [], "unavailable"),
        (["a", "absent"], ["a"], "unavailable"),
    ],
)
async def test_address_outranks_default_without_hiding_event(address, expected, status) -> None:
    kit, _, agents = await _room("a", "b")
    observer = SimpleChannel("observer")
    kit.register_channel(observer)
    await kit.attach_channel("room-1", observer.channel_id)

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def route(event: RoomEvent, context: RoomContext) -> HookResult:
        return HookResult.modify(event.model_copy(update={"metadata": {"_routed_to": "b"}}))

    async with kit:
        result = await kit.deliver("room-1", "external result", addressed_to=address)
        assert result.status == status
        assert [name for name, agent in agents.items() if agent.solicited] == expected
        assert observer.delivered[0].addressed_to == address
        assert observer.delivered[0].visibility == "all"
        assert result.event_id == observer.delivered[0].id
        assert result.inbound is not None
        assert result.turn_complete
        if status == "unavailable":
            assert result.unavailable_targets == ["absent"]
            assert result.error is not None and not result.error.retryable


async def test_concurrent_duplicates_have_one_trigger_and_one_solicitation() -> None:
    kit, _, agents = await _room("a", "b")
    async with kit:
        results = await asyncio.gather(
            *[
                kit.deliver("room-1", "done", addressed_to=["a"], idempotency_key="external:1")
                for _ in range(12)
            ]
        )
        assert {r.status for r in results} == {"sent"}
        assert len({r.event_id for r in results}) == 1
        assert sum(r.duplicate for r in results) == 11
        assert all(not r.turn_complete for r in results if r.duplicate)
        assert agents["a"].solicited == ["done"]
        assert agents["b"].solicited == []
        assert len([e for e in await kit.get_timeline("room-1") if e.idempotency_key]) == 1
        replay = await kit.deliver(
            "room-1",
            "changed",
            addressed_to=["b"],
            idempotency_key="external:1",
        )
        assert replay.event_id == results[0].event_id
        assert replay.inbound.event.addressed_to == ["a"]
        assert agents["b"].solicited == []
        distinct = await kit.deliver(
            "room-1", "next", addressed_to=["b"], idempotency_key="external:2"
        )
        assert distinct.event_id != replay.event_id
        assert agents["b"].solicited == ["next"]


async def test_keys_survive_sqlite_restart_and_are_scoped_to_room(tmp_path: Path) -> None:
    database = tmp_path / "rooms.db"
    async with RoomKit(store=SQLiteStore(database)) as kit:
        kit.register_channel(SimpleChannel("text"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "text")
        first = await kit.deliver("r1", "done", idempotency_key="key")
    async with RoomKit(store=SQLiteStore(database)) as kit:
        kit.register_channel(SimpleChannel("text"))
        replay = await kit.deliver("r1", "done", idempotency_key="key")
        assert replay.duplicate and replay.event_id == first.event_id
        await kit.create_room(room_id="r2")
        await kit.attach_channel("r2", "text")
        other = await kit.deliver("r2", "done", idempotency_key="key")
        assert not other.duplicate and other.event_id != first.event_id


class _YieldingStore(InMemoryStore):
    """An in-memory store whose key lookup yields once it has read, as a
    networked one does, so that two deliveries keyed alike interleave on
    answers that may be stale by the time they come back."""

    async def get_event_by_idempotency_key(
        self, room_id: str, key: str, *, channel_id: str | None
    ) -> RoomEvent | None:
        found = await super().get_event_by_idempotency_key(room_id, key, channel_id=channel_id)
        await asyncio.sleep(0)
        return found


class TestAKeyNamesOnePublicationInItsRoom:
    """RFC §22.1, §13.4, §14.3: a keyed text delivery's key names one
    publication in its room, whatever transport carries it: looked up within
    the room before a transport is resolved, the lookup, the resolution and
    the commit in one section under the room lock."""

    async def _kit(self) -> tuple[RoomKit, dict[str, _RecordingAgent]]:
        kit = RoomKit(store=_YieldingStore())
        agents = {"a": _RecordingAgent("a")}
        kit.register_channel(agents["a"])
        for transport in ("sms", "web"):
            kit.register_channel(SimpleChannel(transport))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "a", category=ChannelCategory.INTELLIGENCE)
        await kit.attach_channel("r1", "sms")
        return kit, agents

    async def _keyed(self, kit: RoomKit) -> list[RoomEvent]:
        """The room's publications keyed ``k``, stored in the framework's
        namespace."""
        timeline = await kit.store.list_events("r1", limit=50)
        return [e for e in timeline if e.idempotency_key == "roomkit:publication:k"]

    async def test_a_retry_that_resolves_another_transport_is_the_same_publication(
        self,
    ) -> None:
        kit, agents = await self._kit()
        async with kit:
            first = await kit.deliver("r1", "done", idempotency_key="k")
            await kit.detach_channel("r1", "sms")
            await kit.attach_channel("r1", "web")

            replay = await kit.deliver("r1", "done", idempotency_key="k")

            assert (first.status, replay.status) == ("sent", "sent")
            assert replay.duplicate and replay.event_id == first.event_id
            assert len(await self._keyed(kit)) == 1
            assert agents["a"].solicited == ["done"]

    async def test_a_retry_finds_its_publication_with_no_transport_left(self) -> None:
        kit, _agents = await self._kit()
        async with kit:
            first = await kit.deliver("r1", "done", idempotency_key="k")
            await kit.detach_channel("r1", "sms")

            replay = await kit.deliver("r1", "done", idempotency_key="k")

            assert replay.status == "sent", replay.reason
            assert replay.duplicate and replay.event_id == first.event_id

    @pytest.mark.parametrize("key", ["k", "roomkit:publication:k"])
    async def test_a_message_keyed_like_a_publication_does_not_stand_in_for_it(
        self, key: str
    ) -> None:
        """A sender's key, on a channel whose client chooses it, is the
        sender's: a message keyed like a coming publication neither blocks it
        nor is taken for it, and a key in the framework's namespace is
        dropped from the message."""
        kit, agents = await self._kit()
        await kit.attach_channel("r1", "web")
        async with kit:
            sent = await kit.process_inbound(
                InboundMessage(
                    channel_id="web",
                    sender_id="mallory",
                    content=TextContent(body="spoof"),
                    idempotency_key=key,
                ),
                room_id="r1",
            )
            published = await kit.deliver("r1", "done", idempotency_key="k")

            assert sent.event is not None
            assert sent.event.idempotency_key == (None if key != "k" else "k")
            assert published.status == "sent" and not published.duplicate
            assert published.event_id != sent.event.id
            assert [e.id for e in await self._keyed(kit)] == [published.event_id]
            assert agents["a"].solicited == ["spoof", "done"]

    async def test_a_room_lock_held_past_process_timeout_refuses_the_publication(
        self,
    ) -> None:
        """The section's wait for the room lock is bounded as the pipeline's
        own is (RFC §13.6): nothing is published."""
        kit, agents = await self._kit()
        kit._process_timeout = 0.05  # noqa: SLF001
        held, release = asyncio.Event(), asyncio.Event()

        async def hold() -> None:
            async with kit.lock_manager.locked("r1"):
                held.set()
                await release.wait()

        async with kit:
            holder = asyncio.create_task(hold())
            await held.wait()
            try:
                outcome = await kit.deliver("r1", "done", idempotency_key="k")
            finally:
                release.set()
                await holder

            assert (outcome.status, outcome.reason) == ("blocked", "process_timeout")
            assert await self._keyed(kit) == []
            assert agents["a"].solicited == []

    async def test_commit_event_refuses_the_publication_namespace(self) -> None:
        """A record committed outside the pipeline cannot take a publication's
        key: it would stand in for the publication."""
        kit, _agents = await self._kit()
        record = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
            content=TextContent(body="a copied message"),
            idempotency_key="roomkit:publication:k",
        )
        async with kit:
            with pytest.raises(ValueError, match="roomkit:publication:"):
                await kit.commit_event("r1", record)
            published = await kit.deliver("r1", "done", idempotency_key="k")

            assert published.status == "sent" and not published.duplicate
            assert [e.id for e in await self._keyed(kit)] == [published.event_id]

    async def test_send_event_refuses_the_publication_namespace(self) -> None:
        kit, _agents = await self._kit()
        async with kit:
            with pytest.raises(ValueError, match="roomkit:publication:"):
                await kit.send_event(
                    "r1", "sms", TextContent(body="x"), idempotency_key="roomkit:publication:k"
                )

    async def test_concurrent_replays_on_two_transports_publish_once(self) -> None:
        kit, agents = await self._kit()
        await kit.attach_channel("r1", "web")
        async with kit:
            results = await asyncio.gather(
                kit.deliver("r1", "done", channel_id="sms", idempotency_key="k"),
                kit.deliver("r1", "done", channel_id="web", idempotency_key="k"),
            )

            assert {r.status for r in results} == {"sent"}
            assert len({r.event_id for r in results}) == 1
            assert [r.duplicate for r in results].count(True) == 1
            assert len(await self._keyed(kit)) == 1
            assert agents["a"].solicited == ["done"]


@pytest.mark.parametrize("trigger", [HookTrigger.BEFORE_DELIVER, HookTrigger.BEFORE_BROADCAST])
async def test_refusal_is_reported_to_caller_and_after_hook(trigger) -> None:
    kit, _, agents = await _room("a")
    observations: list[RoomEvent] = []
    observed = asyncio.Event()

    @kit.hook(trigger)
    async def refuse(event: RoomEvent, context: RoomContext) -> HookResult:
        return HookResult.block("external_events_disabled")

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def observe(event: RoomEvent, context: RoomContext) -> None:
        observations.append(event)
        observed.set()

    async with kit:
        result = await kit.deliver("room-1", "done", addressed_to=["a"], idempotency_key="k")
        await asyncio.wait_for(observed.wait(), 1)
        assert result.status == "blocked"
        assert result.reason == "external_events_disabled"
        assert agents["a"].solicited == []
        assert observations[0].status == "blocked"
        assert observations[0].addressed_to == ["a"]
        assert observations[0].idempotency_key == "k"
        assert observations[0].metadata["delivery_outcome"]["status"] == "blocked"


@pytest.mark.parametrize("access", [Access.NONE, Access.WRITE_ONLY])
async def test_unreadable_agent_is_unavailable_without_fallback(access) -> None:
    kit, _, agents = await _room("a", "b")
    async with kit:
        await kit.set_access("room-1", "a", access)
        result = await kit.deliver("room-1", "done", addressed_to=["a"])
        assert result.status == "unavailable"
        assert result.unavailable_targets == ["a"]
        assert not any(agent.solicited for agent in agents.values())


async def test_muting_keeps_agent_side_effects() -> None:
    kit, _, agents = await _room("a", "b")
    async with kit:
        await kit.mute("room-1", "a")
        result = await kit.deliver("room-1", "done", addressed_to=["a"])
        assert result.status == "sent"
        assert agents["a"].solicited == ["done"]
        assert agents["b"].solicited == []


async def test_absent_transport_and_unbound_channel_are_not_success() -> None:
    async with RoomKit() as kit:
        await kit.create_room(room_id="r")
        kit.register_channel(SimpleChannel("not-bound"))
        assert (await kit.deliver("r", "done")).reason == "no_transport"
        result = await kit.deliver("r", "done", channel_id="not-bound")
        assert result.status == "unavailable"
        assert result.unavailable_targets == ["not-bound"]
        assert await kit.get_timeline("r") == []


async def test_result_is_public_and_serializable_without_runtime_handle() -> None:
    kit, _, _ = await _room("a")
    async with kit:
        result = await kit.deliver("room-1", "done", addressed_to=["a"])
        assert isinstance(result, DeliveryOutcome)
        snapshot = DeliveryOutcome.model_validate_json(result.model_dump_json())
        assert snapshot.event_id == result.event_id
        assert snapshot.inbound is None


async def test_agent_detaching_after_solicitation_does_not_change_outcome() -> None:
    kit, _, agents = await _room("a")
    original = agents["a"].on_event

    async def detach_after_acting(event, binding, context):
        result = await original(event, binding, context)
        await kit.detach_channel("room-1", "a")
        return result

    agents["a"].on_event = detach_after_acting
    async with kit:
        result = await kit.deliver("room-1", "external result", addressed_to=["a"])
        assert result.status == "sent"
        assert agents["a"].solicited == ["external result"]
        assert result.unavailable_targets == []
        assert result.inbound.unavailable_targets == []


async def test_source_detaching_before_commit_refuses_the_delivery() -> None:
    """A source detached before the commit writes nothing into the room (RFC
    §7.5 rules 2 and 7): the delivery is refused, its record stored."""
    kit, _, agents = await _room("a")

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def detach_source(event, context):
        await kit.store.remove_binding("room-1", event.source.channel_id)
        return HookResult.allow()

    async with kit:
        result = await kit.deliver("room-1", "external result", addressed_to=["a"])
        assert (result.status, result.reason) == ("blocked", "source_unbound")
        assert result.event_id is not None
        assert agents["a"].solicited == []


async def test_source_detaching_before_the_lock_refuses_the_delivery() -> None:
    """A detach landing between the call and the commit's room lock is seen
    by the commit, which reads the bindings under that lock (RFC §10.1 step
    6): the delivery is refused as its source's."""
    kit, _, agents = await _room("a")
    locked = kit._lock_manager.locked
    detached: list[str] = []

    @asynccontextmanager
    async def detaching_first(room_id: str) -> AsyncIterator[None]:
        if not detached:
            detached.append(room_id)
            await kit.detach_channel(room_id, "human")
        async with locked(room_id):
            yield

    kit._lock_manager.locked = detaching_first  # type: ignore[method-assign]

    async with kit:
        result = await kit.deliver("room-1", "external result", addressed_to=["a"])
        assert (result.status, result.reason) == ("blocked", "source_unbound")
        assert result.event_id is not None
        assert agents["a"].solicited == []


async def test_no_fallible_context_lookup_after_text_publication() -> None:
    kit, _, agents = await _room("a")
    original = agents["a"].on_event
    build_context = kit._build_context

    async def fail_context(*args, **kwargs):
        raise ConnectionError("context service unavailable after publication")

    async def act_then_disconnect(event, binding, context):
        result = await original(event, binding, context)
        kit._build_context = fail_context
        return result

    agents["a"].on_event = act_then_disconnect
    async with kit:
        try:
            result = await kit.deliver("room-1", "external result", addressed_to=["a"])
        finally:
            kit._build_context = build_context
        assert result.status == "sent"
        assert result.event_id is not None
        assert result.error is None
        assert agents["a"].solicited == ["external result"]


async def test_replay_hook_describes_original_publication() -> None:
    kit, _, agents = await _room("a", "b")
    observations = asyncio.Queue()

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def observe(event, context):
        await observations.put(event)

    async with kit:
        first = await kit.deliver("room-1", "original", addressed_to=["a"], idempotency_key="k")
        await asyncio.wait_for(observations.get(), 1)
        replay = await kit.deliver("room-1", "changed", addressed_to=["b"], idempotency_key="k")
        observed = await asyncio.wait_for(observations.get(), 1)
        assert replay.duplicate and replay.event_id == first.event_id
        assert observed.content.body == "original"
        assert observed.addressed_to == ["a"]
        assert observed.metadata["delivery_outcome"]["event_id"] == first.event_id
        assert observed.metadata["error"] is None
        assert agents["b"].solicited == []


@pytest.mark.parametrize("path", ["direct", "enqueue", "worker"])
@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"idempotency_key": ""}, "empty_idempotency_key"),
        ({"idempotency_key": "   "}, "empty_idempotency_key"),
        ({"session_id": "s"}, "invalid_session_target"),
    ],
)
async def test_invalid_request_is_observable_without_publication(path, kwargs, reason) -> None:
    backend = InMemoryDeliveryBackend() if path == "enqueue" else None
    async with RoomKit(delivery_backend=backend) as kit:
        await kit.create_room(room_id="r")
        observations = asyncio.Queue()

        @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
        async def observe(event, context):
            await observations.put(event)

        if path == "worker":
            result = await execute_delivery(
                kit, DeliveryItem(room_id="r", content="bad", **kwargs)
            )
        else:
            result = await kit.deliver("r", "bad", **kwargs)
        observed = await asyncio.wait_for(observations.get(), 1)
        assert result.status == observed.status == "blocked"
        assert result.reason == reason
        assert observed.metadata["delivery_outcome"] == result.model_dump(mode="json")
        assert await kit.get_timeline("r") == []
        if backend is not None:
            assert await backend.dequeue("test", timeout=0) == []
