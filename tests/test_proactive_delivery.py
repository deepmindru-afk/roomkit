"""External events preserve addressing, publication identity and real outcomes."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from roomkit import Access, DeliveryOutcome, HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.worker import execute_delivery
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.store.sqlite import SQLiteStore
from tests.test_addressing import _room
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
