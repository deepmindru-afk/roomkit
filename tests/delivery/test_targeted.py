"""Queue acceptance, execution and retries preserve proactive delivery identity."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, Queued, RoomKit
from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.worker import execute_delivery
from roomkit.models.enums import ChannelType
from roomkit.models.event import TextContent
from tests.test_addressing import _room
from tests.test_framework import SimpleChannel


async def test_enqueue_replay_preserves_address_and_key() -> None:
    kit, _, agents = await _room("a", "b")
    backend = InMemoryDeliveryBackend()
    kit._delivery_backend = backend
    async with kit:
        # The backend is not started: acceptance alone must not solicit an agent.
        queued = await kit.deliver(
            "room-1",
            "external result",
            addressed_to=["a"],
            idempotency_key="external:1",
        )
        assert queued.status == "queued" and queued.event_id is None
        assert agents["a"].solicited == []
        [item] = await backend.dequeue("worker", timeout=0.1)
        restored = DeliveryItem.model_validate_json(item.model_dump_json())
        assert restored.id == queued.delivery_item_id
        assert restored.addressed_to == ["a"]
        assert restored.idempotency_key == "external:1"
        first = await execute_delivery(kit, restored)
        await backend.nack(item.id, error="ack connection lost")
        [retry] = await backend.dequeue("worker", timeout=0.1)
        second = await execute_delivery(kit, retry)
        await backend.ack(retry.id)
        assert first.status == second.status == "sent"
        assert first.event_id == second.event_id
        assert second.duplicate
        assert agents["a"].solicited == ["external result"]
        assert agents["b"].solicited == []


async def test_worker_retries_unavailable_destination_then_delivers() -> None:
    backend = InMemoryDeliveryBackend()
    async with RoomKit() as kit:
        await kit.create_room(room_id="r")
        item = DeliveryItem(room_id="r", content="result", channel_id="text", idempotency_key="k")
        await backend.enqueue(item)
        [first] = await backend.dequeue("worker", timeout=0.1)
        unavailable = await execute_delivery(kit, first)
        assert unavailable.status == "unavailable"
        await backend.nack(first.id, error=unavailable.reason)
        kit.register_channel(SimpleChannel("text"))
        await kit.attach_channel("r", "text")
        [retry] = await backend.dequeue("worker", timeout=0.1)
        assert retry.idempotency_key == "k" and retry.retry_count == 1
        assert retry.outcome.status == "unavailable"
        sent = await execute_delivery(kit, retry)
        await backend.ack(retry.id)
        assert sent.status == "sent" and not sent.duplicate
        assert retry.status == DeliveryItemStatus.DELIVERED


async def test_real_worker_dead_letters_missing_target_and_records_refusal() -> None:
    backend = InMemoryDeliveryBackend()
    async with RoomKit(delivery_backend=backend) as kit:
        await kit.create_room(room_id="r")
        item = DeliveryItem(room_id="r", content="result", max_retries=2)
        await backend.enqueue(item)
        await backend.start(kit)
        async with asyncio.timeout(2):
            while not await backend.get_dead_letter_items():
                await asyncio.sleep(0.01)
        [dead] = await backend.get_dead_letter_items()
        assert dead.outcome.status == "unavailable"
        assert dead.retry_count == 2
        await backend.close()

        @kit.hook(HookTrigger.BEFORE_DELIVER)
        async def refuse(event, context):
            return HookResult.block("policy")

        blocked = DeliveryItem(room_id="r", content="refused")
        await backend.enqueue(blocked)
        await backend.start(kit)
        async with asyncio.timeout(2):
            while blocked.status != DeliveryItemStatus.BLOCKED:
                await asyncio.sleep(0.01)
        assert blocked.outcome.status == "blocked"
        assert blocked.outcome.reason == "policy"
        assert blocked.retry_count == 0


async def test_worker_after_hook_sees_rewrite_and_real_outcome() -> None:
    kit, _, agents = await _room("a")
    observed = []

    @kit.hook(HookTrigger.BEFORE_DELIVER)
    async def rewrite(event, context):
        return HookResult.modify(
            event.model_copy(update={"content": TextContent(body="redacted")})
        )

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def record(event, context):
        observed.append(event)

    async with kit:
        result = await execute_delivery(
            kit,
            DeliveryItem(
                room_id="room-1",
                content="secret",
                addressed_to=["a"],
                idempotency_key="k",
            ),
        )
        assert result.status == "sent"
        assert agents["a"].solicited == ["redacted"]
        assert observed[0].content.body == "redacted"
        assert observed[0].metadata["delivery_outcome"]["event_id"] == result.event_id
        assert observed[0].addressed_to == ["a"]


@pytest.mark.parametrize("keyed", [False, True])
async def test_queued_keeps_incompatible_targets_and_keyed_publications_separate(keyed) -> None:
    kit, human, agents = await _room("a", "b")
    human.channel_type = ChannelType.VOICE
    strategy = Queued(buffer=0)
    entered, release = asyncio.Event(), asyncio.Event()

    async def wait(*args):
        entered.set()
        await release.wait()

    async with kit:
        with patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=wait):
            first = asyncio.create_task(
                kit.deliver(
                    "room-1",
                    "first",
                    addressed_to=["a"],
                    strategy=strategy,
                    idempotency_key="k" if keyed else None,
                )
            )
            await entered.wait()
            others = [
                asyncio.create_task(
                    kit.deliver(
                        "room-1",
                        body,
                        addressed_to=[target],
                        strategy=strategy,
                        idempotency_key=key if keyed else None,
                    )
                )
                for body, target, key in [("first", "a", "k"), ("second", "b", "other")]
            ]
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, *others), 2)
        for result in results:
            assert result.status == "sent"
            await result.inbound.delivery.wait()
        assert agents["b"].solicited == ["second"]
        if keyed:
            assert agents["a"].solicited == ["first"]
            assert results[1].duplicate
        else:
            assert agents["a"].solicited == ["first\n\nfirst"]
        assert results[0].event_id == results[1].event_id
        assert results[2].event_id != results[0].event_id


async def test_queued_accepts_reentrant_delivery_from_agent_without_deadlocking() -> None:
    kit, _, agents = await _room("a")
    strategy = Queued(buffer=0)
    original = agents["a"].on_event

    async def followup(event, binding, context):
        if isinstance(event.content, TextContent) and event.content.body == "first":
            await kit.deliver("room-1", "second", addressed_to=[], strategy=strategy)
        return await original(event, binding, context)

    agents["a"].on_event = followup
    async with kit:
        result = await asyncio.wait_for(
            kit.deliver(
                "room-1",
                "first",
                addressed_to=["a"],
                strategy=strategy,
            ),
            2,
        )
        await asyncio.wait_for(result.inbound.delivery.wait(), 2)
        bodies = [
            e.content.body
            for e in await kit.get_timeline("room-1")
            if isinstance(e.content, TextContent)
        ]
        assert bodies == ["first", "second"]
