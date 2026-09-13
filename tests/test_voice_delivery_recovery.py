"""Crashes, cancellations and queue redelivery preserve voice reservations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceInjectionResult
from roomkit.core.locks import RoomLockManager
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.store.sqlite import SQLiteStore
from tests.test_proactive_delivery_voice import voice_room


def destination(session):
    return dict(channel_id="voice", session_id=session.id, idempotency_key="event:42")


@pytest.mark.parametrize("committed", [False, True])
async def test_result_persistence_failure_never_repeats_submission(committed) -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        complete = kit.store.complete_voice_delivery

        async def lose_response(record):
            if committed:
                await complete(record)
            raise ConnectionError("receipt unavailable")

        with patch.object(kit.store, "complete_voice_delivery", side_effect=lose_response):
            first = await kit.deliver("r", "done", **destination(sessions[0]))
        replay = await kit.deliver("r", "done", **destination(sessions[0]))
        assert first.status == "unknown" and not first.error.retryable
        assert replay.status == ("sent" if committed else "unknown")
        assert replay.duplicate and len(provider.injected_texts) == 1


async def test_storage_read_failure_is_safe_to_retry_before_any_submission() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        with patch.object(kit.store, "get_voice_delivery", side_effect=ConnectionError("offline")):
            first = await kit.deliver("r", "done", **destination(sessions[0]))
        assert first.status == "failed" and first.error.retryable
        assert provider.injected_texts == []
        assert (await kit.deliver("r", "done", **destination(sessions[0]))).status == "sent"


async def test_cancel_during_submission_preserves_unknown() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        entered = asyncio.Event()
        send = provider.inject_text

        async def pending(*args, **kwargs):
            await send(*args, **kwargs)
            entered.set()
            await asyncio.Future()

        with patch.object(provider, "inject_text", side_effect=pending):
            task = asyncio.create_task(kit.deliver("r", "done", **destination(sessions[0])))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        result = await kit.deliver("r", "done", **destination(sessions[0]))
        assert result.status == "unknown" and result.duplicate
        assert result.reason == "voice_submission_cancelled"
        assert len(provider.injected_texts) == 1


async def test_cancelled_waiter_does_not_cancel_owner() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        entered, release = asyncio.Event(), asyncio.Event()
        send = provider.inject_text

        async def slow(*args, **kwargs):
            entered.set()
            await release.wait()
            return await send(*args, **kwargs)

        with patch.object(provider, "inject_text", side_effect=slow):
            owner = asyncio.create_task(kit.deliver("r", "done", **destination(sessions[0])))
            await asyncio.wait_for(entered.wait(), 1)
            waiter = asyncio.create_task(kit.deliver("r", "done", **destination(sessions[0])))
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            release.set()
            assert (await asyncio.wait_for(owner, 1)).status == "sent"
        assert len(provider.injected_texts) == 1


class NoLocks(RoomLockManager):
    @asynccontextmanager
    async def locked(self, room_id):
        yield


async def test_store_claim_prevents_duplicates_even_without_shared_application_locks() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        kit._lock_manager = NoLocks()
        send = provider.inject_text

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.01)
            return await send(*args, **kwargs)

        with patch.object(provider, "inject_text", side_effect=slow):
            results = await asyncio.gather(
                *[kit.deliver("r", "done", **destination(sessions[0])) for _ in range(12)]
            )
        assert len(provider.injected_texts) == 1
        assert sum(result.status == "sent" for result in results) == 1
        assert all(result.status == "unknown" and result.duplicate for result in results[1:])


@pytest.mark.parametrize("state", ["sent", "unknown", "unresolved"])
async def test_sqlite_restart_replays_result_without_an_active_session(tmp_path, state) -> None:
    database = tmp_path / "rooms.db"
    async with voice_room(1, store=SQLiteStore(database)) as (kit, _, provider, sessions):
        args = destination(sessions[0])
        if state == "unknown":
            with patch.object(provider, "inject_text", side_effect=ConnectionError("uncertain")):
                first = await kit.deliver("r", "done", **args)
        elif state == "unresolved":
            with patch.object(
                kit.store, "complete_voice_delivery", side_effect=ConnectionError("lost")
            ):
                first = await kit.deliver("r", "done", **args)
        else:
            first = await kit.deliver("r", "done", **args)
    async with RoomKit(store=SQLiteStore(database)) as restarted:
        replay = await restarted.deliver("r", "done", **args)
        assert replay.status == first.status
        assert replay.duplicate
        assert replay.session_outcomes[args["session_id"]].status == first.status


@pytest.mark.parametrize("safe", [False, True])
async def test_worker_retries_only_explicit_non_submission_and_hooks_report_final_state(
    safe,
) -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        backend = InMemoryDeliveryBackend()
        observations = []

        @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
        async def observe(event, context):
            observations.append(event)

        send = provider.inject_text
        attempts = 0

        async def attempt(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                if safe:
                    return VoiceInjectionResult(
                        status="not_sent", reason="disconnected", retryable=True
                    )
                await send(*args, **kwargs)
                raise ConnectionError("confirmation lost")
            return await send(*args, **kwargs)

        item = DeliveryItem(room_id="r", content="done", **destination(sessions[0]))
        await backend.enqueue(item)
        try:
            with patch.object(provider, "inject_text", side_effect=attempt):
                await backend.start(kit)
                async with asyncio.timeout(2):
                    while item.status not in ("delivered", "dead_letter"):
                        await asyncio.sleep(0.01)
            assert item.outcome.status == ("sent" if safe else "unknown")
            assert attempts == (2 if safe else 1)
            assert len(provider.injected_texts) == 1
            assert observations[-1].metadata["delivery_outcome"]["status"] == item.outcome.status
            assert item.retry_count == (1 if safe else 0)
        finally:
            await backend.close()


async def test_unreported_result_does_not_fire_a_successful_injection_hook() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        observed = []

        @kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
        async def observe(event, context):
            observed.append(event)

        with patch.object(provider, "inject_text", new=AsyncMock(return_value=None)):
            result = await kit.deliver("r", "done", **destination(sessions[0]))
        assert result.status == "unknown"
        assert observed == []
