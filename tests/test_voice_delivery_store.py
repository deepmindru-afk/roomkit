"""Atomic reservation and completion contracts for all shipped stores."""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

import pytest

from roomkit import DeliveryError, DeliveryOutcome, VoiceDeliveryRecord
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore
from roomkit.store.postgres import PostgresStore
from roomkit.store.sqlite import SQLiteStore


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def stores(request, tmp_path):
    if request.param == "postgres":
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        first, second = PostgresStore(dsn=dsn), PostgresStore(dsn=dsn)
        await first.init()
        await second.init()
    elif request.param == "sqlite":
        first, second = SQLiteStore(tmp_path / "rooms.db"), SQLiteStore(tmp_path / "rooms.db")
    else:
        first = second = InMemoryStore()
    room = await first.create_room(Room(id=f"voice-dedup-{uuid4().hex}"))
    try:
        yield first, second, room.id
    finally:
        await first.delete_room(room.id)
        await first.close()
        if second is not first:
            await second.close()


def request_record(room_id: str, **kwargs) -> VoiceDeliveryRecord:
    return VoiceDeliveryRecord(
        room_id=room_id,
        channel_id="voice",
        session_id="session",
        idempotency_key="event:1",
        content_hash="same-content",
        **kwargs,
    )


async def test_concurrent_claims_have_one_owner(stores) -> None:
    first, second, room = stores
    attempts = [request_record(room) for _ in range(24)]
    records = await asyncio.gather(
        *[
            (first if index % 2 else second).claim_voice_delivery(attempt)
            for index, attempt in enumerate(attempts)
        ]
    )
    assert len({record.attempt_id for record in records}) == 1
    assert sum(a.attempt_id == records[0].attempt_id for a in attempts) == 1
    records[0].content_hash = "caller-mutated"
    stored = await second.get_voice_delivery(room, records[0].key_hash)
    assert stored.content_hash == "same-content"


async def test_completion_is_idempotent_and_fences_other_owners(stores) -> None:
    first, second, room = stores
    record = await first.claim_voice_delivery(request_record(room))
    sent = record.model_copy(
        update={"outcome": DeliveryOutcome(status="sent", session_ids=["session"])}
    )
    assert not await second.complete_voice_delivery(
        sent.model_copy(update={"attempt_id": "other"})
    )
    assert await first.complete_voice_delivery(sent)
    assert await second.complete_voice_delivery(sent)
    if isinstance(first, PostgresStore):
        async with first._acquire() as conn:
            kind = await conn.fetchval(
                "SELECT jsonb_typeof(outcome) FROM voice_deliveries "
                "WHERE room_id = $1 AND key_hash = $2",
                room,
                sent.key_hash,
            )
        assert kind == "object"
    assert not await first.complete_voice_delivery(
        sent.model_copy(update={"outcome": DeliveryOutcome(status="unknown")})
    )
    replay = await second.claim_voice_delivery(request_record(room))
    assert replay.attempt_id == record.attempt_id
    assert replay.outcome == sent.outcome


async def test_safe_retry_claims_new_owner_and_rejects_late_completion(stores) -> None:
    first, second, room = stores
    original = await first.claim_voice_delivery(request_record(room))
    refused = original.model_copy(
        update={
            "outcome": DeliveryOutcome(
                status="failed",
                reason="not_connected",
                error=DeliveryError(code="not_connected", message="No send", retryable=True),
            )
        }
    )
    assert await first.complete_voice_delivery(refused)
    different = request_record(room).model_copy(update={"content_hash": "changed"})
    assert (await second.claim_voice_delivery(different)).attempt_id == original.attempt_id
    retry = request_record(room)
    assert (await second.claim_voice_delivery(retry)).attempt_id == retry.attempt_id
    assert not await first.complete_voice_delivery(refused)
    assert (await first.get_voice_delivery(room, retry.key_hash)).outcome is None


@pytest.mark.parametrize("terminal", [False, True])
async def test_unresolved_or_unknown_attempt_never_becomes_retryable(stores, terminal) -> None:
    first, second, room = stores
    original = await first.claim_voice_delivery(request_record(room))
    if terminal:
        assert await first.complete_voice_delivery(
            original.model_copy(
                update={
                    "outcome": DeliveryOutcome(status="unknown", reason="confirmation_lost"),
                }
            )
        )
    replay = await second.claim_voice_delivery(request_record(room))
    assert replay.attempt_id == original.attempt_id
    assert not replay.retryable


async def test_receipts_are_scoped_and_deleted_with_their_room(stores) -> None:
    first, second, room = stores
    original = await first.claim_voice_delivery(request_record(room))
    other_room = await first.create_room(Room(id=f"voice-other-{uuid4().hex}"))
    try:
        assert await second.get_voice_delivery(other_room.id, original.key_hash) is None
        candidates = [
            request_record(other_room.id),
            request_record(room).model_copy(update={"session_id": "another"}),
            request_record(room).model_copy(update={"channel_id": "another"}),
            request_record(room).model_copy(update={"idempotency_key": "another"}),
        ]
        for item in candidates:
            assert (await second.claim_voice_delivery(item)).attempt_id == item.attempt_id
        await first.delete_room(room)
        assert await second.get_voice_delivery(room, original.key_hash) is None
        assert await second.get_voice_delivery(other_room.id, candidates[0].key_hash) is not None
    finally:
        await first.delete_room(other_room.id)


_PROCESS_CLAIM = """
import asyncio
import os
import sys
from roomkit import VoiceDeliveryRecord
from roomkit.store.postgres import PostgresStore
from roomkit.store.sqlite import SQLiteStore

async def main():
    store = (PostgresStore(dsn=os.environ['POSTGRES_DSN'])
             if sys.argv[1] == 'postgres' else SQLiteStore(sys.argv[2]))
    if sys.argv[1] == 'postgres':
        await store.init()
    request = VoiceDeliveryRecord(room_id=sys.argv[3], channel_id='voice',
        session_id='session', idempotency_key='event:process', content_hash='body')
    sys.stdout.write('ready\\n')
    sys.stdout.flush()
    await asyncio.to_thread(sys.stdin.readline)
    claimed = await store.claim_voice_delivery(request)
    sys.stdout.write(request.attempt_id + ' ' + claimed.attempt_id + '\\n')
    await store.close()

asyncio.run(main())
"""


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_separate_processes_share_one_reservation(backend, tmp_path) -> None:
    """The database boundary protects callers with no shared Python objects."""
    database = tmp_path / "processes.db"
    if backend == "postgres":
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        store = PostgresStore(dsn=dsn)
        await store.init()
    else:
        store = SQLiteStore(database)
    room = await store.create_room(Room(id=f"voice-process-{uuid4().hex}"))
    processes = []
    try:
        for _ in range(2):
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _PROCESS_CLAIM,
                backend,
                str(database),
                room.id,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            processes.append(process)
        for process in processes:
            assert await asyncio.wait_for(process.stdout.readline(), 10) == b"ready\n"
        for process in processes:
            process.stdin.write(b"claim\n")
            await process.stdin.drain()
        results = await asyncio.wait_for(
            asyncio.gather(*[process.communicate() for process in processes]), 10
        )
        claims = []
        for process, (output, error) in zip(processes, results, strict=True):
            assert process.returncode == 0, error.decode()
            claims.append(output.decode().strip().split())
        assert len({claimed for _, claimed in claims}) == 1
        assert sum(request == claimed for request, claimed in claims) == 1
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
        await store.delete_room(room.id)
        await store.close()
