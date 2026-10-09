"""Identity verification hooks and the end of a verification when it happens (RFC §11.7.5).

A verification ends at its expiry by a timer, not on the sender's next
message; the timers are re-armed when a kit starts, and what expired while
nothing ran is ended then.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import (
    HookExecution,
    HookTrigger,
    RoomKit,
    VerificationPolicy,
    VerificationRefusedError,
    VerificationStatus,
)
from roomkit.models.verification import Verification
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore


async def _eventually(predicate: Callable[[], Any], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def _kit(
    store: ConversationStore | None = None, **policy: float
) -> tuple[RoomKit, list[tuple[str, dict[str, Any]]]]:
    kit = RoomKit(store=store or InMemoryStore(), verification_policy=VerificationPolicy(**policy))
    if await kit.store.get_room("r1") is None:
        await kit.create_room(room_id="r1")
    fired: list[tuple[str, dict[str, Any]]] = []
    for trigger in (HookTrigger.ON_VERIFICATION_COMPLETED, HookTrigger.ON_VERIFICATION_ENDED):

        @kit.hook(trigger, execution=HookExecution.ASYNC)
        async def record(event: Any, ctx: Any, trigger: HookTrigger = trigger) -> None:
            fired.append((str(trigger), dict(event.content.data)))

    return kit, fired


async def _verified(kit: RoomKit, identity_id: str = "id-alice") -> Verification:
    request, secret = await kit.request_verification(identity_id, room_id="r1", channel_id="sms")
    return await kit.complete_verification(request.id, secret)


async def test_a_completion_fires_its_hook_in_the_request_room() -> None:
    kit, fired = await _kit()

    await _verified(kit)

    await _eventually(lambda: fired)
    trigger, data = fired[0]
    assert trigger == HookTrigger.ON_VERIFICATION_COMPLETED
    assert data["identity_id"] == "id-alice" and data["channel_id"] == "sms"
    assert data["method"] == "link"
    await kit.close()


async def test_a_verification_ends_when_it_expires_not_on_the_next_message() -> None:
    kit, fired = await _kit(ttl=0.1)

    await _verified(kit)
    await _eventually(lambda: any(t == HookTrigger.ON_VERIFICATION_ENDED for t, _ in fired))

    ended = [d for t, d in fired if t == HookTrigger.ON_VERIFICATION_ENDED]
    assert ended[0]["reason"] == "expired" and ended[0]["channel_id"] == "sms"
    stored = await kit.store.get_verification("id-alice")
    assert stored is not None and stored.status == VerificationStatus.EXPIRED
    await kit.close()


async def test_a_revocation_fires_the_end_once_and_stops_the_timer() -> None:
    kit, fired = await _kit(ttl=0.2)
    await _verified(kit)

    await kit.revoke_verification("id-alice", reason="advisor")
    await asyncio.sleep(0.3)

    ended = [d["reason"] for t, d in fired if t == HookTrigger.ON_VERIFICATION_ENDED]
    assert ended == ["advisor"]
    await kit.close()


async def test_a_request_that_fails_fires_the_end() -> None:
    kit, fired = await _kit(max_attempts=1)
    request, _ = await kit.request_verification("id-alice", room_id="r1", channel_id="sms")

    with pytest.raises(VerificationRefusedError):
        await kit.complete_verification(request.id, "wrong")

    await _eventually(lambda: fired)
    assert fired[0][0] == HookTrigger.ON_VERIFICATION_ENDED and fired[0][1]["reason"] == "failed"
    await kit.close()


async def test_a_starting_kit_rearms_the_timers_of_verifications_in_force() -> None:
    store = InMemoryStore()
    first, _ = await _kit(store, ttl=0.3)
    await _verified(first)
    await first.close()  # its timer dies with it

    second, fired = await _kit(store, ttl=0.3)
    async with second:
        await _eventually(lambda: fired, timeout=2.0)

    assert fired[0][0] == HookTrigger.ON_VERIFICATION_ENDED
    assert fired[0][1]["reason"] == "expired"


async def test_what_expired_while_nothing_ran_is_ended_at_start() -> None:
    store = InMemoryStore()
    first, _ = await _kit(store, ttl=0.05)
    await _verified(first)
    await first.close()
    await asyncio.sleep(0.1)

    second, fired = await _kit(store)
    ended = await second.check_verifications()

    assert [v.identity_id for v in ended] == ["id-alice"]
    assert ended[0].status == VerificationStatus.EXPIRED
    await _eventually(lambda: fired)
    await second.close()


async def test_a_closed_kit_ends_nothing_afterwards() -> None:
    kit, fired = await _kit(ttl=0.1)
    await _verified(kit)
    await _eventually(lambda: fired)  # the completion

    await kit.close()
    await asyncio.sleep(0.2)

    assert [t for t, _ in fired] == [HookTrigger.ON_VERIFICATION_COMPLETED]


class _StoreWithoutVerification(InMemoryStore):
    """A store written before verification existed."""

    async def list_verifications_in_force(self) -> list[Verification]:
        return await ConversationStore.list_verifications_in_force(self)


async def test_a_kit_whose_store_keeps_no_verification_still_starts() -> None:
    kit = RoomKit(store=_StoreWithoutVerification())

    async with kit:
        assert await kit.store.get_room("nope") is None
