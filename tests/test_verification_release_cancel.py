"""A held message's delivery is named by its key until its release is answered,
and a cancelled one is released with no answer (RFC §10.1, §11.7.6)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from roomkit import (
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    VerificationPolicy,
)
from roomkit.channels import SMSChannel, WebSocketChannel
from roomkit.channels.ai import AIChannel
from roomkit.core.mixins.verification import HELD_KEY
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.identity import Identity
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore

ALICE = "+15550000001"
KEY = "task-42"


class _SlowResolver(MockIdentityResolver):
    """Identity resolution that reads a slow store: the call is in flight."""

    def __init__(self, sender: str) -> None:
        super().__init__(mapping={sender: Identity(id=_identity(sender))})
        self.reading = asyncio.Event()

    async def resolve(self, message: Any, context: Any) -> Any:
        self.reading.set()
        await asyncio.sleep(0.1)
        return await super().resolve(message, context)


class _SlowAgent(MockAIProvider):
    """An agent whose answer takes a while, so it can be cut."""

    def __init__(self) -> None:
        super().__init__(responses=["answer"] * 20)
        self.started = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        self.started.set()
        await asyncio.sleep(5)
        return await super().generate(context)


def _identity(sender: str) -> str:
    return f"id-{sender}"


def _kit(
    *,
    store: ConversationStore | None = None,
    ai: MockAIProvider | None = None,
    sender: str = ALICE,
    **policy: Any,
) -> tuple[RoomKit, MockAIProvider, _SlowResolver]:
    ai = ai or MockAIProvider(responses=["answer"] * 20)
    resolver = _SlowResolver(sender)
    kit = RoomKit(
        store=store or InMemoryStore(),
        identity_resolver=resolver,
        verification_policy=VerificationPolicy(**policy),
    )
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(AIChannel("ai", provider=ai))
    kit.register_channel(WebSocketChannel("bank"))

    @kit.hook(HookTrigger.ON_ROOM_CREATED)
    async def joins(event: Any, ctx: Any) -> None:
        await kit.attach_channel(ctx.room.id, "ai")
        await kit.attach_channel(ctx.room.id, "bank")

    return kit, ai, resolver


def _message(body: str, key: str | None = KEY, sender: str = ALICE) -> InboundMessage:
    return InboundMessage(
        channel_id="sms", sender_id=sender, content=TextContent(body=body), delivery_key=key
    )


async def _settle() -> None:
    await asyncio.sleep(0.3)


async def _replies(kit: RoomKit, room_id: str) -> list[RoomEvent]:
    return [e for e in await kit.store.list_events(room_id) if e.source.channel_id == "ai"]


def _named_nothing(kit: RoomKit) -> bool:
    return kit._delivery_keys._named == {} and kit._delivery_keys._held == {}


async def _verify(kit: RoomKit, first: Any, door: str = "link", sender: str = ALICE) -> None:
    """The sender verifies through *door*: the integrator's page completing
    the request (``link``), or the code replied in the conversation."""
    request, secret = await kit.request_verification(
        _identity(sender),
        room_id=first.room_id,
        channel_id="sms",
        method=door,
        window_start=first.created_at,
    )
    if door == "code":
        await kit.process_inbound(_message(secret, key=None, sender=sender))
    else:
        await kit.complete_verification(request.id, secret)
    await _settle()


DOORS = ["link", "code"]


async def _in_flight(kit: RoomKit, resolver: _SlowResolver) -> tuple[Any, int]:
    """The message sent, cancelled while its call resolves the sender."""
    call = asyncio.create_task(kit.process_inbound(_message("Wire 500 to Bob")))
    await resolver.reading.wait()
    counted = await kit.cancel_delivery(KEY, reason="user_stopped")
    return await call, counted


async def _after_the_call(kit: RoomKit, resolver: _SlowResolver) -> tuple[Any, int]:
    """The message sent, cancelled once its call returned, the message held."""
    result = await kit.process_inbound(_message("Wire 500 to Bob"))
    await _settle()
    return result, await kit.cancel_delivery(KEY, reason="user_stopped")


WHEN: dict[str, Callable[[RoomKit, _SlowResolver], Awaitable[tuple[Any, int]]]] = {
    "in flight": _in_flight,
    "after the call": _after_the_call,
}


@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("when", WHEN)
async def test_a_cancelled_held_message_is_released_with_no_answer(when: str, door: str) -> None:
    kit, ai, resolver = _kit()
    result, counted = await WHEN[when](kit, resolver)
    assert counted == 1

    await _verify(kit, result.event, door)

    stored = await kit.store.get_event(result.event.id)
    assert stored is not None and stored.visibility == "all"
    assert HELD_KEY not in stored.metadata
    assert ai.calls == []
    assert _named_nothing(kit)
    await kit.close()


@pytest.mark.parametrize("door", DOORS)
async def test_a_held_message_nobody_cancelled_is_answered_and_named_no_more(door: str) -> None:
    kit, ai, _ = _kit()
    result = await kit.process_inbound(_message("Did my transfer go through?"))
    await _settle()
    assert KEY in kit._delivery_keys._named

    await _verify(kit, result.event, door)

    assert len(ai.calls) == 1
    assert _named_nothing(kit)
    assert await kit.cancel_delivery(KEY) == 0
    await kit.close()


async def test_a_cancel_during_the_released_answer_cuts_it() -> None:
    agent = _SlowAgent()
    kit, _, _ = _kit(ai=agent)
    result = await kit.process_inbound(_message("Did my transfer go through?"))
    await _settle()
    release = asyncio.create_task(_verify(kit, result.event))
    await asyncio.wait_for(agent.started.wait(), 2)

    assert await kit.cancel_delivery(KEY, reason="user_stopped") == 1
    await asyncio.wait_for(release, 2)

    assert await _replies(kit, result.event.room_id) == []
    await kit.close()


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def two_stores(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[tuple[ConversationStore, ConversationStore]]:
    """Two processes' views of one store."""
    if request.param == "memory":
        shared = InMemoryStore()
        yield shared, shared
        return
    if request.param == "sqlite":
        yield SQLiteStore(tmp_path / "rooms.db"), SQLiteStore(tmp_path / "rooms.db")
        return
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set")
    postgres = pytest.importorskip("roomkit.store.postgres")
    stores = (postgres.PostgresStore(dsn=dsn), postgres.PostgresStore(dsn=dsn))
    for store in stores:
        await store.init(min_size=1, max_size=2)
    try:
        yield stores
    finally:
        for store in stores:
            await store.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_a_release_in_another_process_reads_the_cancellation(
    cancel: bool, two_stores: tuple[ConversationStore, ConversationStore]
) -> None:
    """One store, two processes: the message arrives and is cancelled in the
    first, the sender verifies through the second. A sender of its own: a
    database store outlives the test, and its verifications with it."""
    sender = f"+1555{uuid4().int % 10**7:07d}"
    first, first_ai, _ = _kit(store=two_stores[0], sender=sender)
    second, second_ai, _ = _kit(store=two_stores[1], sender=sender)
    result = await first.process_inbound(_message("Wire 500 to Bob", sender=sender))
    await _settle()
    if cancel:
        assert await first.cancel_delivery(KEY, reason="user_stopped") == 1
        await _settle()

    await _verify(second, result.event, sender=sender)

    assert first_ai.calls == []
    assert len(second_ai.calls) == (0 if cancel else 1)
    await first.close()
    await second.close()


@pytest.mark.parametrize("cancelled", ["earlier", "last"])
async def test_only_the_last_released_messages_cancellation_silences_the_release(
    cancelled: str,
) -> None:
    """The last released message alone asks for an answer, the earlier ones
    being its context: its cancellation leaves the release unanswered, an
    earlier one's does not."""
    kit, ai, _ = _kit()
    earlier = await kit.process_inbound(_message("What's my balance?", key="earlier"))
    await kit.process_inbound(_message("Wire 500 to Bob", key="last"))
    await _settle()
    assert await kit.cancel_delivery(cancelled, reason="user_stopped") == 1
    await _settle()

    await _verify(kit, earlier.event)

    assert len(ai.calls) == (1 if cancelled == "earlier" else 0)
    assert _named_nothing(kit)
    await kit.close()


async def test_a_held_message_past_its_release_age_is_named_no_more() -> None:
    kit, _, _ = _kit(release_max_age=0.2)
    await kit.process_inbound(_message("Wire 500 to Bob"))
    await asyncio.sleep(0.3)

    assert await kit.cancel_delivery(KEY) == 0
    assert _named_nothing(kit)
    await kit.close()


async def test_another_organizations_cancel_reaches_no_held_message() -> None:
    kit, ai, _ = _kit()
    room = await kit.create_room(organization_id="tenant-1")
    result = await kit.process_inbound(_message("Did my transfer go through?"), room_id=room.id)
    await _settle()

    assert await kit.cancel_delivery(KEY, organization_id="tenant-2") == 0
    await _verify(kit, result.event)

    assert len(ai.calls) == 1
    await kit.close()


async def test_a_release_during_the_held_messages_own_delivery_is_named() -> None:
    """The sender verifies while the held message's delivery set still runs
    (a slow AFTER_BROADCAST hook): the release's answer is named all the
    same, and a cancel cuts it."""
    agent = _SlowAgent()
    kit, _, _ = _kit(ai=agent)
    gate = asyncio.Event()
    delivered: list[RoomEvent] = []

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def slow(event: RoomEvent, ctx: Any) -> None:
        if event.source.channel_id == "sms":
            delivered.append(event)
            await gate.wait()

    call = asyncio.create_task(kit.process_inbound(_message("Wire 500 to Bob")))
    while not delivered:
        await asyncio.sleep(0.01)
    release = asyncio.create_task(_verify(kit, delivered[0]))
    await asyncio.sleep(0.3)
    gate.set()
    await call
    await asyncio.wait_for(agent.started.wait(), 2)

    assert await kit.cancel_delivery(KEY, reason="user_stopped") == 1
    await asyncio.wait_for(release, 3)
    assert await _replies(kit, delivered[0].room_id) == []
    assert _named_nothing(kit)
    await kit.close()


async def test_a_release_from_a_hook_keeps_its_answer_named() -> None:
    """``complete_verification`` called under the room's lock (a sync hook):
    the release's answer waits in the lane, still named, so a cancel stops it."""
    kit, ai, _ = _kit()
    held = await kit.process_inbound(_message("Wire 500 to Bob"))
    await _settle()
    request, secret = await kit.request_verification(
        _identity(ALICE),
        room_id=held.event.room_id,
        channel_id="sms",
        window_start=held.event.created_at,
    )
    counted: list[int] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def verified_by_the_console(event: RoomEvent, ctx: Any) -> HookResult:
        if event.source.channel_id == "bank":
            await kit.complete_verification(request.id, secret)
            counted.append(await kit.cancel_delivery(KEY, reason="user_stopped"))
            return HookResult.block("console note")
        return HookResult.allow()

    note = InboundMessage(
        channel_id="bank", sender_id="advisor", content=TextContent(body="Verified by phone")
    )
    await kit.process_inbound(note, room_id=held.event.room_id)
    await _settle()

    assert counted == [1]
    assert ai.calls == []
    assert _named_nothing(kit)
    await kit.close()


async def test_a_deferred_held_message_cancelled_by_key_is_released_with_no_answer() -> None:
    kit, ai, _ = _kit()
    result = await kit.process_inbound(_message("Wire 500 to Bob"), defer_delivery=True)
    await result.delivery.wait()

    assert await kit.cancel_delivery(KEY, reason="user_stopped") == 1
    await _verify(kit, result.event)

    assert ai.calls == []
    assert _named_nothing(kit)
    await kit.close()


@pytest.mark.parametrize("when", ["while its delivery set runs", "once it ran"])
async def test_the_handle_reaches_a_held_message_while_its_delivery_set_runs(when: str) -> None:
    """The handle follows the delivery set alone (RFC §11.7.6): past it, the
    key is what names the held message."""
    kit, ai, _ = _kit()
    gate = asyncio.Event()

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def slow(event: RoomEvent, ctx: Any) -> None:
        if event.source.channel_id == "sms":
            await gate.wait()

    result = await kit.process_inbound(_message("Wire 500 to Bob", key=None), defer_delivery=True)
    if when == "once it ran":
        gate.set()
        await result.delivery.wait()
    await result.delivery.cancel(reason="user_stopped")
    gate.set()
    await result.delivery.wait()
    await _settle()

    await _verify(kit, result.event)

    assert len(ai.calls) == (0 if when == "while its delivery set runs" else 1)
    assert _named_nothing(kit)
    await kit.close()


async def test_a_held_message_nothing_will_release_is_not_named() -> None:
    """Blocked after it was held, or held under a policy that releases
    nothing: the key names it no longer than its delivery set."""
    blocking, _, _ = _kit()

    @blocking.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: RoomEvent, ctx: Any) -> HookResult:
        return HookResult.block("not now")

    keeping, _, _ = _kit(release_on_verify=False)
    for kit in (blocking, keeping):
        await kit.process_inbound(_message("Wire 500 to Bob"))
        await _settle()
        assert await kit.cancel_delivery(KEY) == 0
        assert _named_nothing(kit)
        await kit.close()


async def test_a_held_message_without_a_key_is_named_no_longer_than_its_delivery() -> None:
    kit, ai, _ = _kit()
    result = await kit.process_inbound(_message("Did my transfer go through?", key=None))
    await _settle()
    assert _named_nothing(kit)

    await _verify(kit, result.event)

    assert len(ai.calls) == 1
    await kit.close()
