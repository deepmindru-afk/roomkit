"""A held message's delivery is named by its key until its release is answered,
and a cancelled one is released with no answer (RFC §10.1, §11.7.6)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import (
    HookTrigger,
    InboundMessage,
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
from roomkit.store.memory import InMemoryStore

ALICE = "+15550000001"
KEY = "task-42"


class _SlowResolver(MockIdentityResolver):
    """Identity resolution that reads a slow store: the call is in flight."""

    def __init__(self) -> None:
        super().__init__(mapping={ALICE: Identity(id="id-alice")})
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


def _kit(
    *,
    store: InMemoryStore | None = None,
    ai: MockAIProvider | None = None,
    **policy: Any,
) -> tuple[RoomKit, MockAIProvider, _SlowResolver]:
    ai = ai or MockAIProvider(responses=["answer"] * 20)
    resolver = _SlowResolver()
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


def _message(body: str, key: str | None = KEY) -> InboundMessage:
    return InboundMessage(
        channel_id="sms", sender_id=ALICE, content=TextContent(body=body), delivery_key=key
    )


async def _settle() -> None:
    await asyncio.sleep(0.3)


async def _verify(kit: RoomKit, first: Any, door: str = "link") -> None:
    """The sender verifies through *door*: the integrator's page completing
    the request (``link``), or the code replied in the conversation."""
    request, secret = await kit.request_verification(
        "id-alice",
        room_id=first.room_id,
        channel_id="sms",
        method=door,
        window_start=first.created_at,
    )
    if door == "code":
        await kit.process_inbound(_message(secret, key=None))
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
    assert kit._delivery_keys._named == {} and kit._delivery_keys._held == {}
    await kit.close()


@pytest.mark.parametrize("door", DOORS)
async def test_a_held_message_nobody_cancelled_is_answered_and_named_no_more(door: str) -> None:
    kit, ai, _ = _kit()
    result = await kit.process_inbound(_message("Did my transfer go through?"))
    await _settle()
    assert KEY in kit._delivery_keys._named

    await _verify(kit, result.event, door)

    assert len(ai.calls) == 1
    assert kit._delivery_keys._named == {} and kit._delivery_keys._held == {}
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

    replies = [
        e for e in await kit.store.list_events(result.event.room_id) if e.source.channel_id == "ai"
    ]
    assert replies == []
    await kit.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_a_release_in_another_process_reads_the_cancellation(cancel: bool) -> None:
    """One store, two processes: the message arrives and is cancelled in the
    first, the sender verifies through the second."""
    store = InMemoryStore()
    first, first_ai, _ = _kit(store=store)
    second, second_ai, _ = _kit(store=store)
    result = await first.process_inbound(_message("Wire 500 to Bob"))
    await _settle()
    if cancel:
        assert await first.cancel_delivery(KEY, reason="user_stopped") == 1
        await _settle()

    await _verify(second, result.event)

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
    assert kit._delivery_keys._named == {} and kit._delivery_keys._held == {}
    await kit.close()


async def test_a_held_message_past_its_release_age_is_named_no_more() -> None:
    kit, _, _ = _kit(release_max_age=0.2)
    await kit.process_inbound(_message("Wire 500 to Bob"))
    await asyncio.sleep(0.3)

    assert await kit.cancel_delivery(KEY) == 0
    assert kit._delivery_keys._named == {} and kit._delivery_keys._held == {}
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
