"""The inbound pipeline labels, holds and consumes codes (RFC §10.1 step 5b, §11.7.3-4)."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import (
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    VerificationPolicy,
    VerificationStatus,
)
from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.core.mixins.verification import HELD_KEY
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.identity import Identity
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

ALICE = "+15550000001"


_DEFAULT = VerificationPolicy()


def _kit(policy: VerificationPolicy | None = _DEFAULT) -> tuple[RoomKit, MockAIProvider]:
    ai = MockAIProvider(responses=["ok"] * 10)
    kit = RoomKit(
        identity_resolver=MockIdentityResolver(mapping={ALICE: Identity(id="id-alice")}),
        verification_policy=policy,
    )
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(AIChannel("ai", provider=ai))

    @kit.hook(HookTrigger.ON_ROOM_CREATED)
    async def agent_joins(event: Any, ctx: Any) -> None:
        await kit.attach_channel(ctx.room.id, "ai")

    return kit, ai


async def _text(kit: RoomKit, body: str) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id=ALICE, content=TextContent(body=body))
    )


async def _settle() -> None:
    await asyncio.sleep(0.2)


def _prompt(ai: MockAIProvider) -> str:
    return " ".join(str(m.content) for m in ai.calls[-1].messages)


async def test_an_unverified_message_is_held_from_the_agent() -> None:
    kit, ai = _kit()
    asked: list[str] = []

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def required(event: Any, ctx: Any) -> HookResult:
        asked.append(event.content.body)
        return HookResult.allow()  # watching only: the message stays held

    result = await _text(kit, "Did my transfer go through?")
    await _settle()

    stored = await kit.store.get_event(result.event.id)
    assert stored is not None
    assert stored.visibility == "transport" and HELD_KEY in stored.metadata
    assert stored.source.verification == VerificationStatus.UNVERIFIED
    assert asked == ["Did my transfer go through?"]
    assert ai.calls == []
    await kit.close()


async def test_a_hook_lets_a_general_question_through_labelled() -> None:
    kit, ai = _kit()

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def general_questions(event: Any, ctx: Any) -> HookResult:
        if "hours" in event.content.body:
            return HookResult.allow_unverified()
        return HookResult.allow()

    held = await _text(kit, "My account number is 12345, what's my balance?")
    passed = await _text(kit, "What are your opening hours?")
    await _settle()

    stored = await kit.store.get_event(passed.event.id)
    assert stored is not None and stored.visibility == "all"
    assert stored.source.verification == VerificationStatus.UNVERIFIED
    assert len(ai.calls) == 1
    assert "opening hours" in _prompt(ai)
    assert "12345" not in _prompt(ai)  # the held message is not in the agent's context
    assert (await kit.store.get_event(held.event.id)).visibility == "transport"  # type: ignore[union-attr]
    await kit.close()


async def test_a_hook_blocks_a_message_before_it_is_stored() -> None:
    kit, ai = _kit()

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
    async def refuse(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("verify first")

    result = await _text(kit, "Close my account")

    assert result.blocked and result.reason == "verify first"
    assert result.event is None
    await kit.close()


async def test_a_verified_sender_reaches_the_agent() -> None:
    kit, ai = _kit()
    first = await _text(kit, "hello")
    room_id = first.event.room_id
    request, secret = await kit.request_verification("id-alice", room_id=room_id, channel_id="sms")
    await kit.complete_verification(request.id, secret)
    ai.calls.clear()

    result = await _text(kit, "Did my transfer go through?")
    await _settle()

    stored = await kit.store.get_event(result.event.id)
    assert stored is not None and stored.visibility == "all"
    assert stored.source.verification == VerificationStatus.VERIFIED
    assert ai.calls
    await kit.close()


async def test_a_code_reply_is_consumed_and_verifies() -> None:
    kit, ai = _kit()
    first = await _text(kit, "hello")
    room_id = first.event.room_id
    request, code = await kit.request_verification(
        "id-alice", room_id=room_id, channel_id="sms", method="code"
    )
    wrong = str((int(code) + 1) % 10**6).zfill(6)

    refused = await _text(kit, wrong)
    accepted = await _text(kit, f" {code} ")

    assert refused.blocked and refused.reason == "verification_code"
    assert accepted.blocked and accepted.reason == "verification_code"
    events = await kit.store.list_events(room_id, limit=50)
    bodies = " ".join(getattr(e.content, "body", "") for e in events)
    assert code not in bodies and wrong not in bodies
    assert (await kit.verification_of("id-alice")).status == VerificationStatus.VERIFIED
    stored = await kit.store.get_verification_request(request.id)
    assert stored is not None and stored.attempts == 1
    await kit.close()


async def test_digits_with_no_code_request_open_are_an_ordinary_message() -> None:
    kit, _ = _kit()

    result = await _text(kit, "123456")

    assert not result.blocked and result.event is not None
    assert result.event.visibility == "transport"  # held like any unverified message
    await kit.close()


async def test_a_kit_without_a_policy_keeps_its_pipeline() -> None:
    kit, ai = _kit(policy=None)

    result = await _text(kit, "Did my transfer go through?")
    await _settle()

    assert result.event.source.verification is None
    assert result.event.visibility == "all"
    assert ai.calls
    await kit.close()


async def test_a_policy_that_does_not_hold_only_labels() -> None:
    kit, ai = _kit(policy=VerificationPolicy(hold_unverified=False))

    result = await _text(kit, "Did my transfer go through?")
    await _settle()

    assert result.event.source.verification == VerificationStatus.UNVERIFIED
    assert result.event.visibility == "all"
    assert ai.calls
    await kit.close()
