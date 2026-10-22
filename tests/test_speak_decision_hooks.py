"""A SYNC ON_SPEAK_DECISION hook changes the decision before the channel acts
on it, as a SYNC hook changes its event on the other triggers (RMK-914).

An application acts on a decision in code: in a meeting, an offer raises the
agent's hand and the agent waits to be given the floor, which the LLM, left
to itself, would not do. The hook turns the offer into silence; the ASYNC
hooks observe the decision that stands.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    MockSpeakPolicy,
    MockThinker,
    RoomKit,
    SpeakDecision,
    SpeakDecisionEvent,
    Thought,
)
from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from tests.conftest import make_event


async def _kit(policy: MockSpeakPolicy, **channel: Any) -> tuple[RoomKit, MockAIProvider]:
    kit = RoomKit()
    provider = MockAIProvider(responses=["Yes?"])
    kit.register_channel(AIChannel("ai1", provider=provider, speak_policy=policy, **channel))
    kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r1", "sms1")
    return kit, provider


def _observed(kit: RoomKit) -> list[SpeakDecisionEvent]:
    seen: list[SpeakDecisionEvent] = []

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def observe(event: SpeakDecisionEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


def _decided(event: SpeakDecisionEvent, mode: str, reason: str) -> HookResult:
    decision = SpeakDecision(mode, reason=reason)
    return HookResult.modify(dataclasses.replace(event, decision=decision))


async def _say(kit: RoomKit, body: str, advance: Any) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))
    )
    await advance()
    await kit.close()


async def test_an_offer_raises_the_hand_in_code_and_the_agent_stays_silent(
    advance: Any,
) -> None:
    kit, provider = await _kit(MockSpeakPolicy([SpeakDecision("offer", reason="has more")]))
    raised: list[str] = []
    seen = _observed(kit)

    @kit.hook(HookTrigger.ON_SPEAK_DECISION)
    async def raise_hand(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        if event.decision.mode != "offer":
            return HookResult.allow()
        raised.append(event.channel_id)
        return _decided(event, "silent", "hand raised")

    await _say(kit, "We are looking for the licence price.", advance)

    assert raised == ["ai1"]
    assert provider.calls == []  # no offer said aloud
    [observed] = seen
    assert (observed.decision.mode, observed.decision.reason) == ("silent", "hand raised")


async def test_a_hook_gives_the_agent_the_floor(advance: Any) -> None:
    kit, provider = await _kit(MockSpeakPolicy([SpeakDecision("silent", reason="side talk")]))

    @kit.hook(HookTrigger.ON_SPEAK_DECISION)
    async def floor(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        return _decided(event, "speak", "given the floor")

    await _say(kit, "Go ahead.", advance)

    assert len(provider.calls) == 1


async def test_a_block_leaves_the_agent_silent(advance: Any) -> None:
    kit, provider = await _kit(MockSpeakPolicy(["speak"]))
    seen = _observed(kit)

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, name="hand_down")
    async def hold(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        return HookResult.block("wait for the floor")

    await _say(kit, "Anyone?", advance)

    assert provider.calls == []
    [observed] = seen
    assert (observed.decision.mode, observed.decision.reason) == ("silent", "blocked by hand_down")


async def test_each_hook_sees_the_decision_the_previous_one_left(advance: Any) -> None:
    kit, provider = await _kit(MockSpeakPolicy([SpeakDecision("offer", reason="has more")]))
    second_saw: list[str] = []

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, priority=0)
    async def first(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        return _decided(event, "silent", "hand raised")

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, priority=1)
    async def second(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        second_saw.append(event.decision.reason)
        return HookResult.allow()

    await _say(kit, "We are looking for the licence price.", advance)

    assert second_saw == ["hand raised"]
    assert provider.calls == []


@pytest.mark.parametrize(("fail_closed", "calls"), [(False, 1), (True, 0)])
async def test_a_hook_that_fails_leaves_the_decision_unless_it_fails_closed(
    advance: Any, fail_closed: bool, calls: int
) -> None:
    kit, provider = await _kit(MockSpeakPolicy(["speak"]))
    seen = _observed(kit)

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, name="hand", fail_closed=fail_closed)
    async def broken(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        raise RuntimeError("boom")

    await _say(kit, "Anyone?", advance)

    assert len(provider.calls) == calls
    if fail_closed:
        assert [e.decision.reason for e in seen] == ["hook_error:hand"]


@pytest.mark.parametrize(("fail_closed", "mode"), [(False, "speak"), (True, "silent")])
async def test_hooks_that_cannot_run_leave_the_decision_unless_one_fails_closed(
    fail_closed: bool, mode: str
) -> None:
    """The room's context would not build, so no ON_SPEAK_DECISION hook runs:
    a fail-closed one silences the agent, as its own failure would."""
    kit = RoomKit()
    channel = AIChannel("ai1", provider=MockAIProvider(), speak_policy=MockSpeakPolicy(["speak"]))
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, name="hand", fail_closed=fail_closed)
    async def hand(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        return HookResult.allow()

    callback = channel._speak_decision_hook
    assert callback is not None
    kit._build_context = AsyncMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    said = make_event(room_id="r1", channel_id="sms1")
    event = SpeakDecisionEvent("r1", "ai1", said, SpeakDecision("speak"))

    decision = await callback(event)

    assert decision.mode == mode
    if fail_closed:
        assert decision.reason == "hook_error:hand"
    await kit.close()


async def test_a_payload_that_is_not_a_decision_event_leaves_the_decision(
    advance: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit, provider = await _kit(MockSpeakPolicy(["speak"]))

    @kit.hook(HookTrigger.ON_SPEAK_DECISION)
    async def bare(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        return HookResult.modify(SpeakDecision("silent"))  # the decision, not its event

    with caplog.at_level(logging.ERROR):
        await _say(kit, "Anyone?", advance)

    assert len(provider.calls) == 1
    assert any("SpeakDecisionEvent was expected" in r.getMessage() for r in caplog.records)


async def test_the_decision_taken_again_once_the_agent_thought_goes_through_the_hooks(
    advance: Any,
) -> None:
    thought = Thought("They look for the licence price; I know it.", ("It costs 1,200 $.",))
    kit, provider = await _kit(
        MockSpeakPolicy(["silent", SpeakDecision("offer", reason="has something to add")]),
        thinker=MockThinker([thought]),
        think_wait=1.0,
    )
    seen = _observed(kit)

    @kit.hook(HookTrigger.ON_SPEAK_DECISION)
    async def raise_hand(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        if event.decision.mode != "offer":
            return HookResult.allow()
        return _decided(event, "silent", "hand raised")

    await _say(kit, "We are looking for the licence price.", advance)

    assert provider.calls == []
    assert [(e.asked_again, e.decision.reason) for e in seen][-1] == (True, "hand raised")
