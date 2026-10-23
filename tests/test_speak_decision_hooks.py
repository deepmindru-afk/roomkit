"""A SYNC ON_SPEAK_DECISION hook changes the decision before the channel acts
on it, as a SYNC hook changes its event on the other triggers (RMK-914).

An application acts on a decision in code: in a meeting, an offer raises the
agent's hand and the agent waits to be given the floor, which the LLM, left
to itself, would not do. The hook turns the offer into silence; the ASYNC
hooks observe the decision that stands.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from types import SimpleNamespace
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


# -- a thought that comes back late ----------------------------------------------------

_PRICE = Thought("They look for the licence price; I know it.", ("It costs 1,200 $.",))


class _Late(MockThinker):
    """Thinks as MockThinker does once *back* is set: the thought comes back
    after the channel decided without it, however fast the runner is."""

    def __init__(self, thoughts: list[Thought]) -> None:
        super().__init__(thoughts)
        self.back = asyncio.Event()

    async def think(self, previous: Thought, context: Any) -> Thought:
        await self.back.wait()
        return await super().think(previous, context)


async def _thought_back(kit: RoomKit, thinker: _Late) -> None:
    """Let the thought come back and wait for the room's mind to be done with
    it: the late decision it owes runs inside its thinking task."""
    thinker.back.set()
    mind = kit.channels["ai1"]._minds["r1"]  # type: ignore[attr-defined]
    await asyncio.wait_for(asyncio.shield(mind._task), 2)


def _decisions(kit: RoomKit) -> list[SpeakDecisionEvent]:
    """Every decision, through a SYNC hook that turns an offer into a raised hand."""
    seen: list[SpeakDecisionEvent] = []

    @kit.hook(HookTrigger.ON_SPEAK_DECISION)
    async def hand(event: SpeakDecisionEvent, ctx: Any) -> HookResult:
        seen.append(event)
        if event.decision.mode != "offer":
            return HookResult.allow()
        return _decided(event, "silent", "hand raised")

    return seen


async def _said_and_thought(kit: RoomKit, thinker: _Late, *bodies: str) -> None:
    for body in bodies:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))
        )
    await _thought_back(kit, thinker)
    await kit.close()


async def test_a_thought_back_late_is_decided_again_for_the_hooks_and_runs_no_turn() -> None:
    thinker = _Late([_PRICE])
    kit, provider = await _kit(
        MockSpeakPolicy(["silent", SpeakDecision("offer", reason="has something to add")]),
        thinker=thinker,
        think_wait=0.01,
    )
    seen = _decisions(kit)

    await _said_and_thought(kit, thinker, "We are looking for the licence price.")

    first, late = seen
    assert (first.late, first.decision.mode) == (False, "silent")
    assert (late.late, late.asked_again, late.decision.mode) == (True, True, "offer")
    assert late.thought == _PRICE
    assert provider.calls == []  # the hook raised the hand; no turn runs on it


async def test_a_late_decision_to_speak_runs_no_turn_either() -> None:
    thinker = _Late([_PRICE])
    kit, provider = await _kit(
        MockSpeakPolicy(["silent", "speak"]),
        thinker=thinker,
        think_wait=0.01,
    )
    seen = _decisions(kit)

    await _said_and_thought(kit, thinker, "We are looking for the licence price.")

    assert [(e.late, e.decision.mode) for e in seen] == [(False, "silent"), (True, "speak")]
    assert provider.calls == []


async def test_only_the_latest_event_s_thought_is_decided_late() -> None:
    thinker = _Late([_PRICE])
    kit, _ = await _kit(
        MockSpeakPolicy(
            ["silent", "silent", SpeakDecision("offer", reason="has something to add")]
        ),
        thinker=thinker,
        think_wait=0.01,
    )
    seen = _decisions(kit)

    await _said_and_thought(kit, thinker, "We are looking for the licence price.", "Anyone knows?")

    late = [e for e in seen if e.late]
    assert [e.event.content.body for e in late] == ["Anyone knows?"]  # ty: ignore[unresolved-attribute]


async def test_a_thought_with_nothing_to_say_is_not_decided_late() -> None:
    thinker = _Late([Thought("They talk about the weather.")])
    kit, _ = await _kit(
        MockSpeakPolicy(["silent", "offer"]),
        thinker=thinker,
        think_wait=0.01,
    )
    seen = _decisions(kit)

    await _said_and_thought(kit, thinker, "Nice weather today.")

    assert [e.late for e in seen] == [False]


async def test_a_decision_event_carries_the_thought_the_policy_read(advance: Any) -> None:
    kit, _ = await _kit(
        MockSpeakPolicy(["silent", SpeakDecision("offer", reason="has something to add")]),
        thinker=MockThinker([_PRICE]),
        think_wait=1.0,
    )
    seen = _decisions(kit)

    await _say(kit, "We are looking for the licence price.", advance)

    assert [(e.asked_again, e.late, e.thought) for e in seen] == [
        (False, False, Thought()),
        (True, False, _PRICE),
    ]


async def test_a_thought_taken_while_the_agent_answers_wants_to_say_nothing() -> None:
    """Thinking about an event while the agent is still answering the one before,
    the thought's context lacks that answer: what it wants to say is the answer
    itself, so it is said, and decided on neither in time nor late."""
    kit = RoomKit()
    thinker = _Late([_PRICE])
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["Yes?"]),
        speak_policy=MockSpeakPolicy(["silent", SpeakDecision("offer", reason="has more")]),
        thinker=thinker,
        think_wait=0.01,
    )
    kit.register_channel(channel)
    kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r1", "sms1")
    seen = _decisions(kit)
    channel._active_loops["answering"] = SimpleNamespace(room_id="r1")  # ty: ignore[invalid-assignment]

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="I have it."))
    )
    await _thought_back(kit, thinker)
    del channel._active_loops["answering"]
    thought = channel._thought_of("r1")
    await kit.close()

    assert [e.late for e in seen] == [False]
    assert thought == _PRICE.said()
