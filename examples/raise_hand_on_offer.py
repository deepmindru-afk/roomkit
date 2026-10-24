"""An agent that raises its hand instead of offering, in code.

Two people meet in a room with an agent, Nova, who knows what the team's
licences cost. Her AI channel has a speak policy and a thinker (RFC §6.4). A
SYNC ``ON_SPEAK_DECISION`` hook acts on each decision before the channel does:
the application, not the model, takes the action the decision calls for.

1. "Right, we need to order the licences for the team." goes to nobody: Nova
   listens, and thinks she knows the price.
2. "Does anyone know how much it costs, per person?" asks nobody in particular,
   and her thought answers it: the policy decides to offer. The hook raises her
   hand and leaves her silent (``HookResult.modify`` with a silent decision),
   so she interrupts no one.
3. "Nova, you raised your hand, go ahead." gives her the floor: the policy
   decides to speak, and the hook adds a note to the decision saying why her
   hand was up, then lowers it.

A decision taken late (``SpeakDecisionEvent.late``: the thought came back once
the event's turn was over) runs no turn whatever it says: the hook raises the
hand on it too. Nova's answers come from a scripted model, the judgments from
a scripted classifier: no key needed.

Run with:
    uv run python examples/raise_hand_on_offer.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    ClassifierSpeakPolicy,
    HookResult,
    HookTrigger,
    InboundMessage,
    MockClassifier,
    MockThinker,
    RoomKit,
    SMSChannel,
    SpeakDecision,
    SpeakDecisionEvent,
    TextContent,
    Thought,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.providers.sms import MockSMSProvider

logger = setup_logging("example.raise_hand_on_offer")

NOVA = (
    "You are Nova, the team's assistant. You know that the design tool's licence costs "
    "$1,200 a year per person, and that the vendor gives 10% off beyond ten licences."
)

LINES = [
    ("Sylvain", "Right, we need to order the licences for the team."),
    ("Paul", "Does anyone know how much it costs, per person?"),
    ("Sylvain", "Nova, you raised your hand, go ahead."),
]

PRICE = Thought(
    "They are about to order the licences; I know the price.",
    ("A licence costs $1,200 a year per person, with 10% off beyond ten.",),
)

# Per policy call: each line, then the second asked again once the agent thought.
JUDGMENTS = [
    {"directness": 0.0, "unfinished": 0.06, "request": 0.48},
    {"directness": 0.0, "unfinished": 0.06, "request": 0.52, "answers": 0.46},
    {"directness": 0.07, "request": 0.82, "answers": 0.96},
    {"directness": 0.07, "request": 0.82, "answers": 0.96},
    {"directness": 2.99, "request": 0.96},
]


class Hand:
    """Nova's hand: what an application would show in its meeting (a raised-hand
    reaction, a queue to speak), and what she raised it for."""

    def __init__(self) -> None:
        self.raised_for: tuple[str, ...] = ()

    def raise_for(self, wanted: tuple[str, ...]) -> None:
        if not self.raised_for:
            logger.info("  ✋ Nova raises her hand")
        self.raised_for = wanted

    def lower(self) -> tuple[str, ...]:
        wanted, self.raised_for = self.raised_for, ()
        logger.info("  ✋ Nova lowers her hand")
        return wanted


def floor_note(wanted: tuple[str, ...]) -> str:
    """What the agent given the floor reads in its turn's notes."""
    items = "".join(f"\n- {item}" for item in wanted)
    return (
        f"You raised your hand to add:{items}\nYou have the floor now: answer what was "
        "just said to you first, then say it briefly, if it is still useful."
    )


def replaced(event: SpeakDecisionEvent, decision: SpeakDecision) -> HookResult:
    """The hook's result: *event*'s decision replaced by *decision*."""
    return HookResult.modify(dataclasses.replace(event, decision=decision))


async def main() -> None:
    kit = RoomKit()
    classifier = MockClassifier(JUDGMENTS)
    thinker = MockThinker([PRICE], delay=0.4)
    nova = AIChannel(
        "nova",
        provider=MockAIProvider(
            responses=["Thanks! A licence is $1,200 a year per person, 10% off beyond ten."]
        ),
        system_prompt=NOVA,
        speak_policy=ClassifierSpeakPolicy(classifier, agent_name="Nova"),
        thinker=thinker,
        think_wait=4.0,
    )
    kit.register_channel(nova)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    await kit.create_room(room_id="meeting")
    await kit.attach_channel("meeting", "nova", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("meeting", "sms")
    for name in ("Sylvain", "Paul"):
        await kit.ensure_participant("meeting", "sms", name.lower(), display_name=name)

    hand = Hand()

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, name="raise_hand")
    async def on_decision(event: SpeakDecisionEvent, ctx: object) -> HookResult:
        decision = event.decision
        logger.info("  → the policy decides %s (%s)", decision.mode, decision.reason)
        if decision.mode == "silent":
            return HookResult.allow()
        wanted = event.thought.want_to_say if event.thought else ()
        if decision.mode == "offer" or event.late:
            # An intention to speak: the hand goes up, and Nova keeps quiet.
            hand.raise_for(wanted)
            return replaced(event, SpeakDecision("silent", reason="hand raised"))
        if not hand.raised_for:
            return HookResult.allow()
        note = floor_note(hand.lower())
        return replaced(event, dataclasses.replace(decision, notes=(*decision.notes, note)))

    for name, body in LINES:
        logger.info('%s: "%s"', name, body)
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id=name.lower(),
                content=TextContent(body=body),
                metadata={"sender_name": name},
            )
        )
        await asyncio.sleep(0.2)

    answers = [e for e in await kit.store.list_events("meeting") if e.source.channel_id == "nova"]
    for answer in answers:
        logger.info("Nova: %s", answer.content.body)
    await kit.close()
    await classifier.close()
    await thinker.close()


if __name__ == "__main__":
    asyncio.run(main())
