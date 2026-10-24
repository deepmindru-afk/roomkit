"""RMK-947 probe: how can host code send into a room without an agent answering?

Compares two shapes of a "host channel" on today's RoomKit, with no framework
change:

- INTELLIGENCE: a SYSTEM-type channel of category INTELLIGENCE whose
  ``on_event`` answers nothing (what roomkit-graph's WorkflowChannel does);
- TRANSPORT: a SYSTEM-type channel of category TRANSPORT whose ``deliver``
  does nothing — the host speaks as a party of the conversation, the way
  CLIChannel "you" writes to an SMS correspondent.

Each scenario prints what a caller observes: SMS sent, agent calls, what the
agent's model reads, attach refusals, time per inbound message. The author
scenario also compares the SYSTEM channel type with an ordinary one, since the
framework gives SYSTEM sources no author.

Run with:
    uv run python scripts/probe_rmk947_host_channel.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from roomkit import (
    Channel,
    ChannelCategory,
    ChannelOutput,
    ChannelType,
    Discussion,
    InboundMessage,
    Participant,
    Pipeline,
    RoomEvent,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.channels.agent import Agent
from roomkit.models.enums import AgentResponsePolicy
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

CUSTOMER = "+15550100001"
SETTLE_S = 0.1


class HostChannel(Channel):
    """A host channel of either category; it answers and receives nothing."""

    def __init__(
        self, category: ChannelCategory, channel_type: ChannelType = ChannelType.SYSTEM
    ) -> None:
        super().__init__("host")
        self.category = category
        self.channel_type = channel_type
        self.delivered: list[RoomEvent] = []
        self.reacted: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: Any) -> RoomEvent:
        msg = "the host channel receives nothing"
        raise TypeError(msg)

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        self.reacted.append(event)
        return ChannelOutput.empty()


class Setup:
    """A kit with an SMS customer room, an agent and a host channel."""

    def __init__(
        self,
        category: ChannelCategory,
        *,
        agent: bool,
        channel_type: ChannelType = ChannelType.SYSTEM,
    ) -> None:
        self.sms = MockSMSProvider()
        self.ai = MockAIProvider(responses=[f"agent answer {i}" for i in range(1, 20)])
        self.host = HostChannel(category, channel_type)
        self.kit = RoomKit()
        self.kit.register_channel(SMSChannel("sms", provider=self.sms))
        self.kit.register_channel(self.host)
        self.with_agent = agent
        if agent:
            self.kit.register_channel(Agent("assistant", provider=self.ai))

    async def customer(self, text: str) -> str:
        result = await self.kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id=CUSTOMER, content=TextContent(body=text))
        )
        await asyncio.sleep(SETTLE_S)
        return result.event.room_id

    async def open_room(self) -> str:
        room_id = await self.customer("hello")
        if self.with_agent:
            await self.kit.attach_channel(
                room_id, "assistant", category=ChannelCategory.INTELLIGENCE
            )
        await self.kit.attach_channel(room_id, "host")
        return room_id

    async def host_says(self, room_id: str, text: str, **kwargs: Any) -> None:
        await self.kit.send_event(room_id, "host", TextContent(body=text), **kwargs)
        await asyncio.sleep(SETTLE_S)

    def texted(self) -> list[str]:
        return [item["event"].content.body for item in self.sms.sent]


async def scenario_text_the_customer(category: ChannelCategory) -> dict[str, Any]:
    """S1/S2: the host texts the customer; the customer answers."""
    s = Setup(category, agent=False)
    room = await s.open_room()
    await s.host_says(room, "Your order shipped.", addressed_to=[])
    await s.customer("thanks!")
    out = {
        "SMS sent": s.texted(),
        "host.deliver calls": len(s.host.delivered),
        "host.on_event calls": len(s.host.reacted),
    }
    await s.kit.close()
    return out


async def scenario_agent_present(category: ChannelCategory) -> dict[str, Any]:
    """S3/S4: an agent in the room; the host speaks with each addressing."""
    s = Setup(category, agent=True)
    room = await s.open_room()
    calls_before = len(s.ai.calls)
    await s.host_says(room, "HOST nobody", addressed_to=[])
    after_nobody = len(s.ai.calls)
    await s.host_says(room, "HOST unaddressed")
    after_unaddressed = len(s.ai.calls)
    await s.host_says(room, "HOST to assistant", addressed_to=["assistant"])
    after_addressed = len(s.ai.calls)
    await s.customer("what did they say?")
    last = s.ai.calls[-1] if s.ai.calls else None
    seen = (
        [
            (m.role, str(m.content)[:60])
            for m in last.messages
            if "HOST" in str(m.content) or "agent answer" in str(m.content)
        ]
        if last
        else []
    )
    mentions = "host" in (last.system_prompt or "") if last else None
    out = {
        "agent turns on host 'nobody' ([])": after_nobody - calls_before,
        "agent turns on host unaddressed (None)": after_unaddressed - after_nobody,
        "agent turns on host addressed": after_addressed - after_unaddressed,
        "SMS sent": s.texted(),
        "agent reads (role, text)": seen,
        "system prompt names 'host'": mentions,
    }
    await s.kit.close()
    return out


async def _room_with(s: Setup, **room_kwargs: Any) -> str:
    """A room created by the host for the customer, the agent and the host attached."""
    await s.kit.create_room(room_id="r", **room_kwargs)
    await s.kit.attach_channel("r", "sms", metadata={"phone_number": CUSTOMER})
    if "orchestration" not in room_kwargs:
        await s.kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)
    await s.kit.attach_channel("r", "host")
    return "r"


async def scenario_addressed_only(category: ChannelCategory) -> dict[str, Any]:
    """S3b: same, in a room whose policy is ADDRESSED_ONLY."""
    s = Setup(category, agent=True)
    room = await _room_with(s, agent_response_policy=AgentResponsePolicy.ADDRESSED_ONLY)
    before = len(s.ai.calls)
    await s.host_says(room, "HOST unaddressed")
    unaddressed = len(s.ai.calls) - before
    await s.host_says(room, "HOST to assistant", addressed_to=["assistant"])
    out = {
        "agent turns on host unaddressed": unaddressed,
        "agent turns on host addressed": len(s.ai.calls) - before - unaddressed,
    }
    await s.kit.close()
    return out


async def scenario_pipeline(category: ChannelCategory) -> dict[str, Any]:
    """S7: a room run by a Pipeline strategy (its router skips agents' events)."""
    s = Setup(category, agent=True)
    try:
        room = await _room_with(s, orchestration=Pipeline(agents=["assistant"]))
    except Exception as exc:  # the probe reports what the caller meets
        out = {"room setup": f"{type(exc).__name__}: {exc}"}
        await s.kit.close()
        return out
    before = len(s.ai.calls)
    await s.host_says(room, "HOST unaddressed")
    unaddressed = len(s.ai.calls) - before
    await s.host_says(room, "HOST to assistant", addressed_to=["assistant"])
    out = {
        "agent turns on host unaddressed": unaddressed,
        "agent turns on host addressed": len(s.ai.calls) - before - unaddressed,
        "SMS sent": s.texted(),
    }
    await s.kit.close()
    return out


async def scenario_deliver(category: ChannelCategory) -> dict[str, Any]:
    """S9: kit.deliver() with no channel named, the host attached."""
    s = Setup(category, agent=True)
    room = await _room_with(s)
    await s.kit.deliver(room, "Remind the customer their cart expires.")
    await asyncio.sleep(SETTLE_S)
    events = await s.kit.store.list_events(room)
    sources = [(e.source.channel_id, str(getattr(e.content, "body", ""))[:30]) for e in events]
    out = {"events (source, text)": sources, "SMS sent": s.texted()}
    await s.kit.close()
    return out


async def scenario_discussion(category: ChannelCategory) -> dict[str, Any]:
    """S6: a room run by a Discussion strategy."""
    s = Setup(category, agent=True)
    await s.kit.create_room(room_id="r", orchestration=Discussion(agents=["assistant"]))
    await s.kit.attach_channel("r", "sms", metadata={"phone_number": CUSTOMER})
    try:
        await s.kit.attach_channel("r", "host")
        attached = "ok"
    except Exception as exc:  # the probe reports what the caller meets
        attached = f"{type(exc).__name__}: {exc}"
    out = {"attach to a discussion room": attached}
    if attached == "ok":
        await s.host_says("r", "HOST in discussion", addressed_to=[])
        out["SMS sent"] = s.texted()
    await s.kit.close()
    return out


async def scenario_overhead(category: ChannelCategory) -> dict[str, Any]:
    """S8: time of 50 customer messages, host attached, no agent."""
    s = Setup(category, agent=False)
    room = await s.open_room()
    start = time.perf_counter()
    for i in range(50):
        await s.kit.process_inbound(
            InboundMessage(
                channel_id="sms", sender_id=CUSTOMER, content=TextContent(body=f"m{i}")
            ),
            room_id=room,
        )
    elapsed = (time.perf_counter() - start) / 50 * 1000
    out = {"ms per inbound": round(elapsed, 2), "host.on_event calls": len(s.host.reacted)}
    await s.kit.close()
    return out


async def scenario_author(category: ChannelCategory) -> dict[str, Any]:
    """S10: the host speaks as a named participant; what the agent reads."""
    out: dict[str, Any] = {}
    for channel_type in (ChannelType.SYSTEM, ChannelType.WEBHOOK):
        s = Setup(category, agent=True, channel_type=channel_type)
        room = await _room_with(s)
        await s.kit.store.add_participant(
            Participant(id=CUSTOMER, room_id=room, channel_id="sms", display_name="Marie")
        )
        await s.kit.store.add_participant(
            Participant(id="desk", room_id=room, channel_id="host", display_name="Acme Support")
        )
        await s.host_says(room, "HOST your order shipped", addressed_to=[], participant_id="desk")
        await s.kit.process_inbound(
            InboundMessage(
                channel_id="sms", sender_id=CUSTOMER, content=TextContent(body="when?")
            ),
            room_id=room,
        )
        await asyncio.sleep(SETTLE_S)
        last = s.ai.calls[-1]
        out[f"type {channel_type.value}: agent reads"] = [
            str(m.content)[:90] for m in last.messages if m.role == "user"
        ]
        await s.kit.close()
    return out


SCENARIOS = [
    scenario_text_the_customer,
    scenario_agent_present,
    scenario_addressed_only,
    scenario_pipeline,
    scenario_deliver,
    scenario_discussion,
    scenario_overhead,
    scenario_author,
]


async def main() -> None:
    for scenario in SCENARIOS:
        print(f"\n## {scenario.__name__} — {scenario.__doc__}")
        for category in (ChannelCategory.INTELLIGENCE, ChannelCategory.TRANSPORT):
            try:
                result = await scenario(category)
            except Exception as exc:  # the probe reports what the caller meets
                result = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {category.value.upper()}")
            for key, value in result.items():
                print(f"    {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
