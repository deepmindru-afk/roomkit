"""Swarm orchestration strategy example.

Demonstrates a swarm where every agent can hand off to every other
agent — no linear ordering. The entry agent handles initial messages,
and agents hand off freely as the conversation evolves.

Uses the ``Swarm`` orchestration strategy for automatic bidirectional
handoff wiring.

Run with:
    uv run python examples/orchestration_swarm.py
"""

from __future__ import annotations

import asyncio
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import Agent, InboundMessage, RoomKit, Swarm, TextContent, WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.event import RoomEvent
from roomkit.orchestration.handoff import HandoffMemoryProvider
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

# --- Helpers -----------------------------------------------------------------


def find_reply(events: list[RoomEvent], agent_id: str, start: int = 0) -> RoomEvent | None:
    """Find the first event from a specific agent after `start` index."""
    for event in events[start:]:
        if event.source.channel_id == agent_id:
            return event
    return None


def handing_off(target: str, reason: str, summary: str, then: str) -> list[AIResponse]:
    """A scripted model turn: call ``handoff_conversation``, then say *then*.

    A real model decides to hand off; the mock is told to. Either way the call
    goes through the agent's tool loop, which is what tells the handoff tool
    which room it acts on.
    """
    arguments = {"target": target, "reason": reason, "summary": summary}
    return [
        AIResponse(
            content="",
            tool_calls=[
                AIToolCall(id=f"to-{target}", name="handoff_conversation", arguments=arguments)
            ],
        ),
        AIResponse(content=then),
    ]


# --- Main --------------------------------------------------------------------


async def main() -> None:
    # Three specialist agents — any can hand off to any other
    ai_sales = Agent(
        "agent-sales",
        provider=MockAIProvider(
            ai_responses=[
                AIResponse(content="Great choice! Let me help with pricing."),
                *handing_off(
                    "agent-support",
                    reason="User also has a technical issue",
                    summary="User wants Pro plan but has a setup problem.",
                    then="Let me bring in support for that.",
                ),
                AIResponse(content="We offer SSO and audit-log add-ons."),
            ]
        ),
        role="Sales agent",
        description="Handles product inquiries and pricing",
        system_prompt="You are a sales agent.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_support = Agent(
        "agent-support",
        provider=MockAIProvider(
            ai_responses=handing_off(
                "agent-billing",
                reason="API key issue was billing-related (expired trial)",
                summary="User's trial expired. Needs Pro plan activation.",
                then="Your trial expired; billing will activate your plan.",
            )
        ),
        role="Support agent",
        description="Handles technical issues and troubleshooting",
        system_prompt="You handle support requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_billing = Agent(
        "agent-billing",
        provider=MockAIProvider(
            ai_responses=handing_off(
                "agent-sales",
                reason="Plan activated, back to sales for upsell",
                summary="Pro plan active. User may want add-ons.",
                then="Your Pro plan is active.",
            )
        ),
        role="Billing agent",
        description="Handles billing, invoices, and payment issues",
        system_prompt="You handle billing questions.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    # Swarm strategy: all agents can hand off to each other.
    # Sales is the entry point — handles initial messages.
    kit = RoomKit(
        orchestration=Swarm(
            agents=[ai_sales, ai_support, ai_billing],
            entry="agent-sales",
        ),
    )

    # Transport channel
    ws = WebSocketChannel("ws-user")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive, room_id="swarm-room")
    kit.register_channel(ws)

    # Create room — Swarm wires bidirectional handoff automatically
    await kit.create_room(room_id="swarm-room")
    await kit.attach_channel("swarm-room", "ws-user")

    # --- Simulate conversation ------------------------------------------------
    # Each agent's model hands off by calling handoff_conversation in its turn;
    # the next message goes to whoever it handed off to.

    async def say(body: str, expected_agent: str) -> None:
        mark = len(inbox)
        await kit.process_inbound(
            InboundMessage(channel_id="ws-user", sender_id="user", content=TextContent(body=body))
        )
        reply = find_reply(inbox, expected_agent, mark)
        print(f"  User: {body}")
        print(f"  {expected_agent}: {reply.content.body}")  # type: ignore[union-attr]
        state = get_conversation_state(await kit.get_room("swarm-room"))
        print(f"  Active agent now: {state.active_agent_id}")

    # 1. Initial message → sales (entry agent)
    print("=== Sales handles initial message ===")
    await say("Hi, I want to buy the Pro plan.", "agent-sales")

    # 2. Sales → Support (bidirectional handoff)
    print("\n=== Handoff: sales -> support ===")
    await say("My API key doesn't work, though.", "agent-sales")

    # 3. Support → Billing (support can reach billing directly)
    print("\n=== Handoff: support -> billing ===")
    await say("It stopped working this morning.", "agent-support")

    # 4. Billing → Sales (back to sales — bidirectional!)
    print("\n=== Handoff: billing -> sales (back!) ===")
    await say("Can you activate my Pro plan?", "agent-billing")

    print("\n=== Sales again ===")
    await say("What add-ons do you have?", "agent-sales")

    # --- Results -------------------------------------------------------------

    print("\n=== Conversation State ===")
    room = await kit.get_room("swarm-room")
    state = get_conversation_state(room)
    print(f"  Active agent: {state.active_agent_id}")
    print(f"  Handoff count: {state.handoff_count}")

    print("\n=== Handoff History ===")
    for t in state.phase_history:
        print(f"  {t.from_agent} -> {t.to_agent} ({t.reason})")

    # Show each agent's handoff tool targets
    print("\n=== Handoff Tool Targets ===")
    for agent in [ai_sales, ai_support, ai_billing]:
        tool = next(
            (t for t in agent._injected_tools if t.name == "handoff_conversation"),
            None,
        )
        if tool:
            targets = tool.parameters["properties"]["target"].get("enum", [])
            print(f"  {agent.channel_id} can reach: {targets}")

    # Cleanup
    for ch in [ai_sales, ai_support, ai_billing]:
        await ch.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
