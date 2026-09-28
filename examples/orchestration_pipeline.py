"""Multi-agent orchestration pipeline example.

Demonstrates a 3-agent pipeline where conversations flow from
triage -> handler -> resolver, with handoff between agents.

Uses the ``Pipeline`` orchestration strategy for zero-boilerplate
setup — agents, routing, handoff tools, and conversation state are
all wired automatically by ``create_room(orchestration=...)``.

Run with:
    uv run python examples/orchestration_pipeline.py
"""

from __future__ import annotations

import asyncio
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import Agent, InboundMessage, Pipeline, RoomKit, TextContent, WebSocketChannel
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
    # AI agents — each with a distinct mock response
    ai_triage = Agent(
        "agent-triage",
        provider=MockAIProvider(
            ai_responses=handing_off(
                "agent-handler",
                reason="Billing issue needs specialist",
                summary="User has a billing question about their account.",
                then="I'll transfer you to our specialist.",
            )
        ),
        role="Triage agent",
        description="Routes incoming requests to the right specialist",
        system_prompt="You triage incoming requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_handler = Agent(
        "agent-handler",
        provider=MockAIProvider(
            ai_responses=handing_off(
                "agent-resolver",
                reason="Invoice corrected, needs final confirmation",
                summary="Adjusted invoice #4521. User confirmed the new amount.",
                then="I corrected invoice #4521; our resolution team will confirm.",
            )
        ),
        role="Request handler",
        description="Handles and resolves customer requests",
        system_prompt="You handle requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_resolver = Agent(
        "agent-resolver",
        provider=MockAIProvider(responses=["All done! Issue resolved."]),
        role="Resolution specialist",
        description="Confirms resolution and closes requests",
        system_prompt="You resolve and close requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    # Pipeline strategy: agents are chained linearly (triage -> handler -> resolver).
    # Routing, handoff tools, and initial state are all wired automatically.
    kit = RoomKit(
        orchestration=Pipeline(agents=[ai_triage, ai_handler, ai_resolver]),
    )

    # Transport channel for user messages
    ws = WebSocketChannel("ws-user")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive, room_id="support-room")
    kit.register_channel(ws)

    # Create room — orchestration handles agent registration, attachment,
    # and conversation state initialization automatically.
    await kit.create_room(room_id="support-room")
    await kit.attach_channel("support-room", "ws-user")

    # --- Simulate conversation ------------------------------------------------

    print("=== Phase 1: Triage ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    print(f"  Phase: {state.phase}, Active agent: {state.active_agent_id}")

    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="I need help with my billing."),
        )
    )
    reply = find_reply(inbox, "agent-triage", mark)
    print(f"  Triage replied: {reply.content.body}")  # type: ignore[union-attr]

    # Triage's model called handoff_conversation during that turn.
    print("\n=== Handoff: triage -> handler ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    print(
        f"  State: phase={state.phase}, agent={state.active_agent_id}, "
        f"handoffs={state.handoff_count}"
    )

    # User message — now routes to handler, whose model hands off to resolver
    print("\n=== Phase 2: Handling ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="My last invoice looks wrong."),
        )
    )
    reply = find_reply(inbox, "agent-handler", mark)
    print(f"  Handler replied: {reply.content.body}")  # type: ignore[union-attr]

    print("\n=== Handoff: handler -> resolver ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    print(
        f"  State: phase={state.phase}, agent={state.active_agent_id}, "
        f"handoffs={state.handoff_count}"
    )

    # Final message — routes to resolver
    print("\n=== Phase 3: Resolution ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Looks good, thanks!"),
        )
    )
    reply = find_reply(inbox, "agent-resolver", mark)
    print(f"  Resolver replied: {reply.content.body}")  # type: ignore[union-attr]

    # Show full phase history
    print("\n=== Phase History ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    for t in state.phase_history:
        print(f"  {t.from_phase} -> {t.to_phase} ({t.reason})")

    # Cleanup
    for ch in [ai_triage, ai_handler, ai_resolver]:
        await ch.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
