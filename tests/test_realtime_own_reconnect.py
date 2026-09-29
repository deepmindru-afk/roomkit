"""A reconnect a tool call's own handler caused does not abandon that call (RMK-280).

Gemini Live applies a reconfiguration by reconnecting, and a reconnect orphans
every call the old socket issued, the one whose handler asked for it included.
That handler is not interrupted, its result stays off the wire (the new socket
never issued the id), and its outcome is reported as usual. Every other call
the reconnect orphaned is abandoned as before (RFC §9.3).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomContext, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import ToolRefusedError
from roomkit.models.tool_call import ToolCallEvent
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


class ReconnectingProvider(MockRealtimeProvider):
    """Orphans a session's outstanding calls on reconfigure, as Gemini Live does."""

    def __init__(self) -> None:
        super().__init__()
        self.outstanding: dict[str, set[str]] = {}
        self.reconfigured: list[str] = []

    async def reconfigure(self, session: VoiceSession, **kwargs: Any) -> None:
        self.reconfigured.append(session.id)
        orphaned = sorted(self.outstanding.pop(session.id, set()))
        await self.simulate_tool_call_cancellation(session, orphaned)

    async def simulate_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.outstanding.setdefault(session.id, set()).add(call_id)
        await super().simulate_tool_call(session, call_id, name, arguments)

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        self.outstanding.get(session.id, set()).discard(call_id)
        await super().submit_tool_result(session, call_id, result)

    def injected(self, session: VoiceSession) -> list[str]:
        return [
            c.args["text"]
            for c in self.calls
            if c.method == "inject_text" and c.args.get("session_id") == session.id
        ]


_TOOLS = [
    {"name": name, "description": "d", "parameters": {"type": "object", "properties": {}}}
    for name in ("switch_agent", "lookup")
]


async def _channel(
    provider: ReconnectingProvider, handler: Any
) -> tuple[RealtimeVoiceChannel, VoiceSession, list[ToolCallEvent]]:
    ch = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=_TOOLS,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(ch)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt")
    session = await ch.start_session(room.id, "u1", "ws")
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
    async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        observed.append(event)
        return HookResult.allow()

    return ch, session, observed


class TestTheCallWhoseHandlerReconnected:
    async def test_it_runs_to_its_end_and_is_reported_served(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}

        async def switch_agent(name: str, arguments: dict[str, Any]) -> str:
            ch, session = holder["ch"], holder["session"]
            await ch.reconfigure_session(session, system_prompt="You are the new agent.")
            await provider.inject_text(session, "Introduce yourself", role="system")
            return '{"accepted": true}'

        ch, session, observed = await _channel(provider, switch_agent)
        holder.update(ch=ch, session=session)

        await provider.simulate_tool_call(session, "h1", "switch_agent", {})
        await asyncio.sleep(0.1)

        assert provider.reconfigured == [session.id]
        assert provider.injected(session) == ["Introduce yourself"]
        # The new socket never issued h1: nothing goes out for it.
        assert provider.tool_results == []
        assert [(e.tool_call_id, e.cancelled, e.is_error) for e in observed] == [
            ("h1", False, False)
        ]
        assert json.loads(observed[0].result) == {"accepted": True}
        # No provider output is owed for a result that was not sent.
        assert session.id not in ch._awaiting_tool_response
        assert not ch._pending_tool_calls.get(session.id)

    async def test_a_refusal_after_the_reconnect_is_reported_refused(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}

        async def switch_agent(name: str, arguments: dict[str, Any]) -> str:
            await holder["ch"].reconfigure_session(holder["session"], system_prompt="New.")
            raise ToolRefusedError("The target agent is not available.")

        ch, session, observed = await _channel(provider, switch_agent)
        holder.update(ch=ch, session=session)

        await provider.simulate_tool_call(session, "h1", "switch_agent", {})
        await asyncio.sleep(0.1)

        assert provider.tool_results == []
        assert [(e.tool_call_id, e.cancelled, e.is_error) for e in observed] == [
            ("h1", False, True)
        ]


class TestTheOtherCallsTheReconnectOrphaned:
    async def test_a_call_pending_beside_it_is_still_abandoned(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}
        lookup_started = asyncio.Event()
        lookup_interrupted = asyncio.Event()

        async def handler(name: str, arguments: dict[str, Any]) -> str:
            if name == "lookup":
                lookup_started.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    lookup_interrupted.set()
                    raise
                return "too late"
            await holder["ch"].reconfigure_session(holder["session"], system_prompt="New.")
            return '{"accepted": true}'

        ch, session, observed = await _channel(provider, handler)
        holder.update(ch=ch, session=session)
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await asyncio.wait_for(lookup_started.wait(), 1)

        await provider.simulate_tool_call(session, "h1", "switch_agent", {})
        await asyncio.wait_for(lookup_interrupted.wait(), 1)
        await asyncio.sleep(0.1)

        assert provider.tool_results == []
        assert sorted((e.tool_call_id, e.cancelled) for e in observed) == [
            ("c1", True),
            ("h1", False),
        ]

    async def test_a_reconnect_from_elsewhere_abandons_the_call(self) -> None:
        provider = ReconnectingProvider()
        started = asyncio.Event()
        interrupted = asyncio.Event()

        async def lookup(name: str, arguments: dict[str, Any]) -> str:
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                interrupted.set()
                raise
            return "too late"

        ch, session, observed = await _channel(provider, lookup)
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await asyncio.wait_for(started.wait(), 1)

        # The application reconfigures the session, outside any handler
        await ch.reconfigure_session(session, system_prompt="New.")
        await asyncio.wait_for(interrupted.wait(), 1)
        await asyncio.sleep(0.05)

        assert provider.tool_results == []
        assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", True)]


class TestSpeechToSpeechHandoff:
    async def test_every_session_of_the_room_takes_the_new_agent(self) -> None:
        provider = ReconnectingProvider()
        ch = RealtimeVoiceChannel("rtv", provider=provider, transport=MockRealtimeTransport())
        kit = RoomKit()
        kit.register_channel(ch)
        triage = Agent("agent-triage", role="Triage", voice="v-t", system_prompt="Hi.")
        advisor = Agent("agent-advisor", role="Advisor", voice="v-a", system_prompt="Help.")
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
            ],
        )
        pipeline.install(kit, [triage, advisor], voice_channel_id="rtv", greet_on_handoff=True)
        room = await kit.create_room()
        state = ConversationState(active_agent_id="agent-triage", phase="triage")
        await kit.store.update_room(set_conversation_state(room, state))
        await kit.attach_channel(room.id, "rtv")
        caller = await ch.start_session(room.id, "u1", "ws")
        listener = await ch.start_session(room.id, "u2", "ws")
        observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
        async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            observed.append(event)
            return HookResult.allow()

        await provider.simulate_tool_call(
            caller,
            "h1",
            "handoff_conversation",
            {"target": "agent-advisor", "reason": "help", "summary": "ctx"},
        )
        await asyncio.sleep(0.2)

        assert sorted(provider.reconfigured) == sorted([caller.id, listener.id])
        assert len(provider.injected(caller)) == 1
        assert len(provider.injected(listener)) == 1
        assert [(e.tool_call_id, e.cancelled) for e in observed] == [("h1", False)]
        assert provider.tool_results == []
