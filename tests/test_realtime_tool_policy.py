"""A realtime channel's tool policy (RFC §12.4, §12.10.12; RMK-286).

A tool the policy denies is not declared to the session and is refused at the
gate, whichever entry brings the call: the provider, spoken text the channel
recovered, a reasoning backend, or a conference's provider. A role override
applies to the session's participant.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookTrigger,
    RoomContext,
    RoomKit,
    ToolCallEvent,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.participant import Participant
from roomkit.tools.policy import RoleOverride, ToolPolicy
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until


def _tool(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"The {name} tool",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
    }


TOOLS = [_tool("lookup_account"), _tool("delete_account")]
DENY_DELETE = ToolPolicy(deny=["delete_*"])


class _Calls:
    """The host's handler, and what ON_TOOL_CALL's observers saw."""

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.observed: list[ToolCallEvent] = []

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        self.ran.append(name)
        return '{"ok": true}'


async def _channel(
    calls: _Calls,
    *,
    policy: ToolPolicy = DENY_DELETE,
    role: str | None = None,
    backend: ReasoningBackend | None = None,
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, VoiceSession]:
    provider = MockRealtimeProvider(full_duplex=backend is not None)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=calls.handler,
        tool_policy=policy,
        reasoning_backend=backend,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    if role is not None:
        await kit.store.add_participant(
            Participant(id="u1", room_id="r1", channel_id="rt", role=role)
        )

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        calls.observed.append(event)

    session = await channel.start_session("r1", "u1", "ws")
    return kit, channel, provider, session


def _declared(provider: MockRealtimeProvider) -> set[str]:
    connected = next(c.args for c in provider.calls if c.method == "connect")
    return {tool["name"] for tool in connected["tools"] or []}


async def test_a_denied_tool_is_not_declared_and_a_forced_call_is_refused() -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls)

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await asyncio.sleep(0.05)

    assert _declared(provider) == {"lookup_account"}
    assert calls.ran == []
    assert "not permitted" in json.loads(provider.tool_results[0][2])["error"]
    assert [(e.name, e.is_error) for e in calls.observed] == [("delete_account", True)]
    await kit.close()


async def test_a_role_override_applies_to_the_session_participant() -> None:
    observer_only = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["delete_*"])})
    calls = _Calls()
    kit, _, provider, _ = await _channel(calls, policy=observer_only, role="observer")
    assert _declared(provider) == {"lookup_account"}
    await kit.close()

    kit, _, provider, _ = await _channel(_Calls(), policy=observer_only, role="member")
    assert _declared(provider) == {"lookup_account", "delete_account"}
    await kit.close()


async def test_a_recovered_spoken_call_is_refused() -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls)

    await provider.simulate_transcription(session, "call:delete_account{id:42}", "assistant")
    await asyncio.sleep(0.1)

    assert calls.ran == []
    assert any("not permitted" in text for _sid, text, _role in provider.injected_texts)
    await kit.close()


class _Backend(ReasoningBackend):
    """Calls one tool, then answers."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.results: list[str] = []

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        self.tools = [tool["name"] for tool in request.tools or []]
        assert request.execute_tool is not None
        self.results.append(await request.execute_tool("delete_account", {"id": "42"}))
        yield ReasoningOutput("done", is_final=True)


async def test_a_reasoning_backend_gets_the_admitted_tools_and_is_refused_the_rest() -> None:
    calls, backend = _Calls(), _Backend()
    kit, _, provider, session = await _channel(calls, backend=backend)

    await provider.simulate_delegation(session, "d1", "integrator")
    await asyncio.sleep(0.05)

    assert backend.tools == ["lookup_account"]
    assert calls.ran == []
    assert "not permitted" in json.loads(backend.results[0])["error"]
    await kit.close()


async def test_tool_search_never_names_a_denied_tool() -> None:
    calls = _Calls()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[*TOOLS, *(_tool(f"filler_{i}") for i in range(30))],
        tool_handler=calls.handler,
        tool_policy=DENY_DELETE,
        tool_search=True,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_tool_call(session, "c1", "find_tools", {"query": "account"})
    await asyncio.sleep(0.05)

    found = {match["name"] for match in json.loads(provider.tool_results[0][2])["matches"]}
    assert "lookup_account" in found
    assert "delete_account" not in found
    await kit.close()


async def test_a_conference_provider_obeys_its_policy() -> None:
    ran: list[str] = []

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "{}"

    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=TOOLS, tool_handler=handler, tool_policy=DENY_DELETE
        ),
    )
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results))

    assert _declared(provider) == {"lookup_account"}
    assert ran == []
    assert "not permitted" in json.loads(provider.tool_results[0][2])["error"]
    await kit.close()


async def test_an_empty_policy_declares_everything() -> None:
    kit, _, provider, _ = await _channel(_Calls(), policy=ToolPolicy())
    assert _declared(provider) == {"lookup_account", "delete_account"}
    await kit.close()
