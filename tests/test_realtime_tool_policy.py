"""A realtime channel's tool policy (RFC §12.4, §12.10.12; RMK-286).

A tool the policy denies is not declared to the session and is refused at the
gate, whichever entry brings the call: the provider, spoken text the channel
recovered, a reasoning backend, or a conference's provider. A role override
applies to the session's participant.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookTrigger,
    RoomContext,
    RoomKit,
    ToolCallEvent,
)
from roomkit.channels._skill_constants import SKILLS_NO_SCRIPTS_NOTE
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.participant import Participant
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import RoleOverride, ToolPolicy
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_skills_integration import MockScriptExecutor


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

    async def conference(self, room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return await self.handler(name, arguments)


async def _channel(
    calls: _Calls,
    *,
    policy: ToolPolicy = DENY_DELETE,
    role: str | None = None,
    backend: ReasoningBackend | None = None,
    skills: SkillRegistry | None = None,
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
        skills=skills,
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
    """What the session's last connection declared."""
    connected = [c.args for c in provider.calls if c.method == "connect"][-1]
    return {tool["name"] for tool in connected["tools"] or []}


async def test_a_denied_tool_is_not_declared_and_a_forced_call_is_refused() -> None:
    calls = _Calls()
    kit, _, provider, session = await _channel(calls)

    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results) and bool(calls.observed))

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
    await until(lambda: bool(provider.injected_texts) and bool(calls.observed))

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
    await until(lambda: bool(backend.results) and bool(calls.observed))

    assert backend.tools == ["lookup_account"]
    assert calls.ran == []
    assert "not permitted" in json.loads(backend.results[0])["error"]
    assert [(e.name, e.is_error) for e in calls.observed] == [("delete_account", True)]
    await kit.close()


async def _searching_channel(
    provider: MockRealtimeProvider, calls: _Calls, policy: ToolPolicy = DENY_DELETE
) -> tuple[RoomKit, RealtimeVoiceChannel, VoiceSession]:
    """A channel whose catalogue is large enough for Tool Search."""
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[*TOOLS, *(_tool(f"filler_{i}") for i in range(30))],
        tool_handler=calls.handler,
        tool_policy=policy,
        tool_search=True,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, channel, await channel.start_session("r1", "u1", "ws")


async def test_tool_search_never_names_a_denied_tool() -> None:
    provider = MockRealtimeProvider()
    kit, _, session = await _searching_channel(provider, _Calls())

    await provider.simulate_tool_call(session, "c1", "find_tools", {"query": "account"})
    await provider.simulate_tool_call(session, "c2", "list_tools", {})
    await until(lambda: len(provider.tool_results) == 2)

    results = {call_id: result for _sid, call_id, result in provider.tool_results}
    found = {match["name"] for match in json.loads(results["c1"])["matches"]}
    assert "lookup_account" in found
    assert "delete_account" not in found
    assert "lookup_account" in results["c2"]
    assert "delete_account" not in results["c2"]
    await kit.close()


async def test_a_reconfiguration_declares_no_denied_tool() -> None:
    kit, channel, provider, session = await _channel(_Calls())

    await channel.reconfigure_session(session, tools=TOOLS)

    assert _declared(provider) == {"lookup_account"}
    await kit.close()


def _accounts_skill(tmp_path: Path) -> SkillRegistry:
    """A skill that gates ``lookup_account`` until it is activated."""
    skill_dir = tmp_path / "accounts"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: accounts\ndescription: Accounts\n"
        "allowed_tools: lookup_account, delete_account\n---\nServe accounts.",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


async def test_a_skill_activation_reveals_no_denied_tool(tmp_path: Path) -> None:
    kit, _, provider, session = await _channel(_Calls(), skills=_accounts_skill(tmp_path))
    assert not {"lookup_account", "delete_account"} & _declared(provider)

    await provider.simulate_tool_call(session, "c1", "activate_skill", {"name": "accounts"})
    await until(lambda: bool(provider.tool_results) and "lookup_account" in _declared(provider))

    assert "delete_account" not in _declared(provider)
    await kit.close()


async def test_a_reasoning_backend_is_not_offered_a_gated_tool(tmp_path: Path) -> None:
    calls, backend = _Calls(), _Backend()
    kit, _, provider, session = await _channel(
        calls, backend=backend, skills=_accounts_skill(tmp_path)
    )

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: bool(backend.results))

    assert "lookup_account" not in backend.tools
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


async def test_a_role_changed_during_the_session_holds_at_the_next_call() -> None:
    observer_only = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["delete_*"])})
    calls = _Calls()
    kit, _, provider, session = await _channel(calls, policy=observer_only, role="member")
    assert "delete_account" in _declared(provider)

    demoted = await kit.store.get_participant("r1", "u1")
    assert demoted is not None
    await kit.store.update_participant(demoted.model_copy(update={"role": "observer"}))
    await provider.simulate_tool_call(session, "c1", "delete_account", {"id": "42"})
    await until(lambda: bool(provider.tool_results))

    assert calls.ran == []
    assert "not permitted" in json.loads(provider.tool_results[0][2])["error"]
    await kit.close()


class _FixedProvider(MockRealtimeProvider):
    """A provider whose declarations cannot change mid-session."""

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return False


async def test_a_whitelist_keeps_the_call_tool_transport_and_gates_what_it_names() -> None:
    calls, provider = _Calls(), _FixedProvider()
    kit, _, session = await _searching_channel(provider, calls, ToolPolicy(allow=["lookup_*"]))

    for call_id, name in (("c1", "delete_account"), ("c2", "lookup_account")):
        arguments = {"name": name, "arguments_json": '{"id": "42"}'}
        await provider.simulate_tool_call(session, call_id, "call_tool", arguments)
    await until(lambda: len(provider.tool_results) == 2)

    assert "call_tool" in _declared(provider)
    results = {call_id: result for _sid, call_id, result in provider.tool_results}
    assert "not permitted" in json.loads(results["c1"])["error"]
    assert calls.ran == ["lookup_account"]
    await kit.close()


def test_a_conference_config_keeps_its_positional_order() -> None:
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(provider, None, None, None, None, 0.7)  # type: ignore[misc]
    assert config.temperature == 0.7
    assert config.tool_policy is None


async def test_a_conference_policy_says_its_role_overrides_never_apply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = MockRealtimeProvider()
    policy = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["delete_*"])})
    kit, _, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider, tools=TOOLS, tool_handler=_Calls().conference, tool_policy=policy
        ),
    )
    assert "role_overrides ['observer'] never apply" in caplog.text
    await kit.close()


async def test_the_skills_preamble_says_scripts_cannot_run_when_the_policy_denies_them(
    tmp_path: Path,
) -> None:
    """As on AIChannel: the prompt never promises a tool the policy denies (RFC §21.1)."""

    async def prompt_under(policy: ToolPolicy | None) -> str:
        provider = MockRealtimeProvider()
        skill_root = tmp_path / ("open" if policy is None else "denied")
        skill_root.mkdir()
        channel = RealtimeVoiceChannel(
            "rt",
            provider=provider,
            transport=MockRealtimeTransport(),
            skills=_accounts_skill(skill_root),
            script_executor=MockScriptExecutor(),
            tool_policy=policy,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "rt")
        await channel.start_session("r1", "u1", "ws")
        await kit.close()
        connected = [c.args for c in provider.calls if c.method == "connect"][-1]
        return connected["system_prompt"] or ""

    note = SKILLS_NO_SCRIPTS_NOTE.strip()
    assert note in await prompt_under(ToolPolicy(deny=["run_skill_script"]))
    assert note not in await prompt_under(None)
