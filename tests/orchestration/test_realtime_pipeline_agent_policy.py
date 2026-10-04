"""The active agent of a realtime pipeline answers to its own tool policy
(RMK-427, RFC §19.5).

A tool its policy denies is neither declared to the session nor served,
refused with the policy's words, nor listed by Tool Search or the allowed
names; a handoff to another agent applies that agent's policy on every
session of the room, read for its participant; the gate judges a call by the
agent the room talks to, the one that serves it. An agent that carries skills
is refused at the install, before anything is installed: a realtime session
serves the channel's skills, never an agent's.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels._skill_constants import SKILLS_NO_SCRIPTS_NOTE
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.participant import Participant
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import get_conversation_state, set_conversation_state
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.context import current_tool_allowed_names
from roomkit.tools.policy import RoleOverride, ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_realtime_tool_policy import _accounts_skill
from tests.test_skills_integration import MockScriptExecutor

PARAMS = {"type": "object", "properties": {}}
TOOLS = [
    AITool(name="balance", description="Read a balance", parameters=PARAMS),
    AITool(name="wire_money", description="Move money", parameters=PARAMS),
]


ALLOWED: list[set[str] | None] = []


def _agent(agent_id: str, policy: ToolPolicy | None, skills: SkillRegistry | None = None) -> Agent:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ALLOWED.append(current_tool_allowed_names())
        return f"EXECUTED {name}"

    return Agent(
        agent_id,
        provider=MockAIProvider(responses=["x"]),
        tools=TOOLS,
        tool_handler=handler,
        tool_policy=policy,
        skills=skills,
    )


HANDOFF = [
    PipelineStage(phase="intake", agent_id="triage", next="handling"),
    PipelineStage(phase="handling", agent_id="teller"),
]


async def _pipeline(
    agents: list[Agent], stages: list[PipelineStage], *, role: str | None = None, **channel: Any
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rtv", provider=provider, transport=MockRealtimeTransport(), **channel
    )
    kit = RoomKit()
    kit.register_channel(channel)
    for agent in agents:
        kit.register_channel(agent)
    ConversationPipeline(stages=stages).install(kit, agents, voice_channel_id="rtv")
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rtv")
    if role is not None:
        await kit.store.add_participant(
            Participant(id="u", room_id="r1", channel_id="rtv", role=role)
        )
    session = await channel.start_session("r1", "u", "ws")
    return kit, channel, provider, session


async def _call(
    channel: RealtimeVoiceChannel,
    provider: MockRealtimeProvider,
    session: Any,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    before = len(provider.tool_results)
    await provider.simulate_tool_call(session, f"c-{name}-{before}", name, arguments or {})
    for _ in range(200):
        if len(provider.tool_results) > before:
            break
        await asyncio.sleep(0.01)
    return provider.tool_results[-1][2]


def _declared(provider: MockRealtimeProvider) -> set[str]:
    calls = [c for c in provider.calls if c.method in ("connect", "reconfigure")]
    return {t["name"] for t in calls[-1].args.get("tools") or [] if t.get("name")}


async def _hand_off(
    channel: RealtimeVoiceChannel, provider: MockRealtimeProvider, session: Any, target: str
) -> None:
    reconfigured = len([c for c in provider.calls if c.method == "reconfigure"])
    handoff = await _call(
        channel,
        provider,
        session,
        "handoff_conversation",
        {"target": target, "reason": "r", "summary": "s"},
    )
    assert json.loads(handoff)["accepted"] is True
    for _ in range(100):
        if len([c for c in provider.calls if c.method == "reconfigure"]) > reconfigured:
            return
        await asyncio.sleep(0.01)


async def test_the_active_agents_policy_holds_on_its_session() -> None:
    agent = _agent("teller", ToolPolicy(deny=["wire_money"]))
    kit, channel, provider, session = await _pipeline(
        [agent], [PipelineStage(phase="a", agent_id="teller")]
    )

    assert "wire_money" not in _declared(provider)
    assert "balance" in _declared(provider)
    refused = await _call(channel, provider, session, "wire_money")
    assert json.loads(refused)["error"] == (
        "Tool 'wire_money' is not permitted by the agent's tool policy."
    )
    assert await _call(channel, provider, session, "balance") == "EXECUTED balance"
    await kit.close()


async def test_a_handoff_applies_the_new_agents_policy() -> None:
    open_agent = _agent("triage", None)
    strict = _agent("teller", ToolPolicy(deny=["wire_money"]))
    kit, channel, provider, session = await _pipeline(
        [open_agent, strict],
        [
            PipelineStage(phase="intake", agent_id="triage", next="handling"),
            PipelineStage(phase="handling", agent_id="teller"),
        ],
    )
    assert "wire_money" in _declared(provider)

    await _hand_off(channel, provider, session, "teller")

    assert "wire_money" not in _declared(provider)
    refused = await _call(channel, provider, session, "wire_money")
    assert "not permitted" in json.loads(refused)["error"]
    await kit.close()


async def test_a_handoff_to_an_agent_without_a_policy_lifts_the_last_ones() -> None:
    strict = _agent("triage", ToolPolicy(deny=["wire_money"]))
    open_agent = _agent("teller", None)
    kit, channel, provider, session = await _pipeline([strict, open_agent], HANDOFF)
    assert "wire_money" not in _declared(provider)

    await _hand_off(channel, provider, session, "teller")

    assert "wire_money" in _declared(provider)
    assert await _call(channel, provider, session, "wire_money") == "EXECUTED wire_money"
    await kit.close()


async def test_the_gate_judges_a_call_by_the_agent_the_room_talks_to() -> None:
    """The room's agent changes before its sessions are reconfigured (a hook
    of the handoff still running): the agent that serves the call is the one
    whose policy judges it."""
    kit, channel, provider, session = await _pipeline(
        [_agent("triage", None), _agent("teller", ToolPolicy(deny=["wire_money"]))], HANDOFF
    )
    room = await kit.get_room("r1")
    state = get_conversation_state(room).model_copy(update={"active_agent_id": "teller"})
    await kit.store.update_room(set_conversation_state(room, state))

    assert "wire_money" in _declared(provider)
    refused = await _call(channel, provider, session, "wire_money")
    assert "not permitted" in json.loads(refused)["error"]
    await kit.close()


async def test_a_handoff_declares_the_new_agents_tools_for_the_participants_role() -> None:
    observers_denied = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["wire_money"])})
    kit, channel, provider, session = await _pipeline(
        [_agent("triage", None), _agent("teller", observers_denied)], HANDOFF, role="observer"
    )
    assert "wire_money" in _declared(provider)

    await _hand_off(channel, provider, session, "teller")

    assert "wire_money" not in _declared(provider)
    await kit.close()


async def test_a_role_changed_during_the_session_holds_at_the_gate() -> None:
    observers_denied = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["wire_money"])})
    kit, channel, provider, session = await _pipeline(
        [_agent("teller", observers_denied)],
        [PipelineStage(phase="a", agent_id="teller")],
        role="member",
    )
    await kit.store.update_participant(
        Participant(id="u", room_id="r1", channel_id="rtv", role="observer")
    )

    refused = await _call(channel, provider, session, "wire_money")
    assert "not permitted" in json.loads(refused)["error"]
    await kit.close()


async def test_tool_search_and_the_allowed_names_follow_the_agents_policy() -> None:
    agent = _agent("teller", ToolPolicy(deny=["wire_money"]))
    kit, channel, provider, session = await _pipeline(
        [agent], [PipelineStage(phase="a", agent_id="teller")], tool_search=True
    )

    listed = json.loads(await _call(channel, provider, session, "list_tools"))
    assert "wire_money" not in {t["name"] for t in listed["tools"]}
    assert "balance" in {t["name"] for t in listed["tools"]}
    found = json.loads(await _call(channel, provider, session, "find_tools", {"query": "money"}))
    assert "wire_money" not in json.dumps(found)
    ALLOWED.clear()
    await _call(channel, provider, session, "balance")
    [allowed] = ALLOWED
    assert allowed is not None and "wire_money" not in allowed and "balance" in allowed
    await kit.close()


async def test_the_skills_preamble_reads_the_agents_policy(tmp_path: Path) -> None:
    agent = _agent("teller", ToolPolicy(deny=["run_skill_script"]))
    kit, _, provider, _ = await _pipeline(
        [agent],
        [PipelineStage(phase="a", agent_id="teller")],
        skills=_accounts_skill(tmp_path),
        script_executor=MockScriptExecutor(),
    )

    connected = [c.args for c in provider.calls if c.method == "connect"][-1]
    assert SKILLS_NO_SCRIPTS_NOTE.strip() in (connected["system_prompt"] or "")
    assert "run_skill_script" not in _declared(provider)
    await kit.close()


async def test_an_ended_session_keeps_no_agent_policy() -> None:
    agent = _agent("teller", ToolPolicy(deny=["wire_money"]))
    kit, channel, _, session = await _pipeline(
        [agent], [PipelineStage(phase="a", agent_id="teller")]
    )
    assert session.id in channel._session_agent_policies

    await channel.end_session(session)
    assert channel._session_agent_policies == {}

    # A handoff reaching it after its end puts nothing back.
    await channel._use_agent_policy(session, ToolPolicy(deny=["balance"]))
    assert channel._session_agent_policies == {}
    await kit.close()


@pytest.mark.parametrize(("role", "denied"), [("observer", True), ("member", False)])
async def test_the_agents_role_overrides_read_the_participant(role: str, denied: bool) -> None:
    policy = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["wire_money"])})
    agent = _agent("teller", policy)
    kit, channel, provider, session = await _pipeline(
        [agent], [PipelineStage(phase="a", agent_id="teller")], role=role
    )

    assert ("wire_money" in _declared(provider)) is not denied
    await kit.close()


def test_an_agent_with_skills_is_refused_at_the_install(tmp_path: Path) -> None:
    folder = tmp_path / "s"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: s\ndescription: S\nallowed_tools: wire_money\n---\nBody.", encoding="utf-8"
    )
    skills = SkillRegistry()
    skills.discover(tmp_path)
    agent = _agent("teller", None, skills)
    kit = RoomKit()
    kit.register_channel(
        RealtimeVoiceChannel(
            "rtv", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )
    )
    kit.register_channel(agent)

    hooks = len(kit._hook_engine._global_hooks)

    with pytest.raises(ValueError, match="carries skills"):
        ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="teller")]).install(
            kit, [agent], voice_channel_id="rtv"
        )
    # Nothing was installed before the refusal.
    assert len(kit._hook_engine._global_hooks) == hooks


def test_an_agent_with_an_empty_skill_registry_is_refused() -> None:
    """A skill added to the registry after the install would open its gated
    tools in the session with nothing to gate them (RMK-482)."""
    agent = _agent("teller", None, SkillRegistry())
    kit = RoomKit()
    kit.register_channel(
        RealtimeVoiceChannel(
            "rtv", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )
    )
    kit.register_channel(agent)

    with pytest.raises(ValueError, match="carries skills"):
        ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="teller")]).install(
            kit, [agent], voice_channel_id="rtv"
        )
