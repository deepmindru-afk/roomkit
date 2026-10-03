"""The active agent of a realtime pipeline answers to its own tool policy
(RMK-427, RFC §19.5).

A tool its policy denies is neither declared to the session nor served,
refused with the policy's words; a handoff to another agent applies that
agent's policy; a role override of the agent's policy reads the session's
participant. An agent that carries skills is refused at the install: a
realtime session serves the channel's skills, never an agent's.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.participant import Participant
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import RoleOverride, ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

PARAMS = {"type": "object", "properties": {}}
TOOLS = [
    AITool(name="balance", description="Read a balance", parameters=PARAMS),
    AITool(name="wire_money", description="Move money", parameters=PARAMS),
]


def _agent(agent_id: str, policy: ToolPolicy | None, skills: SkillRegistry | None = None) -> Agent:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return f"EXECUTED {name}"

    return Agent(
        agent_id,
        provider=MockAIProvider(responses=["x"]),
        tools=TOOLS,
        tool_handler=handler,
        tool_policy=policy,
        skills=skills,
    )


async def _pipeline(
    agents: list[Agent], stages: list[PipelineStage], *, role: str | None = None
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel("rtv", provider=provider, transport=MockRealtimeTransport())
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

    handoff = await _call(
        channel,
        provider,
        session,
        "handoff_conversation",
        {"target": "teller", "reason": "r", "summary": "s"},
    )
    assert json.loads(handoff)["accepted"] is True
    for _ in range(100):
        if "wire_money" not in _declared(provider):
            break
        await asyncio.sleep(0.01)

    assert "wire_money" not in _declared(provider)
    refused = await _call(channel, provider, session, "wire_money")
    assert "not permitted" in json.loads(refused)["error"]
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

    with pytest.raises(ValueError, match="carry skills of their own"):
        ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="teller")]).install(
            kit, [agent], voice_channel_id="rtv"
        )
