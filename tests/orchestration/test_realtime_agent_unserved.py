"""An agent that carries what a realtime session never serves for it is
refused, each cause named, by one rule: at a realtime pipeline's install, and
as a reasoning backend's agent (RMK-482, RFC §12.4.1, §19.5).

The pipeline's refusal points to the voice channel's own option where it has
one, before anything is installed; an agent's host tools stay served by the
pipeline. The reasoning backend refuses host tools of its own besides.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.models import Skill, SkillMetadata
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend


def _skills() -> SkillRegistry:
    registry = SkillRegistry()
    registry.add(Skill(SkillMetadata(name="s", description="S"), "Body.", Path(".")))
    return registry


def _human() -> HumanInputToolHandler:
    ask = AITool(name="ask_user", description="Ask", parameters={"type": "object"})
    return HumanInputToolHandler({"ask_user"}, tool_definitions=[ask])


CAUSES: dict[str, tuple[Callable[[], dict[str, Any]], str, str]] = {
    "skills": (lambda: {"skills": _skills()}, "skills", "skills=..."),
    "human-input": (
        lambda: {"human_input_handler": _human()},
        "a human-input handler",
        "human_input_handler=...",
    ),
    "planning": (lambda: {"enable_planning": True}, "planning", "never serves an agent's"),
    "sandbox": (lambda: {"sandbox": MagicMock()}, "a sandbox", "never serves an agent's"),
    "external": (
        lambda: {"external_tool_handler": MagicMock()},
        "an external tool handler",
        "never serves an agent's",
    ),
}
EVERY_CAUSE = pytest.mark.parametrize("cause", list(CAUSES))

BALANCE = AITool(name="balance", description="Balance", parameters={"type": "object"})


def _voice_kit(*agents: Agent) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(
        RealtimeVoiceChannel(
            "rtv", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )
    )
    for agent in agents:
        kit.register_channel(agent)
    return kit


def _install(kit: RoomKit, agents: list[Agent]) -> None:
    stages = [PipelineStage(phase=agent.channel_id, agent_id=agent.channel_id) for agent in agents]
    ConversationPipeline(stages=stages).install(kit, agents, voice_channel_id="rtv")


@EVERY_CAUSE
def test_a_pipeline_agent_carrying_what_a_session_never_serves_is_refused(cause: str) -> None:
    own, named, instead = CAUSES[cause]
    agent = Agent("teller", provider=MockAIProvider(), **own())
    kit = _voice_kit(agent)
    hooks = len(kit._hook_engine._global_hooks)

    with pytest.raises(ValueError) as refused:
        _install(kit, [agent])

    assert f"Agent 'teller' carries {named}," in str(refused.value)
    assert instead in str(refused.value)
    assert len(kit._hook_engine._global_hooks) == hooks


@EVERY_CAUSE
def test_a_reasoning_backend_s_agent_is_refused_for_the_same_cause(cause: str) -> None:
    own, named, _ = CAUSES[cause]

    with pytest.raises(ValueError, match=named):
        AgentReasoningBackend(Agent("reasoner", provider=MockAIProvider(), **own()))


def test_an_empty_skill_registry_is_refused_on_both_doors() -> None:
    """A skill added after the check would open its gated tools with nothing
    to gate them: in the backend's own loop, as in a pipeline's session."""
    with pytest.raises(ValueError, match="skills"):
        AgentReasoningBackend(Agent("reasoner", provider=MockAIProvider(), skills=SkillRegistry()))
    agent = Agent("teller", provider=MockAIProvider(), skills=SkillRegistry())

    with pytest.raises(ValueError, match="carries skills"):
        _install(_voice_kit(agent), [agent])


def test_each_cause_of_each_agent_is_named() -> None:
    teller = Agent("teller", provider=MockAIProvider(), enable_planning=True, skills=_skills())
    clerk = Agent("clerk", provider=MockAIProvider(), human_input_handler=_human())
    kit = _voice_kit(teller, clerk)

    with pytest.raises(ValueError) as refused:
        _install(kit, [teller, clerk])

    message = str(refused.value)
    assert "Agent 'teller' carries skills," in message
    assert "Agent 'teller' carries planning," in message
    assert "Agent 'clerk' carries a human-input handler," in message


def test_a_pipeline_agent_with_host_tools_only_is_installed() -> None:
    async def balance(name: str, arguments: dict[str, Any]) -> str:
        return "12 EUR"

    agent = Agent("teller", provider=MockAIProvider(), tools=[BALANCE], tool_handler=balance)

    _install(_voice_kit(agent), [agent])
