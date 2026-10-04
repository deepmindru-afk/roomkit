"""A skill's ``requires`` and gates read the same on every door (RMK-429,
RFC §24.3, §24.2).

An activation whose required tool the conversation does not offer, once its
tool policy is applied, is refused on a text turn as on a realtime session,
and never hands over the schema of a tool the policy denies. A skill marked
unavailable keeps what it gates closed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.tool_loop_modes import respond

PARAMS = {"type": "object", "properties": {}}
MISSING = "Required tools not available: calendar"
GATED = "Tool 'gated_cal' is gated by a skill. Activate the skill first using activate_skill."


class _Fixed(MockRealtimeProvider):
    """A provider whose declarations are fixed for the session."""

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return False

    @property
    def supports_context_preservation(self) -> bool:
        return True


def _registry(tmp_path: Path, *frontmatter: str) -> SkillRegistry:
    folder = tmp_path / "cal"
    folder.mkdir()
    head = "\n".join(["name: cal", "description: cal skill", *frontmatter])
    (folder / "SKILL.md").write_text(f"---\n{head}\n---\nUse calendar.", encoding="utf-8")
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


async def _ok(name: str, arguments: dict[str, Any]) -> str:
    return f"ok:{name}"


async def _text_call(
    skills: SkillRegistry, names: list[str], policy: ToolPolicy | None, call: AIToolCall
) -> str:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name=n, description=n, parameters=PARAMS) for n in names],
        tool_handler=_ok,
        tool_policy=policy,
        skills=skills,
        tool_search=False,
    )
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    await respond(
        channel, make_event(room_id="r1", body="go"), binding, RoomContext(room=Room(id="r1"))
    )
    [result] = [
        str(part.result)
        for message in provider.calls[-1].messages
        if message.role == "tool"
        for part in message.content
    ]
    return result


async def _realtime_call(
    skills: SkillRegistry,
    names: list[str],
    policy: ToolPolicy | None,
    name: str,
    arguments: dict[str, Any],
    *,
    provider: MockRealtimeProvider | None = None,
) -> str:
    provider = provider or MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": n, "description": f"{n} schema", "parameters": PARAMS} for n in names],
        tool_handler=_ok,
        tool_policy=policy,
        skills=skills,
        skill_delivery_mode="on_demand",
        tool_search=False,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c1", name, arguments)
    for _ in range(100):
        await asyncio.gather(*list(channel._scheduled_tasks), return_exceptions=True)
        if provider.tool_results:
            break
        await asyncio.sleep(0.01)
    await kit.close()
    return provider.tool_results[-1][2]


ACTIVATE = AIToolCall(id="c1", name="activate_skill", arguments={"name": "cal"})
CASES = {
    "absent": (["lookup"], None),
    "denied": (["lookup", "calendar"], ToolPolicy(deny=["calendar"])),
}


@pytest.mark.parametrize("case", CASES)
async def test_a_text_turn_refuses_an_activation_missing_its_tool(
    case: str, tmp_path: Path
) -> None:
    names, policy = CASES[case]
    skills = _registry(tmp_path, "requires: calendar")

    result = await _text_call(skills, names, policy, ACTIVATE)

    assert json.loads(result) == {"error": MISSING}


@pytest.mark.parametrize("case", CASES)
async def test_a_realtime_session_refuses_it_the_same_way(case: str, tmp_path: Path) -> None:
    names, policy = CASES[case]
    skills = _registry(tmp_path, "requires: calendar")

    result = await _realtime_call(skills, names, policy, "activate_skill", {"name": "cal"})

    assert json.loads(result) == {"error": MISSING}


async def test_a_denied_tool_s_schema_never_goes_out_with_an_activation(tmp_path: Path) -> None:
    skills = _registry(tmp_path, "requires: calendar")

    result = await _realtime_call(
        skills,
        ["lookup", "calendar"],
        ToolPolicy(deny=["calendar"]),
        "activate_skill",
        {"name": "cal"},
        provider=_Fixed(),
    )

    assert "calendar schema" not in result


async def test_an_offered_required_tool_lets_the_activation_through(tmp_path: Path) -> None:
    skills = _registry(tmp_path, "requires: calendar")

    result = await _text_call(skills, ["lookup", "calendar"], None, ACTIVATE)

    assert json.loads(result)["instructions"] == "Use calendar."


@pytest.mark.parametrize("door", ["text", "realtime"])
async def test_an_unavailable_skill_keeps_its_tools_closed(door: str, tmp_path: Path) -> None:
    skills = _registry(tmp_path, "allowed_tools: gated_cal")
    skills.mark_unavailable("cal", "not granted here")
    names = ["lookup", "gated_cal"]

    if door == "text":
        call = AIToolCall(id="c1", name="gated_cal", arguments={})
        result = await _text_call(skills, names, None, call)
    else:
        result = await _realtime_call(skills, names, None, "gated_cal", {})

    assert json.loads(result) == {"error": GATED}


def test_the_registry_keeps_an_unavailable_skill_s_gates(tmp_path: Path) -> None:
    skills = _registry(tmp_path, "allowed_tools: gated_cal")
    skills.mark_unavailable("cal", "not granted here")

    assert skills.gated_tool_names() == {"gated_cal"}
    assert skills.copy().gated_tool_names() == {"gated_cal"}
    assert skills.copy(marks=False).gated_tool_names() == set()
