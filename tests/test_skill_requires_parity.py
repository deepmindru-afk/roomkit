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

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
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
# Only an unavailable skill gates it: nothing can open it, and the model is
# not told to activate anything.
CLOSED = "Tool 'gated_cal' is gated by a skill that is not available here."


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

    assert json.loads(result) == {"error": CLOSED}


def test_the_registry_keeps_an_unavailable_skill_s_gates(tmp_path: Path) -> None:
    skills = _registry(tmp_path, "allowed_tools: gated_cal")
    skills.mark_unavailable("cal", "not granted here")

    assert skills.gated_tool_names() == {"gated_cal"}
    assert skills.copy().gated_tool_names() == {"gated_cal"}
    assert skills.copy(marks=False).gated_tool_names() == set()


def _skills(tmp_path: Path, **skills: list[str]) -> SkillRegistry:
    for name, frontmatter in skills.items():
        folder = tmp_path / name
        folder.mkdir()
        head = "\n".join([f"name: {name}", f"description: {name} skill", *frontmatter])
        (folder / "SKILL.md").write_text(f"---\n{head}\n---\nBody of {name}.", encoding="utf-8")
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


async def _text_turn(skills: SkillRegistry, names: list[str], *calls: AIToolCall) -> list[str]:
    """What the model read for each call of a turn that makes *calls* in turn."""
    provider = MockAIProvider(
        ai_responses=[
            *(AIResponse(content="", finish_reason="tool_calls", tool_calls=[c]) for c in calls),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name=n, description=n, parameters=PARAMS) for n in names],
        tool_handler=_ok,
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
    return [
        str(part.result)
        for message in provider.calls[-1].messages
        if message.role == "tool"
        for part in message.content
    ]


async def _realtime_calls(
    skills: SkillRegistry, names: list[str], *calls: tuple[str, dict[str, Any]]
) -> list[str]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": n, "description": n, "parameters": PARAMS} for n in names],
        tool_handler=_ok,
        skills=skills,
        tool_search=False,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    for n, (name, arguments) in enumerate(calls):
        await provider.simulate_tool_call(session, f"c{n}", name, arguments)
        for _ in range(100):
            await asyncio.gather(*list(channel._scheduled_tasks), return_exceptions=True)
            if len(provider.tool_results) > n:
                break
            await asyncio.sleep(0.01)
    await kit.close()
    return [result[2] for result in provider.tool_results]


@pytest.mark.parametrize("door", ["text", "realtime"])
async def test_an_activated_skill_opens_a_gate_an_unavailable_one_shares(
    door: str, tmp_path: Path
) -> None:
    skills = _skills(tmp_path, a=["allowed_tools: gated_cal"], b=["allowed_tools: gated_cal"])
    skills.mark_unavailable("b", "not granted here")

    if door == "text":
        read = await _text_turn(
            skills,
            ["gated_cal"],
            AIToolCall(id="c1", name="activate_skill", arguments={"name": "a"}),
            AIToolCall(id="c2", name="gated_cal", arguments={}),
        )
    else:
        read = await _realtime_calls(
            skills, ["gated_cal"], ("activate_skill", {"name": "a"}), ("gated_cal", {})
        )

    assert read[-1] == "ok:gated_cal"


@pytest.mark.parametrize("door", ["text", "realtime"])
async def test_a_required_tool_only_a_closed_gate_holds_is_missing(
    door: str, tmp_path: Path
) -> None:
    skills = _skills(tmp_path, a=["requires: gated_cal"], b=["allowed_tools: gated_cal"])
    skills.mark_unavailable("b", "not granted here")
    activate = ("activate_skill", {"name": "a"})

    if door == "text":
        [read] = await _text_turn(
            skills, ["gated_cal"], AIToolCall(id="c1", name=activate[0], arguments=activate[1])
        )
    else:
        [read] = await _realtime_calls(skills, ["gated_cal"], activate)

    assert json.loads(read) == {"error": "Required tools not available: gated_cal"}


@pytest.mark.parametrize("door", ["text", "realtime"])
async def test_a_channel_tool_counts_as_a_requirement_on_both_doors(
    door: str, tmp_path: Path
) -> None:
    skills = _skills(tmp_path, a=["requires: read_skill_reference"])
    activate = ("activate_skill", {"name": "a"})

    if door == "text":
        [read] = await _text_turn(
            skills, ["lookup"], AIToolCall(id="c1", name=activate[0], arguments=activate[1])
        )
    else:
        [read] = await _realtime_calls(skills, ["lookup"], activate)

    assert "error" not in json.loads(read)


async def test_no_hook_sees_a_denied_tool_s_schema(tmp_path: Path) -> None:
    """Refused before the hooks run: the activation hands no SYNC hook the
    schema of a tool the policy denies."""
    skills = _registry(tmp_path, "requires: calendar")
    provider = _Fixed()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[
            {"name": n, "description": f"{n} schema", "parameters": PARAMS}
            for n in ("lookup", "calendar")
        ],
        tool_handler=_ok,
        tool_policy=ToolPolicy(deny=["calendar"]),
        skills=skills,
        skill_delivery_mode="on_demand",
    )
    kit = RoomKit()
    kit.register_channel(channel)
    seen: list[str] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="seen")
    async def on_call(event: Any, ctx: Any) -> HookResult:
        seen.append(str(event.result))
        return HookResult.allow()

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c1", "activate_skill", {"name": "cal"})
    for _ in range(100):
        await asyncio.gather(*list(channel._scheduled_tasks), return_exceptions=True)
        if provider.tool_results:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert not any("calendar schema" in result for result in seen)


def test_re_registering_an_unavailable_skill_reopens_its_gates_to_activation(
    tmp_path: Path,
) -> None:
    skills = _registry(tmp_path, "allowed_tools: gated_cal")
    skills.mark_unavailable("cal", "not granted here")
    skills.register(tmp_path / "cal")

    assert skills.closed_tool_names() == set()
    assert skills.gated_tool_names(activated={"cal"}) == set()


def test_a_subset_copy_keeps_only_its_skills_closed_gates(tmp_path: Path) -> None:
    skills = _skills(tmp_path, a=["allowed_tools: tool_a"], b=["allowed_tools: tool_b"])
    skills.mark_unavailable("a", "no")
    skills.mark_unavailable("b", "no")

    assert skills.copy(["a"]).closed_tool_names() == {"tool_a"}


def _hub(required: str, offered: Any) -> bool:
    """A host's reading of a hub name: served by any tool named ``<hub>_*``."""
    return any(name == required or name.startswith(f"{required}_") for name in offered)


def _hub_registry(tmp_path: Path) -> SkillRegistry:
    folder = tmp_path / "boards"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: boards\ndescription: boards skill\nrequires: boards\n---\nUse boards.",
        encoding="utf-8",
    )
    registry = SkillRegistry(requires_match=_hub)
    registry.discover(tmp_path)
    return registry


BOARDS = ["boards_get_card", "boards_create_card"]


@pytest.mark.parametrize("door", ["text", "realtime"])
async def test_a_host_matcher_reads_a_hub_requirement(door: str, tmp_path: Path) -> None:
    """A host that names a hub in ``requires`` says how it is served
    (RMK-429): the activation goes through on both doors."""
    skills = _hub_registry(tmp_path)
    activate = ("activate_skill", {"name": "boards"})

    if door == "text":
        [read] = await _text_turn(
            skills, BOARDS, AIToolCall(id="c1", name=activate[0], arguments=activate[1])
        )
    else:
        [read] = await _realtime_calls(skills, BOARDS, activate)

    assert "error" not in json.loads(read)


async def test_a_hub_activation_hands_over_the_tools_that_serve_it(tmp_path: Path) -> None:
    skills = _hub_registry(tmp_path)

    result = await _realtime_call(
        skills, ["lookup", *BOARDS], None, "activate_skill", {"name": "boards"}, provider=_Fixed()
    )

    served = sorted(tool["name"] for tool in json.loads(result)["required_tools"])
    assert served == sorted(BOARDS)


def test_a_copy_keeps_the_host_matcher(tmp_path: Path) -> None:
    assert _hub_registry(tmp_path).copy().requires_match is _hub
