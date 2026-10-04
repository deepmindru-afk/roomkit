"""One refusal per cause, on every door (RMK-428, RFC §21.1, §12.4).

A tool the policy denies, a tool a skill keeps closed and a name nothing
carries read the same text on an AI channel's turn, a realtime session, a
conference and a reasoning backend's turn; and access is checked before the
arguments, so a refused tool never names what its schema requires.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.agent import Agent
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
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.conftest import make_event
from tests.tool_loop_modes import respond

POLICY = ToolPolicy(deny=["secret", "wire_money"])
WIRE_PARAMS = {
    "type": "object",
    "properties": {"iban": {"type": "string"}},
    "required": ["iban"],
}
SCHEMAS = {
    "lookup": {"type": "object", "properties": {}},
    "secret": {"type": "object", "properties": {}},
    "gated_cal": {"type": "object", "properties": {}},
    "wire_money": WIRE_PARAMS,
}
CAUSES = {
    "secret": "Tool 'secret' is not permitted by the agent's tool policy.",
    "gated_cal": (
        "Tool 'gated_cal' is gated by a skill. Activate the skill first using activate_skill."
    ),
    "nope": "Tool 'nope' is not declared.",
}


def _dicts() -> list[dict[str, Any]]:
    return [{"name": n, "description": n, "parameters": p} for n, p in SCHEMAS.items()]


def _skills(tmp_path: Path) -> SkillRegistry:
    folder = tmp_path / "cal"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: cal\ndescription: cal skill\nallowed_tools: gated_cal\n---\nBody.",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


async def _ok(name: str, arguments: dict[str, Any]) -> str:
    return f"ok:{name}"


def _calling(*names: str, arguments: dict[str, Any] | None = None) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"c-{n}", name=n, arguments=arguments or {}) for n in names],
    )


def _results(provider: MockAIProvider) -> dict[str, str]:
    """What the model read for each call, by tool name."""
    return {
        part.name: json.loads(str(part.result))["error"]
        for message in provider.calls[-1].messages
        if message.role == "tool"
        for part in message.content
    }


async def _text_door(tmp_path: Path, *names: str, arguments: Any = None) -> dict[str, str]:
    provider = MockAIProvider(
        ai_responses=[_calling(*names, arguments=arguments), AIResponse(content="done")]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name=n, description=n, parameters=p) for n, p in SCHEMAS.items()],
        tool_handler=_ok,
        tool_policy=POLICY,
        skills=_skills(tmp_path),
        tool_search=False,
    )
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    event = make_event(room_id="r1", body="go", channel_id="sms1")
    await respond(channel, event, binding, RoomContext(room=Room(id="r1")))
    return _results(provider)


async def _realtime(
    tmp_path: Path, backend: AgentReasoningBackend | None = None
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider(full_duplex=backend is not None)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=_dicts(),
        tool_handler=_ok,
        tool_policy=POLICY,
        skills=_skills(tmp_path),
        tool_search=False,
        reasoning_backend=backend,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    return kit, channel, provider, session


async def _realtime_call(
    channel: RealtimeVoiceChannel,
    provider: MockRealtimeProvider,
    session: Any,
    name: str,
    args: Any,
) -> str:
    before = len(provider.tool_results)
    await provider.simulate_tool_call(session, f"c-{name}-{before}", name, args)
    for _ in range(100):
        await asyncio.gather(*list(channel._scheduled_tasks), return_exceptions=True)
        if len(provider.tool_results) > before:
            break
        await asyncio.sleep(0.01)
    return json.loads(provider.tool_results[-1][2])["error"]


async def test_a_text_turn_refuses_with_each_cause(tmp_path: Path) -> None:
    assert await _text_door(tmp_path, *CAUSES) == CAUSES


async def test_a_realtime_session_refuses_with_each_cause(tmp_path: Path) -> None:
    kit, channel, provider, session = await _realtime(tmp_path)

    read = {n: await _realtime_call(channel, provider, session, n, {}) for n in CAUSES}

    assert read == CAUSES
    await kit.close()


async def test_a_reasoning_backend_refuses_with_the_session_s_cause(tmp_path: Path) -> None:
    model = MockAIProvider(ai_responses=[_calling(*CAUSES), AIResponse(content="done")])
    backend = AgentReasoningBackend(Agent("reasoner", provider=model))
    kit, _, provider, session = await _realtime(tmp_path, backend)

    await provider.simulate_delegation(session, "d1", "integrator")
    for _ in range(200):
        if len(model.calls) >= 2:
            break
        await asyncio.sleep(0.01)

    # Offered only what the session may call, and refused in its words.
    assert "secret" not in {t.name for t in model.calls[0].tools}
    assert _results(model) == CAUSES
    await kit.close()


async def test_a_conference_refuses_with_each_cause() -> None:
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(
        provider=provider, tools=_dicts(), tool_handler=lambda *a: "ok", tool_policy=POLICY
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)

    for n, name in enumerate(("secret", "nope")):
        await provider.simulate_tool_call(session, f"c{n}", name, {})
        await until(lambda n=n: len(provider.tool_results) > n)

    read = [json.loads(result[2])["error"] for result in provider.tool_results]
    assert read == [CAUSES["secret"], CAUSES["nope"]]
    await kit.close()


@pytest.mark.parametrize("door", ["text", "realtime", "conference"])
async def test_a_denied_tool_is_refused_before_its_arguments(door: str, tmp_path: Path) -> None:
    """wire_money({}) misses its required iban: the refusal is the policy's,
    never "missing required argument 'iban'", which names its schema."""
    denied = "Tool 'wire_money' is not permitted by the agent's tool policy."
    if door == "text":
        assert (await _text_door(tmp_path, "wire_money"))["wire_money"] == denied
        return
    if door == "realtime":
        kit, channel, provider, session = await _realtime(tmp_path)
        assert await _realtime_call(channel, provider, session, "wire_money", {}) == denied
        await kit.close()
        return
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(
        provider=provider, tools=_dicts(), tool_handler=lambda *a: "ok", tool_policy=POLICY
    )
    kit, conference, _, _ = await realtime_kit(provider=provider, config=config)
    session = await conference._realtime.ensure_session(ROOM)
    await provider.simulate_tool_call(session, "c1", "wire_money", {})
    await until(lambda: bool(provider.tool_results))
    assert json.loads(provider.tool_results[-1][2])["error"] == denied
    await kit.close()
