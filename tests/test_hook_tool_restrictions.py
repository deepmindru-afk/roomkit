"""What a hook takes away from a turn's tools stays taken away (RMK-272).

A BEFORE_AI_GENERATION hook that withdraws a tool withdraws it from every
round of the turn, not only the first; one that adds a tool keeps it. An
ON_TOOL_CALL hook that blocks ``activate_skill`` blocks the activation, not
only the answer the model reads: the skill's gated tools stay closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent, ToolCallEvent, ToolCallVerdict
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_READ = AITool(name="safe_read", description="Read a record", parameters={})
_DELETE = AITool(name="delete_account", description="Delete the account", parameters={})
_EXPORT = AITool(name="export_report", description="Export a report", parameters={})
_WIRE = AITool(name="wire_money", description="Wire money", parameters={})


def _round(call_id: str, name: str, arguments: dict[str, Any] | None = None) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments or {})],
    )


_DONE = AIResponse(content="done", finish_reason="stop")


class _Recorder:
    def __init__(self) -> None:
        self.ran: list[str] = []
        self.observed: list[ToolCallEvent] = []

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        self.ran.append(name)
        return "ok"

    async def observe(self, event: ToolCallEvent) -> None:
        self.observed.append(event)


def _declared(context: AIContext) -> set[str]:
    return {tool.name for tool in context.tools or []}


async def _turn(ch: AIChannel) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    return await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )


def _read_only(gen_event: AIGenerationEvent) -> None:
    """A hook that withdraws the destructive tool and offers an export."""
    tools = [t for t in gen_event.ai_context.tools if t.name != "delete_account"]
    gen_event.ai_context = gen_event.ai_context.model_copy(update={"tools": [*tools, _EXPORT]})


async def test_a_tool_the_hook_withdrew_is_declared_at_no_round_and_refused(
    streaming: bool,
) -> None:
    provider = MockAIProvider(
        ai_responses=[_round("c0", "safe_read"), _round("c1", "delete_account"), _DONE],
        streaming=streaming,
    )
    calls = _Recorder()
    ch = AIChannel("ai1", provider=provider, tool_handler=calls.handler, tools=[_READ, _DELETE])
    ch._tool_observer_hook = calls.observe

    async def hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
        _read_only(gen_event)
        return SyncPipelineResult(allowed=True)

    ch._before_generation_hook = hook

    run = await _turn(ch)

    assert all("delete_account" not in _declared(call) for call in provider.calls)
    assert calls.ran == ["safe_read"]
    assert run.calls[1].name == "delete_account" and run.calls[1].failed
    assert [event.name for event in calls.observed] == ["delete_account"]


async def test_a_tool_the_hook_added_stays_declared(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[_round("c0", "safe_read"), _round("c1", "export_report"), _DONE],
        streaming=streaming,
    )
    calls = _Recorder()
    ch = AIChannel("ai1", provider=provider, tool_handler=calls.handler, tools=[_READ, _DELETE])

    async def hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
        _read_only(gen_event)
        return SyncPipelineResult(allowed=True)

    ch._before_generation_hook = hook

    await _turn(ch)

    assert all("export_report" in _declared(call) for call in provider.calls)
    assert calls.ran == ["safe_read", "export_report"]


def _payments_skill(tmp_path: Path) -> SkillRegistry:
    skill_dir = tmp_path / "payments"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: payments\ndescription: Payments\nallowed_tools: wire_money\n---\nPay.",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


async def test_a_blocked_activation_opens_no_gate(tmp_path: Path, streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "payments"}),
            _round("c1", "wire_money"),
            _DONE,
        ],
        streaming=streaming,
    )
    calls = _Recorder()
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=calls.handler,
        tools=[_WIRE],
        skills=_payments_skill(tmp_path),
    )

    async def refuse_activation(event: ToolCallEvent) -> ToolCallVerdict | None:
        if event.name == "activate_skill":
            return ToolCallVerdict(result="skill not allowed for this user", blocked=True)
        return None

    ch._tool_call_hook = refuse_activation

    run = await _turn(ch)

    assert run.calls[0].failed
    assert "wire_money" not in _declared(provider.calls[1])
    assert calls.ran == []
    assert run.calls[1].failed
    # Neither for the turn nor for the conversation.
    assert not ch._skill_activation.is_active("r1", "payments")


async def test_a_served_activation_still_opens_its_gate(tmp_path: Path, streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "payments"}),
            _round("c1", "wire_money"),
            _DONE,
        ],
        streaming=streaming,
    )
    calls = _Recorder()
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=calls.handler,
        tools=[_WIRE],
        skills=_payments_skill(tmp_path),
    )

    await _turn(ch)

    assert "wire_money" in _declared(provider.calls[1])
    assert calls.ran == ["wire_money"]
    assert ch._skill_activation.is_active("r1", "payments")
