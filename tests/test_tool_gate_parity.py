"""The gate and the serving of a tool call do the same thing on every door
(RMK-394): a tool Tool Search hides is validated whoever serves it, a refusal
is bounded as a result, a realtime skill tool runs in the tool call context,
and a realtime or conference handler's structured copy reaches ON_TOOL_CALL.
The external handler's order is in ``test_external_tool_handler``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.sandbox.executor import SandboxExecutor
from roomkit.sandbox.models import SandboxResult
from roomkit.skills.executor import ScriptExecutor, ScriptResult
from roomkit.skills.registry import SkillRegistry
from roomkit.tools import current_tool_call, current_tool_room_id
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_framework import SimpleChannel
from tests.test_tool_search_orchestration import CRM, _calling, _model, _orchestrated, _turn

_STRICT = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
    "additionalProperties": False,
}


def _tool_result(channel: AIChannel, first: int) -> str:
    """What the model read back for the first round's call."""
    messages = _model(channel).calls[first + 1].messages
    return next(str(p.result) for m in messages if m.role == "tool" for p in m.content)


# -- A tool Tool Search hides is validated whoever serves it -----------------


class _Sandbox(SandboxExecutor):
    def __init__(self) -> None:
        self.ran: list[tuple[str, Any]] = []

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.ran.append((command, arguments))
        return SandboxResult(exit_code=0, output="ran")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [{"name": "sandbox_bash", "description": "Run a command.", "parameters": _STRICT}]


@pytest.mark.parametrize("name", ["crm_strict", "sandbox_bash"])
async def test_a_hidden_tool_is_validated_whoever_serves_it(name: str) -> None:
    sandbox = _Sandbox()
    strict = AITool(name="crm_strict", description="Strict.", parameters=_STRICT)
    channel = _orchestrated(
        False, [*CRM, strict], _calling(name, cmd=42), tool_search=True, sandbox=sandbox
    )

    first = await _turn(channel, "r1")

    assert name not in {t.name for t in _model(channel).calls[first].tools}
    assert "missing required argument 'command'" in _tool_result(channel, first)
    assert sandbox.ran == []


async def test_a_hidden_human_input_tool_is_validated_before_a_person_is_asked() -> None:
    asked: list[dict[str, Any]] = []

    class _Spy(HumanInputHandler):
        async def create(
            self, tool_name: str, arguments: dict[str, Any], *a: Any, **k: Any
        ) -> Any:
            asked.append(arguments)
            return await super().create(tool_name, arguments, *a, **k)

    ask = AITool(
        name="AskUserQuestion",
        description="Ask the user.",
        parameters={
            **_STRICT,
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    )
    human = HumanInputToolHandler(
        tool_names={"AskUserQuestion"}, tool_definitions=[ask], handler=_Spy()
    )
    channel = _orchestrated(
        False,
        CRM,
        _calling("AskUserQuestion", bogus=1),
        tool_search=True,
        human_input_handler=human,
    )

    first = await asyncio.wait_for(_turn(channel, "r1"), 5)

    assert "missing required argument 'question'" in _tool_result(channel, first)
    assert asked == []


# -- A refusal is bounded as a result ----------------------------------------


async def test_a_gate_refusal_is_bounded_as_a_result() -> None:
    reason = "Denied by compliance: " + "policy clause. " * 20_000
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ],
    )

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tool_handler=lookup,
            tools=[AITool(name="lookup", description="Look up")],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
    async def deny(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult.block(reason)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    part = next(m for m in provider.calls[-1].messages if m.role == "tool").content[0]
    assert part.is_error
    assert len(str(part.result)) < 20_000
    await kit.close()


# -- A realtime skill tool runs in the tool call context ---------------------


class _RecordingExecutor(ScriptExecutor):
    def __init__(self) -> None:
        self.seen: list[tuple[str | None, str | None]] = []

    async def execute(self, skill: Any, script_name: str, arguments: Any = None) -> ScriptResult:
        call = current_tool_call()
        self.seen.append((current_tool_room_id(), call.tool_call_id if call else None))
        return ScriptResult(exit_code=0, stdout="hi", stderr="")


def _registry(tmp: Path) -> SkillRegistry:
    skill_dir = tmp / "s1"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: s1\ndescription: Test\n---\nBody", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.sh").write_text("echo hi\n", encoding="utf-8")
    registry = SkillRegistry()
    registry.discover(tmp)
    return registry


async def test_a_realtime_skill_script_runs_in_the_tool_call_context(tmp_path: Path) -> None:
    executor = _RecordingExecutor()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        skills=_registry(tmp_path),
        script_executor=executor,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")

    arguments = {"skill_name": "s1", "script_name": "hello.sh"}
    await provider.simulate_tool_call(session, "c1", "run_skill_script", arguments)
    await until(lambda: bool(provider.tool_results))

    assert executor.seen == [("r1", "c1")]
    await kit.close()


# -- A realtime or conference handler's structured copy reaches ON_TOOL_CALL -

_COPY = {"contact": "Jane Doe"}


def _publish_copy() -> str:
    call = current_tool_call()
    assert call is not None
    call.structured_content = dict(_COPY)
    return '{"contact": "Jane Doe"}'


def _watch(kit: RoomKit, replace_with: Any = ...) -> tuple[list[Any], list[Any]]:
    """What the SYNC chain and the observers see of the structured copy."""
    chain: list[Any] = []
    observed: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="see")
    async def see(event: ToolCallEvent, ctx: Any) -> HookResult:
        chain.append(event.structured_content)
        if replace_with is ...:
            return HookResult.allow()
        return HookResult(action="allow", metadata={"structured_content": replace_with})

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
    async def observe(event: ToolCallEvent, ctx: Any) -> None:
        observed.append(event.structured_content)

    return chain, observed


@pytest.mark.parametrize("replace_with", [..., {"contact": "J. D."}, None])
async def test_a_realtime_hook_sees_and_may_replace_the_structured_copy(replace_with: Any) -> None:
    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return _publish_copy()

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=lookup,
        tools=[{"name": "hr_lookup", "description": "HR", "parameters": {"type": "object"}}],
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    chain, observed = _watch(kit, replace_with)
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_tool_call(session, "c1", "hr_lookup", {})
    await until(lambda: bool(observed))

    assert chain == [_COPY]
    assert observed == [_COPY if replace_with is ... else replace_with]
    await kit.close()


async def test_a_conference_hook_sees_the_structured_copy() -> None:
    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return _publish_copy()

    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(
        provider=provider, tools=[{"name": "hr_lookup"}], tool_handler=handler
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    chain, observed = _watch(kit)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "c1", "hr_lookup", {})
    await until(lambda: bool(observed))

    assert chain == [_COPY]
    assert observed == [_COPY]
    await kit.close()
