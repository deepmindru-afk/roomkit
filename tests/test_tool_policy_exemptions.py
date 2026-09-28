"""The tool policy governs every tool the channel injects but five (RFC §21.1).

Only the tools that read or unlock and never act escape it, by exact name:
``activate_skill``, ``read_skill_reference``, ``read_stored_result``,
``find_tools`` and ``list_tools``. Sandbox commands, skill scripts and plan
updates are governed like a host tool, the declared list and the execution
guard agree, and Tool Search never names a tool the model may not call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.sandbox.executor import SandboxExecutor
from roomkit.sandbox.models import SandboxResult
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import ToolPolicy
from tests.conftest import make_event
from tests.test_skills_integration import MockScriptExecutor, _make_skill_dir_full
from tests.tool_loop_modes import LoopRun, respond


class _Sandbox(SandboxExecutor):
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.ran.append(command)
        return SandboxResult(exit_code=0, output="ran")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "sandbox_bash",
                "description": "Run a shell command in the sandbox.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ]


def _calls(*calls: tuple[str, dict[str, Any]]) -> list[AIResponse]:
    return [
        AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                AIToolCall(id=f"c{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(calls)
            ],
        ),
        AIResponse(content="done", finish_reason="stop"),
    ]


def _binding(tools: list[dict[str, Any]] | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": tools or []},
    )


async def _turn(ch: AIChannel, tools: list[dict[str, Any]] | None = None) -> LoopRun:
    return await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding(tools),
        RoomContext(room=Room(id="r1")),
    )


def _declared(context: AIContext) -> set[str]:
    return {tool.name for tool in context.tools or []}


def _tool_payload(context: AIContext, name: str) -> dict[str, Any]:
    part = next(
        part
        for message in context.messages
        if message.role == "tool"
        for part in message.content
        if part.name == name
    )
    return json.loads(part.result)


def _registry(tmp_path: Path, *, gating: str | None = None) -> SkillRegistry:
    _make_skill_dir_full(tmp_path, "ops", scripts=["deploy.sh"])
    if gating is not None:
        gated = tmp_path / "gatekeeper"
        gated.mkdir()
        (gated / "SKILL.md").write_text(
            "---\nname: gatekeeper\ndescription: Gates tools\n"
            f"allowed_tools: {gating}\n---\nBody.",
            encoding="utf-8",
        )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


async def test_deny_all_neither_declares_nor_runs_what_the_channel_injects(
    tmp_path: Path, streaming: bool
) -> None:
    sandbox = _Sandbox()
    scripts = MockScriptExecutor()
    provider = MockAIProvider(
        ai_responses=_calls(
            ("sandbox_bash", {"command": "rm -rf /data"}),
            ("run_skill_script", {"name": "ops", "script": "deploy.sh"}),
        ),
        streaming=streaming,
    )
    observed: list[ToolCallEvent] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        raise AssertionError(f"{name} reached the host handler")

    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=handler,
        tool_policy=ToolPolicy(deny=["*"]),
        sandbox=sandbox,
        skills=_registry(tmp_path),
        script_executor=scripts,
        tool_search=False,
    )

    async def observe(event: ToolCallEvent) -> None:
        observed.append(event)

    ch._tool_observer_hook = observe

    run = await _turn(ch)

    declared = _declared(provider.calls[0])
    assert "sandbox_bash" not in declared
    assert "run_skill_script" not in declared
    # The tools that only read or unlock stay reachable (RFC §21.1).
    assert {"activate_skill", "read_skill_reference"} <= declared
    assert sandbox.ran == []
    assert scripts.calls == []
    assert [call.failed for call in run.calls] == [True, True]
    assert {event.name for event in observed} == {"sandbox_bash", "run_skill_script"}
    assert all(event.is_error for event in observed)
    assert "not permitted" in _tool_payload(provider.calls[1], "sandbox_bash")["error"]


async def test_a_host_tool_that_looks_like_a_sandbox_tool_obeys_the_policy(
    streaming: bool,
) -> None:
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    provider = MockAIProvider(ai_responses=_calls(("sandbox_x", {})), streaming=streaming)
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=handler,
        tool_policy=ToolPolicy(deny=["sandbox_x"]),
        tools=[AITool(name="sandbox_x", description="A host tool", parameters={})],
    )

    run = await _turn(ch)

    assert "sandbox_x" not in _declared(provider.calls[0])
    assert ran == []
    assert run.calls[0].failed


async def test_a_skill_gating_sandbox_tools_hides_them_and_refuses_them(
    tmp_path: Path, streaming: bool
) -> None:
    """The listing and the guard read one rule: a gated sandbox tool is not
    offered, and a call to it is refused as gated, not run."""
    sandbox = _Sandbox()
    provider = MockAIProvider(
        ai_responses=_calls(("sandbox_bash", {"command": "ls"})), streaming=streaming
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=lambda name, arguments: "ok",
        sandbox=sandbox,
        skills=_registry(tmp_path, gating="sandbox_*"),
        tool_search=False,
    )

    await _turn(ch)

    assert "sandbox_bash" not in _declared(provider.calls[0])
    assert sandbox.ran == []
    assert "gated by a skill" in _tool_payload(provider.calls[1], "sandbox_bash")["error"]


async def test_tool_search_names_no_denied_or_gated_tool(tmp_path: Path, streaming: bool) -> None:
    catalogue = [
        {"name": "admin_delete_tenant", "description": "Delete a tenant and all its data."},
        {"name": "export_tenant", "description": "Export a tenant's data."},
        {"name": "search_docs", "description": "Search the tenant documentation."},
    ]
    provider = MockAIProvider(
        ai_responses=_calls(("list_tools", {}), ("find_tools", {"query": "tenant data"})),
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=lambda name, arguments: "ok",
        tool_policy=ToolPolicy(deny=["admin_*"]),
        skills=_registry(tmp_path, gating="export_*"),
        tool_search=True,
    )

    await _turn(ch, catalogue)

    listed = {tool["name"] for tool in _tool_payload(provider.calls[1], "list_tools")["tools"]}
    found = {match["name"] for match in _tool_payload(provider.calls[1], "find_tools")["matches"]}
    assert "search_docs" in listed
    assert "search_docs" in found
    for name in ("admin_delete_tenant", "export_tenant"):  # denied, gated
        assert name not in listed
        assert name not in found


async def test_a_skill_name_hint_offers_only_reachable_tools(
    tmp_path: Path, streaming: bool
) -> None:
    """A model mistaking a tool family for a skill is pointed at the tools it
    may call, never at one the policy denies."""
    catalogue = [
        {"name": "tenant_export", "description": "Export."},
        {"name": "tenant_delete", "description": "Delete."},
    ]
    provider = MockAIProvider(
        ai_responses=_calls(("activate_skill", {"name": "tenant"})), streaming=streaming
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=lambda name, arguments: "ok",
        tool_policy=ToolPolicy(deny=["tenant_delete"]),
        skills=_registry(tmp_path),
    )

    await _turn(ch, catalogue)

    hint = _tool_payload(provider.calls[1], "activate_skill")["tools_hint"]
    assert "tenant_export" in hint
    assert "tenant_delete" not in hint
