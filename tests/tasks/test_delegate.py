"""Tests for delegation tool, handler, and setup_delegation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import (
    RealtimeVoiceChannel,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import (
    DELEGATE_TOOL,
    DelegateHandler,
    build_delegate_tool,
    setup_delegation,
    setup_realtime_delegation,
)
from roomkit.tasks.models import DelegatedTask
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.tool_room import tool_call_in

# -- Tool definition ----------------------------------------------------------


class TestDelegateTool:
    def test_tool_definition(self):
        assert DELEGATE_TOOL.name == "delegate_task"
        props = DELEGATE_TOOL.parameters["properties"]
        assert "agent" in props
        assert "task" in props
        assert "context" in props
        assert "share_channels" in props
        assert DELEGATE_TOOL.parameters["required"] == ["agent", "task"]

    def test_build_delegate_tool_empty_targets(self):
        tool = build_delegate_tool([])
        assert tool is DELEGATE_TOOL

    def test_build_delegate_tool_with_targets(self):
        tool = build_delegate_tool(
            [
                ("pr-reviewer", "Reviews PRs"),
                ("code-writer", None),
            ]
        )
        assert tool.name == "delegate_task"
        agent_prop = tool.parameters["properties"]["agent"]
        assert agent_prop["enum"] == ["pr-reviewer", "code-writer"]
        assert "pr-reviewer: Reviews PRs" in tool.description
        assert "code-writer" in tool.description


# -- DelegateHandler ----------------------------------------------------------


class TestDelegateHandler:
    async def test_handle_calls_kit_delegate(self):
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="pr-reviewer",
            task="review PR",
        )
        kit.delegate = AsyncMock(return_value=task_handle)

        handler = DelegateHandler(kit, notify="voice-agent")
        result = await handler.handle(
            room_id="room-1",
            calling_agent_id="voice-agent",
            arguments={
                "agent": "pr-reviewer",
                "task": "review PR #42",
                "context": {"repo": "roomkit"},
                "share_channels": ["email-out"],
            },
        )

        assert result["status"] == "delegated"
        assert result["task_id"] == "t1"
        assert result["agent_id"] == "pr-reviewer"

        kit.delegate.assert_called_once_with(
            room_id="room-1",
            agent_id="pr-reviewer",
            task="review PR #42",
            context={"repo": "roomkit"},
            share_channels=["email-out"],
            notify="voice-agent",
        )

    async def test_handle_uses_default_share_channels(self):
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t2",
            child_room_id="child-2",
            parent_room_id="room-1",
            agent_id="agent-a",
            task="do stuff",
        )
        kit.delegate = AsyncMock(return_value=task_handle)

        handler = DelegateHandler(kit, default_share_channels=["email-out", "slack"])
        await handler.handle(
            room_id="room-1",
            calling_agent_id="agent-main",
            arguments={"agent": "agent-a", "task": "do stuff"},
        )

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["email-out", "slack"]


# -- setup_delegation ---------------------------------------------------------


class TestSetupDelegation:
    def test_declares_the_tool_and_serves_it(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        handler = DelegateHandler(kit)

        setup_delegation(channel, handler)

        tool_names = [t.name for t in channel.extra_tools]
        assert "delegate_task" in tool_names
        # The host's handler is left as it was.
        assert channel.tool_handler is None

    def test_double_setup_raises(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        handler = DelegateHandler(kit)

        setup_delegation(channel, handler)
        with pytest.raises(RuntimeError, match="already called"):
            setup_delegation(channel, handler)

    async def test_wrapped_handler_intercepts_delegate_task(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="pr-reviewer",
            task="review",
        )
        kit.delegate = AsyncMock(return_value=task_handle)

        handler = DelegateHandler(kit, notify="ai-main")
        setup_delegation(channel, handler)

        with tool_call_in("room-1"):
            result_str = await channel._channel_tool_handler(
                "delegate_task",
                {"agent": "pr-reviewer", "task": "review PR"},
            )

        result = json.loads(result_str)
        assert result["status"] == "delegated"
        assert result["task_id"] == "t1"

    async def test_wrapped_handler_no_room_id_returns_error(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        handler = DelegateHandler(kit)
        setup_delegation(channel, handler)

        # Called directly, outside any tool loop: no call names a room.
        result_str = await channel._channel_tool_handler(
            "delegate_task",
            {"agent": "a", "task": "b"},
        )

        result = json.loads(result_str)
        assert "error" in result

    async def test_wrapped_handler_passes_through_other_tools(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))

        called = []

        async def original_handler(name: str, arguments: dict) -> str:
            called.append(name)
            return json.dumps({"ok": True})

        channel.tool_handler = original_handler

        kit = MagicMock()
        handler = DelegateHandler(kit)
        setup_delegation(channel, handler)

        result_str = await channel._channel_tool_handler("some_other_tool", {})
        assert json.loads(result_str) == {"ok": True}
        assert called == ["some_other_tool"]


# -- setup_realtime_delegation ------------------------------------------------


def _rtv(tools: list[dict[str, Any]] | None = None, tool_handler: Any = None) -> Any:
    return RealtimeVoiceChannel(
        "rtv-main",
        provider=MockRealtimeProvider(),
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=tool_handler,
    )


def _declared(rtv: Any) -> list[str]:
    """What a session of room-1 on *rtv* declares."""
    session = VoiceSession(id="s1", room_id="room-1", participant_id="p", channel_id="rtv-main")
    return [t["name"] for t in rtv._compose_session_tools(session, rtv._tools) or []]


class TestSetupRealtimeDelegation:
    def test_declares_the_tool_and_leaves_the_host_handler(self):
        rtv = _rtv(tools=[{"name": "existing", "description": "test", "parameters": {}}])

        setup_realtime_delegation(rtv, DelegateHandler(MagicMock()))

        assert _declared(rtv) == ["existing", "delegate_task"]
        assert rtv._tools == [{"name": "existing", "description": "test", "parameters": {}}]
        assert rtv.tool_handler is None

    def test_double_setup_raises(self):
        rtv = _rtv()
        handler = DelegateHandler(MagicMock())

        setup_realtime_delegation(rtv, handler)
        with pytest.raises(RuntimeError, match="already called"):
            setup_realtime_delegation(rtv, handler)

    async def test_serves_delegate_task_from_the_room_of_the_call(self):
        rtv = _rtv()
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="exec-agent",
            task="do work",
        )
        kit.delegate = AsyncMock(return_value=task_handle)
        setup_realtime_delegation(rtv, DelegateHandler(kit, notify="rtv-main"))

        entry = rtv._registry.lookup("delegate_task", "room-1")
        with tool_call_in("room-1"):
            result_str = await entry.serve({"agent": "exec-agent", "task": "do work"})

        result = json.loads(result_str)
        assert result["status"] == "delegated"
        assert result["task_id"] == "t1"
        assert kit.delegate.call_args.kwargs["room_id"] == "room-1"

    async def test_outside_a_tool_call_returns_error(self):
        rtv = _rtv()
        setup_realtime_delegation(rtv, DelegateHandler(MagicMock()))

        entry = rtv._registry.lookup("delegate_task", None)
        result = json.loads(await entry.serve({"agent": "a", "task": "b"}))

        assert "error" in result

    async def test_other_tools_stay_the_host_s(self):
        called = []

        async def original_handler(name: str, arguments: dict) -> str:
            called.append(name)
            return json.dumps({"ok": True})

        rtv = _rtv(tool_handler=original_handler)
        setup_realtime_delegation(rtv, DelegateHandler(MagicMock()))

        result_str = await rtv.tool_handler("some_other_tool", {})
        assert json.loads(result_str) == {"ok": True}
        assert called == ["some_other_tool"]

    def test_a_channel_without_tools_declares_it(self):
        rtv = _rtv()

        setup_realtime_delegation(rtv, DelegateHandler(MagicMock()))

        assert _declared(rtv) == ["delegate_task"]
