"""Tests for delegation tool, handler, and setup_delegation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import (
    RealtimeVoiceChannel,
    _current_voice_session,
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
        assert channel._tool_handler is not None
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
            result_str = await channel._tool_handler(
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
        result_str = await channel._tool_handler(
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

        result_str = await channel._tool_handler("some_other_tool", {})
        assert json.loads(result_str) == {"ok": True}
        assert called == ["some_other_tool"]


# -- setup_realtime_delegation ------------------------------------------------


def _declare_for_real(rtv: MagicMock) -> None:
    """Let a mocked voice channel declare the tools orchestration injects."""
    rtv._inject_orchestration_tool.side_effect = lambda tool: (
        RealtimeVoiceChannel._inject_orchestration_tool(rtv, tool)
    )


class TestSetupRealtimeDelegation:
    def test_injects_tool_dict_and_wraps_handler(self):
        """Should add delegate tool to _tools and wrap _tool_handler."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        _declare_for_real(rtv)
        rtv.channel_id = "rtv-main"
        rtv._tools = [{"name": "existing", "description": "test", "parameters": {}}]
        rtv._tool_handler = None

        kit = MagicMock()
        handler = DelegateHandler(kit)

        setup_realtime_delegation(rtv, handler)

        tool_names = [t["name"] for t in rtv._tools]
        assert "delegate_task" in tool_names
        assert rtv._tool_handler is not None

    def test_double_setup_raises(self):
        """Should raise RuntimeError if called twice."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        _declare_for_real(rtv)
        rtv.channel_id = "rtv-main"
        rtv._tools = []
        rtv._tool_handler = None

        kit = MagicMock()
        handler = DelegateHandler(kit)

        setup_realtime_delegation(rtv, handler)
        with pytest.raises(RuntimeError, match="already called"):
            setup_realtime_delegation(rtv, handler)

    async def test_intercepts_delegate_task(self):
        """Wrapped handler should call DelegateHandler for delegate_task."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        _declare_for_real(rtv)
        rtv.channel_id = "rtv-main"
        rtv._tools = []
        rtv._tool_handler = None

        session = MagicMock(spec=VoiceSession)
        session.id = "sess-1"
        rtv.session_rooms = {"sess-1": "room-1"}

        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="exec-agent",
            task="do work",
        )
        kit.delegate = AsyncMock(return_value=task_handle)

        handler = DelegateHandler(kit, notify="rtv-main")
        setup_realtime_delegation(rtv, handler)

        # Set voice session context
        token = _current_voice_session.set(session)
        try:
            result_str = await rtv._tool_handler(
                "delegate_task",
                {"agent": "exec-agent", "task": "do work"},
            )
        finally:
            _current_voice_session.reset(token)

        result = json.loads(result_str)
        assert result["status"] == "delegated"
        assert result["task_id"] == "t1"

    async def test_no_session_returns_error(self):
        """Without voice session context, should return error."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        _declare_for_real(rtv)
        rtv.channel_id = "rtv-main"
        rtv._tools = []
        rtv._tool_handler = None

        kit = MagicMock()
        handler = DelegateHandler(kit)
        setup_realtime_delegation(rtv, handler)

        token = _current_voice_session.set(None)
        try:
            result_str = await rtv._tool_handler(
                "delegate_task",
                {"agent": "a", "task": "b"},
            )
        finally:
            _current_voice_session.reset(token)

        result = json.loads(result_str)
        assert "error" in result

    async def test_passes_through_other_tools(self):
        """Non-delegate tools should pass through to original handler."""
        called = []

        async def original_handler(name: str, arguments: dict) -> str:
            called.append(name)
            return json.dumps({"ok": True})

        rtv = MagicMock(spec=RealtimeVoiceChannel)

        _declare_for_real(rtv)
        rtv.channel_id = "rtv-main"
        rtv._tools = []
        rtv._tool_handler = original_handler

        kit = MagicMock()
        handler = DelegateHandler(kit)
        setup_realtime_delegation(rtv, handler)

        result_str = await rtv._tool_handler("some_other_tool", {})
        assert json.loads(result_str) == {"ok": True}
        assert called == ["some_other_tool"]

    def test_none_tools_initializes_list(self):
        """When _tools is None, should create new list with delegate tool."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        _declare_for_real(rtv)
        rtv.channel_id = "rtv-main"
        rtv._tools = None
        rtv._tool_handler = None

        kit = MagicMock()
        handler = DelegateHandler(kit)
        setup_realtime_delegation(rtv, handler)

        assert len(rtv._tools) == 1
        assert rtv._tools[0]["name"] == "delegate_task"
