"""The policy exemption covers the tool the channel serves, not a name (RFC §21.1, RMK-294).

``activate_skill``, ``read_skill_reference``, ``read_stored_result``,
``find_tools`` and ``list_tools`` escape the tool policy when the channel
serves them itself. A tool of the host carrying one of these names, on a
channel that does not serve it, is governed like any other.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels._conference_tools import declared_tools
from roomkit.channels._served_tools import CollisionLog
from roomkit.channels._tool_registry import ToolNameCollisionError
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import DELEGATE_TOOL, DelegateHandler, setup_delegation
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.policy import ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_hook_tool_restrictions import _DONE, _Recorder, _round, _turn
from tests.test_realtime_fixed_tools import call

_SEARCH_ONLY = ToolPolicy(allow=["search_*"])


@pytest.mark.parametrize("name", ["list_tools", "activate_skill", "read_skill_reference"])
async def test_a_host_tool_under_an_exempt_name_is_governed(streaming: bool, name: str) -> None:
    """The channel serves no skill and no Tool Search here: the name is the host's."""
    provider = MockAIProvider(ai_responses=[_round("c0", name), _DONE], streaming=streaming)
    calls = _Recorder()
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=calls.handler,
        tools=[AITool(name=name, description="host tool", parameters={})],
        tool_policy=_SEARCH_ONLY,
        tool_search=False,
    )

    run = await _turn(ch)

    assert calls.ran == []
    assert run.calls[0].failed
    assert all(name not in {t.name for t in c.tools or []} for c in provider.calls)


async def test_the_channel_s_own_exempt_tool_still_escapes_the_policy(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[_round("c0", "read_stored_result", {"result_id": "x"}), _DONE],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name="search_docs", description="Search", parameters={})],
        tool_policy=_SEARCH_ONLY,
    )

    run = await _turn(ch)

    # Served by the channel: its own answer (nothing stored), not a policy refusal.
    assert "not permitted" not in str(run.calls[0].result)


async def test_realtime_a_host_tool_under_an_exempt_name_is_governed() -> None:
    handler = AsyncMock(return_value="host ran")
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": "list_tools", "description": "host tool", "parameters": {}}],
        tool_policy=_SEARCH_ONLY,
        tool_search=False,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt")
    session = await channel.start_session(room.id, "participant", object())
    try:
        result = await call(channel, provider, session, "list_tools", {})
    finally:
        await kit.close()

    handler.assert_not_awaited()
    assert "not permitted" in json.dumps(result)


class TestOneDeclarationPerName:
    """RMK-294: a name the channel serves is declared with its definition only,
    and no name is declared twice (RFC §21.1)."""

    @pytest.mark.parametrize("name", ["read_stored_result", "list_tools"])
    def test_a_static_host_tool_under_a_served_name_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match=name):
            AIChannel(
                "ai1",
                provider=MockAIProvider(),
                tools=[AITool(name=name, description="host tool", parameters={})],
            )

    def test_turning_tool_search_off_frees_its_names(self) -> None:
        AIChannel(
            "ai1",
            provider=MockAIProvider(),
            tools=[AITool(name="list_tools", description="host tool", parameters={})],
            tool_search=False,
        )

    async def test_a_dynamic_host_tool_under_a_served_name_is_not_declared(
        self, streaming: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tools=[AITool(name="search_docs", description="Search", parameters={})],
        )
        # A tool the binding brings with the turn, under the channel's own name.
        host = {"name": "read_stored_result", "description": "host", "parameters": {}}

        await _turn(ch, binding_tools=[host])

        # The channel declares its own from the first round (RFC §6.4).
        declared = {t.name: t.description for t in provider.calls[0].tools or []}
        assert declared["read_stored_result"] != "host"
        assert "read_stored_result" in caplog.text

    def test_orchestration_over_a_host_tool_s_name_is_refused(self, streaming: bool) -> None:
        """Declared once and served by the other, the model would call one
        tool's schema on the other's server: refused when it is given (RMK-307)."""
        provider = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tools=[AITool(name="delegate_task", description="host", parameters={})],
        )

        with pytest.raises(ToolNameCollisionError, match="a tool of the host"):
            setup_delegation(ch, DelegateHandler(MagicMock()))

    async def test_a_turn_tool_under_an_orchestration_name_is_not_declared(
        self, streaming: bool
    ) -> None:
        """A tool the turn brings under a name orchestration serves comes too
        late to be refused: the orchestration's definition is declared."""
        provider = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
        ch = AIChannel("ai1", provider=provider)
        setup_delegation(ch, DelegateHandler(MagicMock()))
        host = {"name": "delegate_task", "description": "host", "parameters": {}}

        await _turn(ch, binding_tools=[host])

        declared = [t for t in provider.calls[0].tools or [] if t.name == "delegate_task"]
        assert [t.description for t in declared] == [DELEGATE_TOOL.description]

    async def test_plan_tasks_is_the_host_s_without_a_planner(self, streaming: bool) -> None:
        provider = MockAIProvider(
            ai_responses=[_round("c0", "plan_tasks"), _DONE], streaming=streaming
        )
        calls = _Recorder()
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=calls.handler,
            tools=[AITool(name="plan_tasks", description="host planner", parameters={})],
        )

        await _turn(ch)

        assert calls.ran == ["plan_tasks"]

    def test_realtime_a_static_host_tool_under_a_served_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="find_tools"):
            RealtimeVoiceChannel(
                "rt",
                provider=MockRealtimeProvider(),
                transport=MockRealtimeTransport(),
                tools=[{"name": "find_tools", "description": "host", "parameters": {}}],
                tool_search=True,
            )

    def test_realtime_a_session_s_host_tools_are_declared_once(self) -> None:
        channel = RealtimeVoiceChannel(
            "rt",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            tools=[{"name": "lookup", "description": "Look up", "parameters": {}}],
            tool_search=True,
            tool_search_pinned=["lookup"],
        )
        composed = channel._compose_session_tools(
            "s1",
            [
                {"name": "find_tools", "description": "host", "parameters": {}},
                {"name": "lookup", "description": "first", "parameters": {}},
                {"name": "lookup", "description": "later", "parameters": {}},
            ],
        )

        assert [t["name"] for t in composed or []].count("find_tools") == 1
        assert all(t.get("description") != "host" for t in composed or [])
        assert [t["description"] for t in composed or [] if t["name"] == "lookup"] == ["later"]


class TestEveryEntryReadsTheSameDeclaration:
    """RMK-294 review: the delegation gate, the realtime gate's schema, the
    generation hook and the conference follow the same rule (RFC §21.1)."""

    async def _session(self, **kwargs: Any) -> tuple[RoomKit, RealtimeVoiceChannel, Any]:
        channel = RealtimeVoiceChannel(
            "rt",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            tool_handler=AsyncMock(return_value="host ran"),
            **kwargs,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt")
        session = await channel.start_session(room.id, "participant", object())
        return kit, channel, session

    async def test_a_backend_call_to_a_channel_tool_name_is_refused(self) -> None:
        """A reasoning backend reaches the handler only: no name is the channel's there."""
        kit, channel, session = await self._session(
            tools=[{"name": "lookup_order", "description": "Look up", "parameters": {}}],
            tool_search=True,
            tool_policy=ToolPolicy(allow=["lookup_*"]),
        )
        try:
            _, denial, _ = await channel._authorize_realtime_tool(
                "list_tools", {}, "c1", session.room_id, session, channel_serves=False
            )
        finally:
            await kit.close()

        assert denial is not None and "not declared" in denial.body

    def test_a_host_tool_given_twice_is_refused(self) -> None:
        """Declared once and served by the other, the model would call one
        tool's schema on the other's server (RMK-307)."""
        schema_a = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        schema_b = {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}
        with pytest.raises(ValueError, match="'lookup' is given twice"):
            RealtimeVoiceChannel(
                "rt",
                provider=MockRealtimeProvider(),
                transport=MockRealtimeTransport(),
                tools=[
                    {"name": "lookup", "description": "first", "parameters": schema_a},
                    {"name": "lookup", "description": "later", "parameters": schema_b},
                ],
            )

    @pytest.mark.parametrize(
        ("name", "tool_search"),
        [("read_stored_result", None), ("find_tools", True)],
        ids=["added-under-a-served-name", "a-served-tool-redefined"],
    )
    async def test_a_generation_hook_cannot_declare_a_served_name(
        self,
        streaming: bool,
        caplog: pytest.LogCaptureFixture,
        name: str,
        tool_search: bool | None,
    ) -> None:
        """Adding a tool under a name the channel serves, or redefining one it
        declared, keeps the channel's definition, and a warning names it
        (RFC §21.1)."""
        provider = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tools=[AITool(name="search_docs", description="Search", parameters={})],
            tool_search=tool_search,
        )
        forged = AITool(name=name, description="the hook's", parameters={})

        async def hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            tools = [t for t in gen_event.ai_context.tools if t.name != name]
            gen_event.ai_context = gen_event.ai_context.model_copy(
                update={"tools": [*tools, forged]}
            )
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = hook

        await _turn(ch)

        declared = {t.name: t.description for t in provider.calls[0].tools or []}
        assert declared.get(name) != "the hook's"
        assert name in caplog.text

    def test_the_conference_declares_a_name_once(self) -> None:
        provider = MockRealtimeProvider()
        config = ConferenceRealtimeConfig(
            provider=provider,
            tools=[
                {"name": "lookup", "description": "first", "parameters": {}},
                {"name": "lookup", "description": "later", "parameters": {}},
            ],
        )

        declared = declared_tools(config, CollisionLog("conf"))

        assert [t["description"] for t in declared or []] == ["later"]

    def test_a_static_tool_under_a_human_input_tool_name_is_refused(self) -> None:
        ask = AITool(name="ask", description="Ask the user", parameters={})
        with pytest.raises(ValueError, match="ask"):
            AIChannel(
                "ai1",
                provider=MockAIProvider(),
                tools=[ask],
                human_input_handler=HumanInputToolHandler(
                    tool_names={"ask"}, tool_definitions=[ask]
                ),
            )
