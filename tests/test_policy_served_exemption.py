"""The policy exemption covers the tool the channel serves, not a name (RFC §21.1, RMK-294).

``activate_skill``, ``read_skill_reference``, ``read_stored_result``,
``find_tools`` and ``list_tools`` escape the tool policy when the channel
serves them itself. A tool of the host carrying one of these names, on a
channel that does not serve it, is governed like any other.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
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
