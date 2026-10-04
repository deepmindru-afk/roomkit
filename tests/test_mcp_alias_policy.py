"""A call under an MCP alias is judged under the tool it runs (RMK-483, RFC §21.1).

``MCPToolProvider.as_tool_handler()`` serves ``mcp__<server>__<tool>`` as
``<tool>``. The tool policy and a skill's gate judge both names, on every
door, so a tool they refuse never runs under its alias; the alias itself is
still served for a tool they admit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import mcp.types as mt
import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.base import AITool
from roomkit.tools.mcp import MCPToolProvider
from roomkit.tools.policy import ToolPolicy, judged_names, served_tool_name
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_realtime_skills import _registry_with_skill

ALIAS = "mcp__crm__delete_records"


class _Server:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.ran.append(name)
        return mt.CallToolResult(
            content=[mt.TextContent(type="text", text=f"{name} done")], isError=False
        )


def _mcp_handler(server: _Server) -> Any:
    provider = MCPToolProvider("http://crm.invalid/mcp")
    provider._connected = True
    provider._session = server  # type: ignore[assignment]
    provider._tools = [
        AITool(name=n, description=n, parameters={}) for n in ("search_records", "delete_records")
    ]
    provider._tool_set = {"search_records", "delete_records"}
    return provider.as_tool_handler()


def test_an_alias_is_judged_under_both_names() -> None:
    assert served_tool_name(ALIAS) == "delete_records"
    assert served_tool_name("delete_records") == "delete_records"
    assert judged_names(ALIAS) == (ALIAS, "delete_records")
    assert judged_names("search_records") == ("search_records",)


async def _on_session(policy: ToolPolicy, server: _Server, name: str) -> str:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=_mcp_handler(server),
        tool_policy=policy,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c1", name, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()
    return provider.tool_results[0][2]


async def _on_conference(policy: ToolPolicy, server: _Server, name: str) -> str:
    provider = MockRealtimeProvider()
    handler = _mcp_handler(server)

    async def room_handler(room_id: str, tool: str, arguments: dict[str, Any]) -> Any:
        return await handler(tool, arguments)

    config = ConferenceRealtimeConfig(
        provider=provider, tool_handler=room_handler, tool_policy=policy
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    await provider.simulate_tool_call(session, "c1", name, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()
    return provider.tool_results[0][2]


DOORS = {"session": _on_session, "conference": _on_conference}


@pytest.mark.parametrize("door", list(DOORS))
@pytest.mark.parametrize(
    "policy",
    [ToolPolicy(deny=["delete_*"]), ToolPolicy(allow=["search_*"])],
    ids=["deny", "allow"],
)
async def test_a_tool_the_policy_refuses_never_runs_under_its_alias(
    door: str, policy: ToolPolicy
) -> None:
    server = _Server()

    answer = await DOORS[door](policy, server, ALIAS)

    assert "not permitted by the agent's tool policy" in answer
    assert server.ran == []


@pytest.mark.parametrize("door", list(DOORS))
async def test_an_alias_of_an_admitted_tool_is_still_served(door: str) -> None:
    server = _Server()

    answer = await DOORS[door](ToolPolicy(deny=["delete_*"]), server, "mcp__crm__search_records")

    assert "search_records done" in answer
    assert server.ran == ["search_records"]
    await asyncio.sleep(0)


async def test_a_tool_a_skill_gates_never_runs_under_its_alias(tmp_path: Path) -> None:
    server = _Server()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=_mcp_handler(server),
        skills=_registry_with_skill(tmp_path, allowed_tools="delete_*"),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")

    await provider.simulate_tool_call(session, "c1", ALIAS, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()

    assert "gated by a skill" in provider.tool_results[0][2]
    assert server.ran == []
