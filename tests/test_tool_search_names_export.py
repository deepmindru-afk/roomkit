"""The Tool Search tool names are public and name what a channel serves (RMK-468).

A host that treats the discovery tools apart (a guard that must not judge a
catalogue schema as tool output, a view that labels them) reads their names
from roomkit rather than spelling them itself.
"""

from __future__ import annotations

import roomkit
import roomkit.channels
from roomkit import TOOL_FIND_TOOLS, TOOL_LIST_TOOLS, TOOL_SEARCH_INFRA_TOOL_NAMES
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_HOST_TOOLS = [
    AITool(name="search", description="Search the archive", parameters={}),
    AITool(name="write_note", description="Write a note", parameters={}),
]


def test_the_names_are_exported_from_roomkit_and_roomkit_channels() -> None:
    for name in ("TOOL_FIND_TOOLS", "TOOL_LIST_TOOLS", "TOOL_SEARCH_INFRA_TOOL_NAMES"):
        assert name in roomkit.__all__
        assert getattr(roomkit.channels, name) is getattr(roomkit, name)
    assert (TOOL_FIND_TOOLS, TOOL_LIST_TOOLS) == ("find_tools", "list_tools")
    assert frozenset({TOOL_FIND_TOOLS, TOOL_LIST_TOOLS}) == TOOL_SEARCH_INFRA_TOOL_NAMES


async def _declared(*, tool_search: bool) -> set[str]:
    provider = MockAIProvider(ai_responses=[AIResponse(content="done", finish_reason="stop")])
    ch = AIChannel(
        "ai1", provider=provider, tool_search=tool_search, tool_search_pinned={"search"}
    )
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [tool.model_dump() for tool in _HOST_TOOLS]},
    )
    await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )
    return {tool.name for tool in provider.calls[0].tools or []}


async def test_the_names_are_the_tools_tool_search_adds_to_a_channel() -> None:
    added = await _declared(tool_search=True) - await _declared(tool_search=False)

    assert added == TOOL_SEARCH_INFRA_TOOL_NAMES
