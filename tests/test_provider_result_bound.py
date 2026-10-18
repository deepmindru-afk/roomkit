"""A provider-served call's result is bounded as every outcome the model
reads (RMK-480, RFC §21.5).

In a round that mixes a call the provider ran itself with a call the channel
serves, both answering 40 000 characters, the next round reads both bounded
and the stored END rows keep the bounded copies; ON_TOOL_CALL's observers
heard the provider's result whole.
"""

from __future__ import annotations

from typing import Any

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai.base import AIResponse, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel
from tests.tool_doors import TOOL

_BIG = "z" * 40_000


async def _local(name: str, arguments: dict[str, Any]) -> str:
    return _BIG


async def test_a_provider_served_result_is_bounded_like_a_served_one(streaming: bool) -> None:
    calls = [
        AIToolCall(
            id="p1", name="web_fetch", arguments={"url": "x"}, served=ServedCall(result=_BIG)
        ),
        AIToolCall(id="c1", name="lookup", arguments={"q": "x"}),
    ]
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=calls),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=_local, tools=[TOOL]))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    heard: dict[str, int] = {}

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def _audit(event: Any, ctx: Any) -> None:
        heard[event.tool_call_id] = len(str(event.result))

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    parts = [p for m in provider.calls[1].messages if m.role == "tool" for p in m.content]
    read = {p.tool_call_id: len(p.as_text()) for p in parts}
    rows = {
        e.content.tool_id: len(str(e.content.result))
        for e in await kit.store.list_events("r1")
        if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
    }
    await kit.close()

    assert read["p1"] == read["c1"] < len(_BIG)
    assert rows["p1"] == rows["c1"] < len(_BIG)
    assert heard["p1"] == len(_BIG)
