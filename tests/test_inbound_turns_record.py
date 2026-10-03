"""The caller reads how each agent's turn ended, even a turn that wrote no
message (RMK-437, RFC §6.4).

``InboundResult.response_metadata["turns"][channel_id]`` carries each
replying channel's ``loop_end_reason`` (an ACP agent's stop reason) and
``ai_usage``: one agent's never overwrites another's. An ACP turn's
``ON_AI_RESPONSE`` names its stop reason, ``completed`` for ``end_turn``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import acp
import pytest
from acp.schema import PromptResponse

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_channels.test_acp import _channel
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
SILENT_CALL = AIResponse(
    content="",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


def _agent(channel_id: str, responses: list[AIResponse]) -> Agent:
    return Agent(
        channel_id,
        provider=MockAIProvider(ai_responses=responses, streaming=True),
        tools=[LOOKUP],
        tool_handler=_found,
        tool_search=False,
        max_tool_rounds=1,
    )


async def _ask(kit: RoomKit, *agents: str) -> Any:
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    for agent in agents:
        await kit.attach_channel("r", agent, category=ChannelCategory.INTELLIGENCE)
    return await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )


async def test_a_turn_that_wrote_no_message_names_its_end_to_the_caller() -> None:
    kit = RoomKit()
    kit.register_channel(_agent("ai", [SILENT_CALL] * 5))

    result = await _ask(kit, "ai")

    turn = result.response_metadata["turns"]["ai"]
    assert turn["loop_end_reason"] == "max_rounds"
    assert "ai_usage" in turn
    await kit.close()


async def test_two_agents_each_keep_their_own_end() -> None:
    kit = RoomKit()
    kit.register_channel(_agent("cut", [SILENT_CALL] * 5))
    kit.register_channel(_agent("done", [AIResponse(content="Here you go.")]))

    result = await _ask(kit, "cut", "done")

    turns = result.response_metadata["turns"]
    assert turns["cut"]["loop_end_reason"] == "max_rounds"
    assert turns["done"]["loop_end_reason"] == "completed"
    await kit.close()


@pytest.mark.parametrize(
    ("stop", "end"),
    [("max_tokens", "max_tokens"), ("cancelled", "cancelled"), ("end_turn", "completed")],
)
async def test_an_acp_turn_names_its_stop_reason(stop: str, end: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), emit_updates=False)

        async def prompt(session_id: str, *a: Any, **k: Any) -> PromptResponse:
            await connection.client.session_update(
                session_id, acp.update_agent_message_text("Let me look into that.")
            )
            return PromptResponse(stop_reason=stop)

        connection.prompt = prompt  # type: ignore[method-assign]
        kit.register_channel(channel)
        seen: list[Any] = []

        @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="seen")
        async def on_response(event: Any, ctx: Any) -> None:
            seen.append(event.loop_end_reason)

        result = await _ask(kit, channel.channel_id)
        await asyncio.sleep(0.05)

        assert result.response_metadata["turns"][channel.channel_id] == {"loop_end_reason": end}
        assert seen == [end]
        await kit.close()
