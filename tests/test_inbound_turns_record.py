"""The caller reads how each agent's turn ended, even a turn that wrote no
message (RMK-437, RFC §6.4).

``InboundResult.response_metadata["turns"][channel_id]`` carries each
replying channel's ``loop_end_reason`` (an ACP agent's stop reason) and
``ai_usage``: one agent's never overwrites another's, a streamed reply's or a
buffered one's, and an answer to an answer is no reply to the caller. The key
is RoomKit's: a value a host writes there is not carried. An ACP turn's
``ON_AI_RESPONSE`` names its stop reason, ``completed`` for ``end_turn``; an
ACP turn never prompted names no end.
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
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.models.response_metadata import recorded_turn_end
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_response_metadata
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


def _agent(
    channel_id: str, responses: list[AIResponse], *, streaming: bool = True, handler: Any = None
) -> Agent:
    return Agent(
        channel_id,
        provider=MockAIProvider(ai_responses=responses, streaming=streaming),
        tools=[LOOKUP],
        tool_handler=handler or _found,
        tool_search=False,
        max_tool_rounds=1,
    )


class _Unread(SimpleChannel):
    """A streaming transport that never reads the answer."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(self, text_stream: Any, *args: Any) -> ChannelOutput:
        return ChannelOutput.empty()


async def _ask(kit: RoomKit, *agents: str, transport: SimpleChannel | None = None) -> Any:
    kit.register_channel(transport or SimpleChannel("sms"))
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


async def test_a_supervisor_whose_pass_was_cut_names_its_end() -> None:
    """A buffered reply: the end is on the supervisor's fallback."""
    kit = RoomKit()
    supervisor = _agent("sup", [SILENT_CALL] * 5)
    worker = Agent("worker", provider=MockAIProvider(responses=["worker answer"]))
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(supervisor)
    kit.register_channel(worker)
    orchestration = Supervisor(
        supervisor=supervisor, workers=[worker], strategy="parallel", auto_delegate=True
    )
    await kit.create_room(room_id="r", orchestration=orchestration)
    await kit.attach_channel("r", "sms")

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )

    assert result.response_metadata["turns"]["sup"]["loop_end_reason"] == "max_rounds"
    await kit.close()


async def test_an_answer_to_an_answer_is_not_the_channel_s_reply() -> None:
    """b answers the user, then a's answer, where its loop is cut: the caller
    reads b's own reply, not its answer to a."""
    kit = RoomKit()
    kit.register_channel(_agent("a", [AIResponse(content="A says hi.")] * 5))
    kit.register_channel(_agent("b", [AIResponse(content="B ok."), *[SILENT_CALL] * 10]))

    result = await _ask(kit, "a", "b")

    turns = result.response_metadata["turns"]
    assert turns["a"]["loop_end_reason"] == "completed"
    assert turns["b"]["loop_end_reason"] == "completed"
    await kit.close()


@pytest.mark.parametrize("value", [3, ["x"], {"host": "mine"}])
async def test_a_host_value_under_turns_is_not_carried(value: Any) -> None:
    async def writes_turns(name: str, arguments: dict[str, Any]) -> str:
        record = current_response_metadata()
        if record is not None:
            record["turns"] = value
        return "found"

    kit = RoomKit()
    kit.register_channel(_agent("a", [SILENT_CALL] * 5))
    kit.register_channel(_agent("b", [SILENT_CALL] * 5, handler=writes_turns))

    result = await _ask(kit, "a", "b")

    assert result.error is None
    turns = result.response_metadata["turns"]
    assert set(turns) == {"a", "b"}
    assert turns["a"]["loop_end_reason"] == "max_rounds"
    await kit.close()


async def test_an_acp_turn_never_prompted_names_no_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, _, _ = _channel(Path(tmp), emit_updates=False)
        kit.register_channel(channel)

        result = await _ask(kit, channel.channel_id, transport=_Unread("sms"))

        assert channel.channel_id not in result.response_metadata.get("turns", {})
        await kit.close()


@pytest.mark.parametrize(
    ("acp", "end"),
    [
        ({}, None),
        ({"prompt_returned": True}, "completed"),
        ({"interrupted": True}, "interrupted"),
        # Its stop reason first: a drain that failed after the prompt
        # returned does not hide why the agent stopped.
        (
            {"prompt_returned": True, "stop_reason": "max_tokens", "interrupted": True},
            "max_tokens",
        ),
    ],
)
def test_an_acp_record_names_its_end(acp: dict[str, Any], end: str | None) -> None:
    assert recorded_turn_end({"acp": acp}) == end
