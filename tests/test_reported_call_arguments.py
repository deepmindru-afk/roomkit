"""A call's report carries the arguments it ran with, or that the gate had
when it stopped it, on every door (RMK-480, RFC §9.3).

The model flattens a hub tool's call; every gate folds it back into shape
before BEFORE_TOOL_USE. A call that hook blocks is reported with the folded
arguments, and one it rewrites into a shape the schema refuses with the
rewritten ones, as a call that ran is reported with those it ran with; a
call its turn cut while it ran, with those it ran with.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel
from tests.tool_doors import DOORS, TOOL, Hooks, run_door

_HUB = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"action": {"type": "string"}, "params": {"type": "object"}},
    "required": ["action"],
}
_FLAT = {"action": "list", "board_id": "b1"}
_FOLDED = {"action": "list", "params": {"board_id": "b1"}}


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return "fine"


async def _block(event: Any, ctx: Any) -> HookResult:
    return HookResult.block("no")


async def _invalid_rewrite(event: Any, ctx: Any) -> HookResult:
    return HookResult(action="allow", metadata={"arguments": {"action": 7}})


@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize(
    ("before", "reported"),
    [(_block, _FOLDED), (_invalid_rewrite, {"action": 7}), (None, _FOLDED)],
    ids=["blocked", "rewritten-invalid", "served"],
)
async def test_a_report_carries_the_arguments_the_gate_had(
    door: str, before: Any, reported: dict[str, Any]
) -> None:
    seen = await run_door(
        door, _served, arguments=dict(_FLAT), hooks=Hooks(before=before), schema=_HUB
    )

    assert [event.arguments for event in seen.reports] == [reported]
    assert [row.arguments for row in seen.rows] in ([], [reported])


async def test_a_call_its_turn_cut_is_reported_with_the_arguments_it_ran_with(
    streaming: bool,
) -> None:
    """A de-tokenising BEFORE_TOOL_USE hook rewrites the model's arguments;
    the turn is cut while the handler runs with them: the cancelled report
    carries them, as the realtime session's does."""
    ran: list[dict[str, Any]] = []

    async def hanging(name: str, arguments: dict[str, Any]) -> str:
        ran.append(dict(arguments))
        await asyncio.sleep(30)
        return "late"

    async def detokenise(event: Any, ctx: Any) -> HookResult:
        return HookResult(action="allow", metadata={"arguments": {"q": "real@x"}})

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"q": "<EMAIL_1>"})],
            ),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=hanging, tools=[TOOL]))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="gate")(detokenise)
    reports: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def _audit(event: Any, ctx: Any) -> None:
        reports.append(event)

    message = InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    turn = asyncio.create_task(kit.process_inbound(message))
    await until(lambda: bool(ran))
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)
    await until(lambda: bool(reports))
    await kit.close()

    assert [(e.cancelled, e.arguments) for e in reports] == [(True, {"q": "real@x"})]
