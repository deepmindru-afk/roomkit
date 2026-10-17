"""A human-input request of a turn is answered on the room owner (RFC §10.6.5).

The tool waiting for the human runs on the turn's executor; the owner keeps
the request's record, answers it (``kit.answer_input`` / ``kit.reject_input``,
or the channel's handler, which forwards there), and holds an answer that
comes before the request.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.models.pending_input import PendingInputEvent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.turns._owner_inputs import OwnerInputs
from roomkit.turns.wire import InputAnswer, InputRequest
from tests.test_framework import SimpleChannel

ASK = AITool(
    name="ask_user",
    description="Ask the user.",
    parameters={"type": "object", "properties": {"question": {"type": "string"}}},
)


async def _kit(
    *,
    timeout: float = 5,
) -> tuple[RoomKit, AIChannel, HumanInputToolHandler, list[PendingInputEvent], list[Any]]:
    """A kit whose agent asks the user once, then answers."""
    call = AIResponse(
        content="",
        tool_calls=[AIToolCall(id="c1", name="ask_user", arguments={"question": "Color?"})],
    )
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="Thanks.")])
    human = HumanInputToolHandler(tool_names={"ask_user"}, timeout=timeout, tool_definitions=[ASK])
    agent = AIChannel("agent", provider=provider, human_input_handler=human)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    asked: list[PendingInputEvent] = []
    reports: list[Any] = []

    @kit.turn_executor.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC)
    async def on_input(event: PendingInputEvent, ctx: Any) -> HookResult:
        asked.append(event)
        return HookResult.allow()

    @kit.turn_executor.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def on_call(event: ToolCallEvent, ctx: Any) -> None:
        reports.append(event)

    return kit, agent, human, asked, reports


async def _asked(asked: list[PendingInputEvent]) -> str:
    for _ in range(200):
        if asked:
            return asked[0].pending_id
        await asyncio.sleep(0.01)
    raise AssertionError("The human was never asked")


def _say(kit: RoomKit) -> asyncio.Task[Any]:
    message = InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="Hi"))
    return asyncio.create_task(kit.process_inbound(message, room_id="r1"))


async def test_the_owner_answers_a_turns_request() -> None:
    kit, agent, _, asked, reports = await _kit()
    turn = _say(kit)
    pending_id = await _asked(asked)

    assert await kit.answer_input(pending_id, "Blue") is True
    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.result == "Blue"
    # The request is answered once.
    assert await kit.answer_input(pending_id, "Red") is False
    assert await kit.reject_input(pending_id, "late") is False
    await kit.close()


async def test_the_channels_handler_answers_through_the_owner() -> None:
    kit, _, human, asked, reports = await _kit()
    turn = _say(kit)
    pending_id = await _asked(asked)

    assert human.handler.resolve(pending_id, "Green") is True
    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.result == "Green"
    # The owner recorded it: the other door answers it no more.
    assert await kit.answer_input(pending_id, "Red") is False
    await kit.close()


async def test_a_rejection_reaches_the_tool_as_a_refusal() -> None:
    kit, _, _, asked, reports = await _kit()
    turn = _say(kit)
    pending_id = await _asked(asked)

    assert await kit.reject_input(pending_id, "Not now") is True
    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.is_error
    assert "Not now" in str(report.result)
    await kit.close()


async def test_a_request_no_one_answers_in_time_is_closed_on_the_owner() -> None:
    kit, _, _, asked, reports = await _kit(timeout=0.2)
    turn = _say(kit)
    pending_id = await _asked(asked)

    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.is_error
    assert await kit.answer_input(pending_id, "Too late") is False
    await kit.close()


async def test_a_request_a_hook_blocks_is_closed_on_the_owner() -> None:
    kit, _, _, _, reports = await _kit()
    denied: list[str] = []

    @kit.turn_executor.hook(
        HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC, priority=-1
    )
    async def deny(event: PendingInputEvent, ctx: Any) -> HookResult:
        denied.append(event.pending_id)
        return HookResult.block("No humans today")

    await asyncio.wait_for(_say(kit), 5)

    [report] = reports
    assert report.is_error
    [pending_id] = denied
    # Settled on the executor: the owner takes no answer for it.
    assert await kit.answer_input(pending_id, "Too late") is False
    await kit.close()


# -- The owner's record ----------------------------------------------------------


class _Turn:
    """A turn of the owner's, as the record reads it."""

    def __init__(self, room_id: str = "r1") -> None:
        self.request = type("Request", (), {"room_id": room_id})()
        self.sent: list[InputAnswer] = []

    async def send_control(self, message: InputAnswer) -> None:
        self.sent.append(message)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_an_answer_that_comes_before_its_request_is_held_for_it() -> None:
    inputs = OwnerInputs(hold=60)
    turn = _Turn()

    assert await inputs.answer("p1", "Early") is True
    inputs.opened(turn, InputRequest(request_id="p1", tool_name="ask_user", arguments={}))  # type: ignore[arg-type]
    await _settle()

    assert turn.sent == [InputAnswer(request_id="p1", outcome="answered", value="Early")]


async def test_an_answer_held_past_its_bound_is_let_go() -> None:
    inputs = OwnerInputs(hold=0.05)
    turn = _Turn()

    assert await inputs.answer("p1", "Early") is True
    await asyncio.sleep(0.1)
    inputs.opened(turn, InputRequest(request_id="p1", tool_name="ask_user", arguments={}))  # type: ignore[arg-type]
    await _settle()

    assert turn.sent == []


async def test_an_answer_for_another_rooms_request_is_refused() -> None:
    inputs = OwnerInputs(hold=60)
    turn = _Turn("r2")
    inputs.opened(turn, InputRequest(request_id="p1", tool_name="ask_user", arguments={}))  # type: ignore[arg-type]

    assert await inputs.answer("p1", "From r1", room_id="r1") is False
    assert await inputs.answer("p1", "From r2", room_id="r2") is True
    await _settle()

    assert [a.value for a in turn.sent] == ["From r2"]


@pytest.mark.parametrize("ended", ["closed", "turn_ended"])
async def test_a_request_closed_or_whose_turn_ended_takes_no_answer(ended: str) -> None:
    inputs = OwnerInputs(hold=60)
    turn = _Turn()
    inputs.opened(turn, InputRequest(request_id="p1", tool_name="ask_user", arguments={}))  # type: ignore[arg-type]

    if ended == "closed":
        inputs.closed("p1")
    else:
        inputs.turn_ended(turn)  # type: ignore[arg-type]

    assert await inputs.answer("p1", "Late") is False
    await _settle()
    assert turn.sent == []
