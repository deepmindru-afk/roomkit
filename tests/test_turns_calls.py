"""The calls a turn makes to its room owner, each side alone (RFC §10.6.5)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.channels._tool_registry import orchestration_tool
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AITool
from roomkit.tools.context import ToolCallContext, tool_turn_context
from roomkit.turns._executor_calls import ExecutorCalls
from roomkit.turns._owner_calls import OwnerCalls
from roomkit.turns._registry_tools import OwnerToolStore, RegistryOwnerTools
from roomkit.turns.wire import (
    CallCancel,
    CallContext,
    OwnerTool,
    ToolCall,
    ToolResult,
    TurnRequest,
)

PROBE = AITool(name="probe", description="Probe.", parameters={"type": "object"})


def _request() -> TurnRequest:
    return TurnRequest(
        turn_id="t1",
        room_id="r1",
        channel_id="agent",
        event=RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ops", channel_type=ChannelType.WEBSOCKET),
            content=TextContent(body="go"),
        ),
        binding=ChannelBinding(
            channel_id="agent",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        context=RoomContext(room=Room(id="r1")),
        owner_tools=[OwnerTool(tool=PROBE)],
    )


class _Link:
    """What the executor sent its owner."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, frame: Any) -> int:
        self.sent.append(frame)
        return len(self.sent)


async def _calling(calls: ExecutorCalls, link: _Link) -> asyncio.Task[Any]:
    """Start a call of the turn's ``probe`` and wait for its ``tool_call``."""
    serve = calls.owner_tools.entries["probe"].serve
    assert serve is not None

    async def call() -> Any:
        with tool_turn_context(room_id="r1", call=ToolCallContext("r1", "c1", "agent")):
            return await serve({})

    task = asyncio.create_task(call())
    while not link.sent:
        await asyncio.sleep(0)
    return task


async def test_a_call_cut_before_its_result_is_withdrawn() -> None:
    link = _Link()
    calls = ExecutorCalls(_request(), link.send)
    task = await _calling(calls, link)
    [call] = link.sent

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert isinstance(call, ToolCall)
    assert link.sent[1:] == [CallCancel(call_id=call.call_id)]


async def test_a_call_answered_is_not_withdrawn() -> None:
    link = _Link()
    calls = ExecutorCalls(_request(), link.send)
    task = await _calling(calls, link)
    [call] = link.sent

    calls.resolve(ToolResult(call_id=call.call_id, outcome="served", result={"value": "ok"}))

    assert await task == "ok"
    assert len(link.sent) == 1


async def test_an_owner_lost_fails_the_calls_it_will_never_answer() -> None:
    link = _Link()
    calls = ExecutorCalls(_request(), link.send)
    task = await _calling(calls, link)

    calls.fail_all(TurnLinkError("owner_lost"))

    with pytest.raises(TurnLinkError):
        await task


async def test_a_result_for_a_call_no_longer_awaited_is_dropped() -> None:
    link = _Link()
    calls = ExecutorCalls(_request(), link.send)
    task = await _calling(calls, link)
    [call] = link.sent
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    calls.resolve(ToolResult(call_id=call.call_id, outcome="served", result={"value": "late"}))
    calls.resolve(ToolResult(call_id="unknown", outcome="served", result={"value": "x"}))


async def test_a_call_the_link_refuses_fails() -> None:
    async def refusing(frame: Any) -> int:
        raise TurnLinkError("owner_lost")

    calls = ExecutorCalls(_request(), refusing)
    serve = calls.owner_tools.entries["probe"].serve
    assert serve is not None

    with pytest.raises(TurnLinkError), tool_turn_context(room_id="r1"):
        await serve({})


@pytest.mark.parametrize(
    ("room_id", "channel_id"), [("r2", "agent"), ("r1", "other")], ids=["room", "channel"]
)
async def test_the_owner_refuses_a_call_naming_another_room_or_channel(
    room_id: str, channel_id: str
) -> None:
    served: list[str] = []
    answers: list[Any] = []

    async def serve(arguments: dict[str, Any]) -> str:
        served.append("ran")
        return "ran"

    async def send(message: Any) -> None:
        answers.append(message)

    store = OwnerToolStore({}.get)
    store.of("agent").register(orchestration_tool(PROBE, serve), room_id="r1", owner="s")
    calls = OwnerCalls(_request(), RegistryOwnerTools(store), None, send, set())
    calls.start(
        ToolCall(
            call_id="x",
            name="probe",
            arguments={},
            context=CallContext(tool_call_id="c1", room_id=room_id, channel_id=channel_id),
        ),
        None,
    )
    while not answers:
        await asyncio.sleep(0)

    [answer] = answers
    assert answer.outcome == "unserved"
    assert served == []
