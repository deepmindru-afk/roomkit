"""An agent cancels a task of its room with ``cancel_task``, and is not told twice
(RMK-549; RFC §23.3, §23.4).

A request the person changed or dropped ("the weather in Québec… no, Montréal")
must be able to stop its work: the task ends ``cancelled`` as any task cancelled
from outside, and the agent that cancelled it is not handed the cancellation
back, which would have it say so twice. A host's ``kit.cancel_task`` still tells
the agent, which did not ask for it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import ToolNameCollisionError, UnservedToolCallError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, TaskStatus
from roomkit.models.event import TextContent
from roomkit.orchestration._worker_run import WorkerStatus, follow_worker
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks import CANCEL_TASK_TOOL, CancelTaskTool
from roomkit.tasks.handback import CALLER_HANDS_BACK
from roomkit.tasks.status import TASK_STATUS_TOOL
from roomkit.tools.context import ToolCallContext, tool_turn_context
from roomkit.turns._registry_tools import RegistryOwnerTools
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel


class _Endless(MockAIProvider):
    """A worker that never finishes on its own."""

    def __init__(self) -> None:
        super().__init__(responses=["never"])
        self.started = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _Room:
    """Room ``r``: a transport, the speaking agent, a worker; what is handed back
    and which tasks ended."""

    def __init__(self, worker: MockAIProvider) -> None:
        self.kit = RoomKit()
        self.worker = worker
        self.handed_back: list[dict[str, Any]] = []
        self.ended: list[dict[str, Any]] = []

    async def open(self) -> _Room:
        kit = self.kit
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(Agent("speaker", provider=MockAIProvider(responses=["Voilà."])))
        kit.register_channel(Agent("worker", provider=self.worker))
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "sms")
        await kit.attach_channel("r", "speaker", category=ChannelCategory.INTELLIGENCE)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
        async def hand_back(event: Any, ctx: Any) -> HookResult:
            self.handed_back.append(dict(event.metadata))
            return HookResult.allow()

        @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
        async def completed(event: Any, ctx: Any) -> None:
            self.ended.append(dict(event.metadata))

        return self

    async def running_task(self, room_id: str = "r") -> Any:
        task = await self.kit.delegate(
            room_id, "worker", "Compte jusqu'à mille.", notify="speaker"
        )
        await asyncio.wait_for(self.worker.started.wait(), 5)
        return task


def _call(channel_id: str = "speaker") -> ToolCallContext:
    return ToolCallContext(room_id="r", tool_call_id="call-1", channel_id=channel_id)


async def _cancel(tool: CancelTaskTool, task_id: str, *, channel_id: str = "speaker") -> Any:
    with tool_turn_context(room_id="r", call=_call(channel_id)):
        return json.loads(await tool.handler(CANCEL_TASK_TOOL, {"task_id": task_id}))


async def test_the_agent_cancels_a_running_task_and_is_not_told_twice() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    result = await task.wait(timeout=5)
    await until(lambda: bool(room.ended), timeout=5)
    await asyncio.sleep(0.1)  # a hand-back would be under way by now
    entries = [e for e in await room.kit.status_bus.recent(50) if e.action == "task"]
    await room.kit.close()

    assert answer == {
        "task_id": task.id,
        "status": "cancelled",
        "message": "Cancelled: its result will not come back.",
    }
    assert result.status == TaskStatus.CANCELLED
    assert room.ended[0]["task_status"] == TaskStatus.CANCELLED
    assert entries[-1].metadata["task_status"] == "cancelled"
    assert room.handed_back == []  # the tool's answer told it


async def test_a_task_the_host_cancels_is_still_handed_back_to_the_agent() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    assert await room.kit.cancel_task(task.id) is True
    await until(lambda: bool(room.handed_back), timeout=5)
    await room.kit.close()

    assert room.handed_back[0]["task_id"] == task.id
    assert room.handed_back[0]["task_status"] == "cancelled"


async def test_a_cancel_by_another_agent_still_tells_the_notified_one() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    answer = await _cancel(CancelTaskTool(room.kit), task.id, channel_id="assistant-bis")
    await until(lambda: bool(room.handed_back), timeout=5)
    await room.kit.close()

    assert answer["status"] == "cancelled"
    assert room.handed_back[0]["task_status"] == "cancelled"


async def test_another_rooms_task_is_out_of_reach() -> None:
    room = await _Room(_Endless()).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    theirs = await room.running_task("other")

    answer = await _cancel(CancelTaskTool(room.kit), theirs.id)
    still_running = theirs.result is None
    await room.kit.close()

    assert answer == {
        "task_id": theirs.id,
        "status": "unknown",
        "message": f"No task {theirs.id} here.",
    }
    assert still_running


async def test_a_task_that_already_ended_is_left_as_it_stands() -> None:
    room = await _Room(MockAIProvider(responses=["Mille."])).open()
    task = await room.kit.delegate("r", "worker", "Compte jusqu'à mille.", notify="speaker")
    await task.wait(timeout=5)
    await until(lambda: bool(room.ended), timeout=5)

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    await room.kit.close()

    assert answer == {
        "task_id": task.id,
        "status": "completed",
        "message": "The task had already ended.",
    }
    assert task.result is not None and task.result.status == TaskStatus.COMPLETED


async def test_a_call_without_a_task_or_outside_a_conversation_is_refused() -> None:
    kit = RoomKit()
    tool = CancelTaskTool(kit)
    with tool_turn_context(room_id="r"):
        missing = json.loads(await tool.handler(CANCEL_TASK_TOOL, {}))
    outside = json.loads(await tool.handler(CANCEL_TASK_TOOL, {"task_id": "t"}))
    await kit.close()

    assert missing == {"error": "task_id is required"}
    assert "error" in outside


async def test_cancel_task_declines_another_tools_call() -> None:
    kit = RoomKit()
    with pytest.raises(UnservedToolCallError):
        await CancelTaskTool(kit).handler("task_status", {})
    await kit.close()


def test_the_definition_asks_for_the_task_id() -> None:
    definition = CancelTaskTool(RoomKit()).definition
    assert definition["name"] == CANCEL_TASK_TOOL
    assert definition["parameters"]["required"] == ["task_id"]


async def test_a_forged_status_line_reaches_no_other_rooms_task() -> None:
    """The StatusBus is not what decides: a ``task`` line for room ``r`` naming
    another room's task, posted by any code holding the kit, cancels nothing."""
    room = await _Room(_Endless()).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    theirs = await room.running_task("other")
    room.kit.status_bus.post(
        "worker",
        "task",
        "pending",
        metadata={"room_id": "r", "task_id": theirs.id, "task_status": "running"},
    )
    await asyncio.sleep(0)

    answer = await _cancel(CancelTaskTool(room.kit), theirs.id)
    await asyncio.sleep(0.05)
    still_running = theirs.result is None
    await room.kit.close()

    assert answer["status"] == "unknown"
    assert still_running


async def test_the_owner_serves_the_task_tools_set_up_on_an_agent() -> None:
    """``setup_task_tools`` makes them the room owner's: listed in a turn's
    request, served on the owner for a turn on any executor."""
    room = await _Room(_Endless()).open()
    speaker = room.kit.channels["speaker"]
    room.kit.setup_task_tools(speaker)  # type: ignore[arg-type]
    task = await room.running_task()

    declared = room.kit._turns._tools.declared("speaker", "r")
    names = sorted(tool.tool.name for tool in declared)
    with tool_turn_context(room_id="r", call=_call()):
        entry = speaker._registry.owner_lookup(CANCEL_TASK_TOOL, "r")  # type: ignore[attr-defined]
        assert entry is not None
        answer = json.loads(await entry.serve({"task_id": task.id}))
    await room.kit.close()

    assert names == ["cancel_task", "task_status"]
    assert answer["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="already"):
        room.kit.setup_task_tools(speaker)  # type: ignore[arg-type]


async def test_a_task_whose_end_is_being_told_reads_ended_not_unknown() -> None:
    """While its ON_TASK_COMPLETED hooks run, the room's own task is still the
    room's: cancel_task says it had already ended (RFC §23.4)."""
    room = await _Room(MockAIProvider(responses=["Mille."])).open()
    in_hook = asyncio.Event()
    release = asyncio.Event()

    @room.kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def slow(event: Any, ctx: Any) -> None:
        in_hook.set()
        await release.wait()

    task = await room.kit.delegate("r", "worker", "Compte.", notify="speaker")
    await asyncio.wait_for(in_hook.wait(), 5)
    cancelling = asyncio.ensure_future(_cancel(CancelTaskTool(room.kit), task.id))
    await asyncio.sleep(0.05)
    release.set()
    answer = await asyncio.wait_for(cancelling, 5)
    await room.kit.close()

    assert answer["status"] == "completed"
    assert room.kit._task_rooms == {}


async def test_setup_task_tools_sets_up_both_tools_or_neither() -> None:
    kit = RoomKit()
    agent = Agent("speaker", provider=MockAIProvider(responses=["x"]), tools=[CancelTaskTool(kit)])
    kit.register_channel(agent)

    with pytest.raises(ToolNameCollisionError):
        kit.setup_task_tools(agent)

    assert agent._registry.lookup(TASK_STATUS_TOOL, None) is None
    await kit.close()


async def test_a_strategy_s_unposted_worker_is_never_a_room_s_to_cancel() -> None:
    """A task a strategy runs without posting it (``post_status=False``) is
    the strategy's to follow, though its follower posts lines (RFC §23.4)."""
    room = await _Room(_Endless()).open()
    task = await room.kit.delegate(
        "r", "worker", "Compte.", notify=CALLER_HANDS_BACK, post_status=False
    )
    follower = asyncio.ensure_future(
        follow_worker(
            room.kit,
            task,
            timeout=None,
            status=WorkerStatus({"room_id": "r", "mode": "per_worker_async"}),
        )
    )
    await asyncio.wait_for(room.worker.started.wait(), 5)

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    still_running = task.result is None
    await room.kit.cancel_task(task.id)
    await asyncio.wait_for(follower, 5)
    await room.kit.close()

    assert answer["status"] == "unknown"
    assert still_running


async def test_a_cancel_with_no_tool_call_is_held_to_the_room_too() -> None:
    room = await _Room(_Endless()).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    theirs = await room.running_task("other")
    room.kit.status_bus.post(
        "worker", "task", "pending", metadata={"room_id": "r", "task_id": theirs.id}
    )
    await asyncio.sleep(0)

    with tool_turn_context(room_id="r"):
        answer = json.loads(
            await CancelTaskTool(room.kit).handler(CANCEL_TASK_TOOL, {"task_id": theirs.id})
        )
    still_running = theirs.result is None
    await room.kit.close()

    assert answer["status"] == "unknown" and still_running


async def test_a_turn_s_cancel_task_is_served_by_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[str] = []
    serve = RegistryOwnerTools.serve

    async def spy(self: RegistryOwnerTools, request: Any, call: Any) -> Any:
        task = asyncio.current_task()
        served.append(f"{call.name}@{task.get_name() if task else ''}")
        return await serve(self, request, call)

    monkeypatch.setattr(RegistryOwnerTools, "serve", spy)
    room = await _Room(_Endless()).open()
    speaker = room.kit.channels["speaker"]
    room.kit.setup_task_tools(speaker)  # type: ignore[arg-type]
    task = await room.running_task()
    call = AIToolCall(id="c1", name=CANCEL_TASK_TOOL, arguments={"task_id": task.id})
    speaker._provider._ai_responses = [  # type: ignore[attr-defined]
        AIResponse(content="", tool_calls=[call]),
        AIResponse(content="Annulé."),
    ]
    speaker._provider._index = 0  # type: ignore[attr-defined]

    await room.kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Annule.")),
        room_id="r",
    )
    result = await task.wait(timeout=5)
    await asyncio.sleep(0.2)
    await room.kit.close()

    assert result.status == TaskStatus.CANCELLED
    assert any(s.startswith("cancel_task@roomkit-owner-call-") for s in served)
    assert room.handed_back == []


async def test_a_submit_that_fails_leaves_no_task_of_the_room() -> None:
    room = await _Room(_Endless()).open()

    async def down(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("down")

    room.kit._task_runner.submit = down  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await room.kit.delegate("r", "worker", "Compte.", notify="speaker")
    left = dict(room.kit._task_rooms)
    await room.kit.close()

    assert left == {}
