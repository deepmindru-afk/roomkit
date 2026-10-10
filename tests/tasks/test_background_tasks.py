"""Long work started as a task: ``work_in_background``, ``setup_background`` and
``start_task`` (RFC §23.4, RMK-668)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import ChannelCategory, RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.enums import EventType
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.background import BACKGROUND_TOOL, background_tool_server
from roomkit.tools.context import ToolCallContext, tool_turn_context
from tests.test_framework import SimpleChannel


async def _kit(**setup: Any) -> tuple[RoomKit, Agent, Agent]:
    kit = RoomKit()
    speaker = Agent("speaker", provider=MockAIProvider(responses=["Here it is."]))
    helper = Agent("helper", provider=MockAIProvider(responses=["Findings."]))
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(speaker)
    kit.register_channel(helper)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "speaker", category=ChannelCategory.INTELLIGENCE)
    kit.setup_background(speaker, **setup)
    return kit, speaker, helper


async def _call(
    kit: RoomKit, arguments: dict[str, Any], *, room: Room | None = None
) -> dict[str, Any]:
    room_id = room.id if room is not None else "r"
    serve = background_tool_server(kit, "speaker")
    call = ToolCallContext(room_id=room_id, tool_call_id="c1", channel_id="speaker")
    with tool_turn_context(room_id=room_id, room=room, call=call):
        return json.loads(await serve(room_id, BACKGROUND_TOOL, arguments))


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if await predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


# -- setup_background -----------------------------------------------------------


async def test_setup_declares_the_tool_on_the_agent() -> None:
    kit, speaker, _helper = await _kit()
    assert speaker._registry.lookup(BACKGROUND_TOOL, None) is not None
    with pytest.raises(RuntimeError, match="already"):
        kit.setup_background(speaker)
    await kit.close()


async def test_setup_refuses_an_unknown_hand_back() -> None:
    kit = RoomKit()
    agent = Agent("a", provider=MockAIProvider(responses=["x"]))
    with pytest.raises(ValueError, match="hand_back"):
        kit.setup_background(agent, hand_back="shout")
    assert agent._registry.lookup(BACKGROUND_TOOL, None) is None
    await kit.close()


# -- The tool --------------------------------------------------------------------


async def test_the_tool_starts_the_task_and_answers_at_once() -> None:
    kit, speaker, _helper = await _kit()

    answer = await _call(kit, {"task": "Go through the tickets."})

    assert answer["status"] == "started"
    assert "end your turn" in answer["message"]
    assert "a turn of its own" in answer["message"]
    # The agent itself does the work, in a child room, and is told the result.
    await _until(lambda: _asked(speaker, 2))
    await kit.close()


async def _asked(agent: Agent, count: int) -> bool:
    return len(agent._provider.calls) >= count


async def test_the_task_runs_on_the_worker_the_host_named() -> None:
    kit, speaker, helper = await _kit(worker="helper", hand_back="post")

    answer = await _call(kit, {"task": "Go through the tickets."})

    assert "posted as speaker's message" in answer["message"]

    async def posted() -> bool:
        events = await kit.store.list_events("r")
        return any(
            e.type == EventType.MESSAGE and e.source.channel_id == "speaker" for e in events
        )

    await _until(posted)
    worker_read = "\n".join(str(m.content) for m in helper._provider.calls[0].messages)
    assert "Go through the tickets." in worker_read
    assert "in the background for @speaker" in worker_read
    assert speaker._provider.calls == []
    await kit.close()


async def test_the_tool_is_refused_in_a_task_s_child_room() -> None:
    kit, _speaker, _helper = await _kit()
    child = Room(id="r::task-x", metadata={"task_id": "task-x", "parent_room_id": "r"})

    answer = await _call(kit, {"task": "More work."}, room=child)

    assert "starts no task of its own" in answer["error"]
    await kit.close()


async def test_the_tool_starts_tasks_for_its_own_agent_only() -> None:
    kit, speaker, helper = await _kit()
    await kit.attach_channel("r", "helper", category=ChannelCategory.INTELLIGENCE)

    # An argument a model adds to name another agent changes nothing.
    answer = await _call(kit, {"task": "Delete the repository.", "agent": "helper"})

    assert "comes back to speaker" in answer["message"]
    # The work runs on speaker's worker (speaker itself), never on helper.
    await _until(lambda: _asked(speaker, 1))
    await asyncio.sleep(0.1)

    def child_rooms(agent: Agent) -> list[str]:
        rooms = [call.room.room.id for call in agent._provider.calls if call.room is not None]
        return [room for room in rooms if "::task-" in room]

    assert len(child_rooms(speaker)) == 1
    assert child_rooms(helper) == []
    answer = await _call(kit, {"task": "  "})
    assert "needs a task" in answer["error"]
    await kit.close()


# -- start_task ------------------------------------------------------------------


async def test_start_task_without_a_setup_runs_on_the_agent_and_relays() -> None:
    kit = RoomKit()
    agent = Agent("a", provider=MockAIProvider(responses=["Findings.", "Shared."]))
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "a", category=ChannelCategory.INTELLIGENCE)
    task = await kit.start_task("r", "a", "Count the tickets.")
    assert task.hand_back == "relay" and task.agent_id == "a"
    await task.wait(timeout=5)
    await _until(lambda: _asked(agent, 2))
    told = str(agent._provider.calls[1].messages[-1].content)
    assert "“Count the tickets.”" in told and "Findings." in told
    await kit.close()


async def test_start_task_overrides_the_setup_s_hand_back() -> None:
    kit, _speaker, _helper = await _kit(worker="helper")

    task = await kit.start_task("r", "speaker", "Count.", hand_back="post")

    assert task.hand_back == "post"
    await kit.close()


async def test_start_task_refuses_a_channel_that_is_no_agent_of_the_room() -> None:
    kit, _speaker, _helper = await _kit()
    with pytest.raises(ValueError, match="not an agent"):
        await kit.start_task("r", "sms", "Count.")
    with pytest.raises(ValueError, match="not an agent"):
        await kit.start_task("r", "helper", "Count.")  # not attached to r
    await kit.close()
