"""``delegate_task``'s cache answers a repeat of a call only once its task
completed (RMK-462 family, RFC §23.3).

A failed or cancelled task is run again on a repeat, as a strategy's
dispatch is: a cached ``delegated`` answer would promise a result that
never comes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.enums import ChannelCategory
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.cache import CompletedTaskCache
from roomkit.tasks.delegate import DelegateHandler
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})


class _Loops(MockAIProvider):
    """Calls its tool every round: a round cap of 1 cuts the task."""

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        return AIResponse(
            content="Still checking.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=f"c{len(self.calls)}", name="lookup", arguments={})],
        )


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _slow(name: str, arguments: dict[str, Any]) -> str:
    await asyncio.sleep(5)
    return "found"


async def _kit(worker: Agent) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("boss", provider=MockAIProvider(responses=["ok"])))
    kit.register_channel(worker)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "boss", category=ChannelCategory.INTELLIGENCE)
    return kit


async def _remembered(handler: DelegateHandler) -> None:
    """Until the handler has kept what its task's end gives the cache, or
    left it out: its watch on the task has ended."""
    await until(lambda: not handler._background_tasks)


@pytest.mark.parametrize("ending", ["failed", "cancelled"])
async def test_a_task_that_did_not_complete_is_run_again(ending: str) -> None:
    provider = _Loops()
    worker = Agent(
        "worker",
        provider=provider,
        tools=[LOOKUP],
        tool_handler=_found if ending == "failed" else _slow,
        tool_search=False,
        max_tool_rounds=1,
    )
    kit = await _kit(worker)
    handler = DelegateHandler(kit, cache=CompletedTaskCache())
    arguments = {"agent": "worker", "task": "Find it."}

    first = await handler.handle("r", "boss", arguments)
    if ending == "cancelled":
        await until(lambda: bool(provider.calls))  # its slow tool runs
        await kit.task_runner.cancel(first["task_id"])
    await _remembered(handler)
    second = await handler.handle("r", "boss", arguments)
    await kit.close()

    assert second["task_id"] != first["task_id"]
    assert not second.get("from_cache", False)


async def test_a_completed_task_answers_a_repeat_from_the_cache() -> None:
    worker = Agent("worker", provider=MockAIProvider(responses=["Found it."]), tool_search=False)
    kit = await _kit(worker)
    handler = DelegateHandler(kit, cache=CompletedTaskCache())
    arguments = {"agent": "worker", "task": "Find it."}

    first = await handler.handle("r", "boss", arguments)
    await _remembered(handler)
    second = await handler.handle("r", "boss", arguments)
    await kit.close()

    assert second["task_id"] == first["task_id"] and second.get("from_cache") is True
    # The repeat gets what the task returned, not the receipt of its start.
    assert second["status"] == "completed"
    assert second["result"] == "Found it."
