"""A supervisor's worker past its task timeout, on every door that waits for
it (RFC §19.7.3, §23.3).

The worker's delegation is cut at the bound and ends cancelled; the call that
delegated answers once, with the worker read as failed, its tool loop left as
it was by the cut turn of the worker that ran inline under it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, ToolCallEvent
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

_LOOKUP = AITool(name="lookup", description="Look it up.", parameters={"type": "object"})
_BOUND = 0.3
_WORK = 1.5

DOORS = {
    "parallel": ("delegate_workers", {"strategy": "parallel"}),
    "supervised-sequential": ("delegate_workers", {"strategy": "sequential"}),
}


class _CallsOnce(MockAIProvider):
    """Calls *tool* once, then answers."""

    def __init__(self, tool: str) -> None:
        super().__init__()
        self._tool = tool

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) > 1:
            return AIResponse(content="The answer.")
        call = AIToolCall(id=f"call-{self._tool}", name=self._tool, arguments={"task": "Find it."})
        return AIResponse(content="", finish_reason="tool_calls", tool_calls=[call])


async def _slow_lookup(name: str, arguments: dict[str, Any]) -> str:
    await asyncio.sleep(_WORK)
    return "found"


async def _run(door: str) -> tuple[list[str], list[str], float]:
    """One supervisor turn through *door*: the worker's task ends, what the
    delegating call reported, and how long the turn took."""
    tool, settings = DOORS[door]
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    supervisor = Agent("sup", provider=_CallsOnce(tool))
    worker = Agent(
        "worker",
        provider=_CallsOnce("lookup"),
        tools=[_LOOKUP],
        tool_handler=_slow_lookup,
        tool_search=False,
    )
    kit.register_channel(supervisor)
    strategy = Supervisor(supervisor=supervisor, workers=[worker], task_timeout=_BOUND, **settings)
    await kit.create_room(room_id="r", orchestration=strategy)
    await kit.attach_channel("r", "sms")
    ends: list[str] = []
    reports: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def _ended(event: Any, ctx: Any) -> None:
        if event.metadata["agent_id"] == "worker":
            ends.append(str(event.metadata["task_status"]))

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def _reported(event: ToolCallEvent, ctx: Any) -> None:
        if event.name == tool:
            reports.append(str(event.result))

    started = time.monotonic()
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Find it."))
    )
    elapsed = time.monotonic() - started
    await asyncio.sleep(0.05)
    await kit.close()
    return ends, reports, elapsed


@pytest.mark.parametrize("door", list(DOORS))
async def test_a_worker_past_its_bound_is_cut_and_the_call_answers_once(door: str) -> None:
    ends, reports, elapsed = await _run(door)

    assert ends == ["cancelled"]
    assert elapsed < _WORK
    assert len(reports) == 1
    assert "Tool call cancelled" not in reports[0]
