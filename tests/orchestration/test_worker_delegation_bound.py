"""A supervisor's worker delegation, bounded and followed, on every door that
waits for it (RFC §19.7.3, §23.3; RMK-478).

Past its task timeout the worker's delegation is cut and ends cancelled; the
call that delegated answers once, with the worker read as failed, its tool
loop left as it was by the cut turn of the worker that ran inline under it.
However the delegation ends, the worker's last status entry is terminal.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, ToolCallEvent
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.orchestration.status_bus import StatusLevel
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
    "per-worker-wait": ("delegate_to_worker", {"wait_for_result": True}),
}
EVERY_DOOR = pytest.mark.parametrize("door", list(DOORS))


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


@dataclass
class _Turn:
    """What one supervisor turn left: the worker's task ends, the delegating
    call's reports, the worker's status entries, and how long it took."""

    ends: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    entries: list[tuple[StatusLevel, str]] = field(default_factory=list)
    elapsed: float = 0.0


async def _run(door: str, *, bound: float = _BOUND, call_bound: float | None = None) -> _Turn:
    """One supervisor turn through *door*, its workers bounded by *bound*, its
    delegating call by *call_bound* when given."""
    tool, settings = DOORS[door]
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    call_bounds = {tool: call_bound} if call_bound is not None else None
    supervisor = Agent("sup", provider=_CallsOnce(tool), tool_timeouts=call_bounds)
    worker = Agent(
        "worker",
        provider=_CallsOnce("lookup"),
        tools=[_LOOKUP],
        tool_handler=_slow_lookup,
        tool_search=False,
    )
    kit.register_channel(supervisor)
    strategy = Supervisor(supervisor=supervisor, workers=[worker], task_timeout=bound, **settings)
    await kit.create_room(room_id="r", orchestration=strategy)
    await kit.attach_channel("r", "sms")
    turn = _Turn()

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def _ended(event: Any, ctx: Any) -> None:
        if event.metadata["agent_id"] == "worker":
            turn.ends.append(str(event.metadata["task_status"]))

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def _reported(event: ToolCallEvent, ctx: Any) -> None:
        if event.name == tool:
            turn.reports.append(str(event.result))

    started = time.monotonic()
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Find it."))
    )
    turn.elapsed = time.monotonic() - started
    await asyncio.sleep(0.05)
    entries = await kit.status_bus.recent(20, agent_id="worker")
    turn.entries = [(entry.status, entry.detail) for entry in entries]
    await kit.close()
    return turn


@EVERY_DOOR
async def test_a_worker_past_its_bound_is_cut_and_the_call_answers_once(door: str) -> None:
    turn = await _run(door)

    assert turn.ends == ["cancelled"]
    assert turn.elapsed < _WORK
    assert len(turn.reports) == 1
    assert "Tool call cancelled" not in turn.reports[0]
    assert turn.entries[-1] == (StatusLevel.FAILED, "The task timed out after 0.3s.")


@EVERY_DOOR
async def test_a_delegation_its_call_cut_posts_the_worker_failed(door: str) -> None:
    turn = await _run(door, bound=30.0, call_bound=_BOUND)

    assert turn.ends == ["cancelled"]
    assert turn.entries[-1] == (StatusLevel.FAILED, "cancelled")
