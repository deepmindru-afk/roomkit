"""A supervisor's background run ends before the supervisor hears of it
(RMK-451, RFC §19.7.3, §23.3).

The supervisor is told its workers' results, or that they failed, once the
room is released: a dispatch it makes in answer starts a new run instead of
reading the run that ended's ``dispatched`` answer, and a supervisor that
dispatches again on every outcome stops at ``max_chain_depth``. The status bus
carries one terminal entry per run.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.channels.agent import Agent
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.orchestration.strategies.supervisor import delegate as supervisor_delegate
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _DispatchesOnEveryOutcome(MockAIProvider):
    """A supervisor that dispatches its workers again on every hand-back."""

    def __init__(self) -> None:
        super().__init__()
        self.told: list[str] = []
        self.dispatches = 0

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        last = context.messages[-1]
        if isinstance(last.content, list) and any(
            getattr(part, "type", None) == "tool_result" for part in last.content
        ):
            return AIResponse(content="On it.", finish_reason="stop")
        self.told.append(str(last.content))
        self.dispatches += 1
        call = AIToolCall(
            id=f"c{self.dispatches}",
            name="delegate_workers",
            arguments={"task": f"t{self.dispatches}"},
        )
        return AIResponse(content="", finish_reason="tool_calls", tool_calls=[call])


async def _run(
    monkeypatch: pytest.MonkeyPatch, *, fails: bool
) -> tuple[list[str], _DispatchesOnEveryOutcome, list[Any]]:
    runs: list[str] = []

    async def run_workers(
        kit: Any, room_id: str, strategy: Any, workers: Any, task: str, **_: Any
    ) -> Any:
        runs.append(task)
        await asyncio.sleep(0.01)
        if fails:
            raise RuntimeError("connection refused: postgres://admin:secret@db")
        return [{"worker": "w1", "output": f"Done {task}", "completed": True}]

    monkeypatch.setattr(supervisor_delegate, "_run_workers", run_workers)
    kit = RoomKit(max_chain_depth=5)
    provider = _DispatchesOnEveryOutcome()
    supervisor = Supervisor(
        Agent("boss", provider=provider, tool_search=False),
        [Agent("w1", provider=MockAIProvider(responses=["x"]))],
        strategy="parallel",
        async_delivery=True,
    )
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1", orchestration=supervisor)
    await kit.attach_channel("r1", "sms1")
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="analyse X"))
    )
    entries: list[Any] = []
    for _ in range(300):
        entries = await kit.status_bus.recent(50, agent_id="orchestration")
        if len(entries) == 4 and len(provider.calls) >= 8:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    entries = await kit.status_bus.recent(50, agent_id="orchestration")
    await kit.close()
    return runs, provider, entries


async def test_a_dispatch_answering_a_failure_runs_until_the_chain_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs, provider, entries = await _run(monkeypatch, fails=True)

    # The first run, then one per failure handed back, until max_chain_depth.
    assert runs == ["t1", "t2", "t3", "t4"]
    failures = [t for t in provider.told if "workers failed" in t]
    assert len(failures) == 3 and not any("secret" in t for t in provider.told)
    assert [e.status for e in entries] == [StatusLevel.FAILED] * 4


async def test_a_dispatch_answering_results_runs_rather_than_reading_the_last_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs, provider, entries = await _run(monkeypatch, fails=False)

    assert runs == ["t1", "t2", "t3", "t4"]
    assert sum("workers completed" in t for t in provider.told) == 3
    # The last run's results reach nobody (the chain bound refuses them): its
    # one terminal entry says so rather than reporting them handed back.
    statuses = [e.status for e in entries]
    assert sorted(statuses) == sorted([StatusLevel.COMPLETED] * 3 + [StatusLevel.FAILED])
    assert [e.detail for e in entries if e.status == StatusLevel.FAILED][0].startswith(
        "not handed back"
    )
