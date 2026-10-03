"""An AI channel measures the turn's footprint before reading its memory (RFC §20).

What the history may take is what the window leaves once the rest of the turn
is in it. The channel measures that rest as round 0 sends it (the system prompt
with the agent's identity, the tools declared under Tool Search and the tool
policy, the reply budget), and a budget-aware memory reserves at least that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from roomkit.channels.agent import Agent
from roomkit.memory import (
    BudgetAwareMemory,
    MemoryProvider,
    MemoryResult,
    SlidingWindowMemory,
    current_turn_footprint,
)
from roomkit.memory.token_estimator import estimate_tokens, estimate_tool_tokens
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import SkillRegistry
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext
from roomkit.tools.policy import ToolPolicy
from tests.conftest import make_event
from tests.test_skills import _make_skill_dir_full
from tests.test_skills_integration import MockScriptExecutor
from tests.tool_loop_modes import respond


class _Measured(MemoryProvider):
    """Records the footprint the channel measured when it reads the room."""

    def __init__(self) -> None:
        self.footprints: list[int | None] = []

    async def retrieve(
        self, room_id: str, current_event: RoomEvent, context: RoomContext, **kwargs: Any
    ) -> MemoryResult:
        self.footprints.append(current_turn_footprint())
        return MemoryResult()


def _binding(**metadata: Any) -> ChannelBinding:
    return ChannelBinding(
        channel_id="agent",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata=metadata,
    )


async def test_the_footprint_is_the_turn_as_round_0_sends_it(
    tmp_path: Path, streaming: bool
) -> None:
    _make_skill_dir_full(tmp_path, "reports", scripts=["build.py"])
    skills = SkillRegistry()
    skills.discover(tmp_path)
    provider = MockAIProvider(responses=["ok"], streaming=streaming)
    memory = _Measured()
    agent = Agent(
        "agent",
        provider=provider,
        role="Billing advisor",
        system_prompt="Answer billing questions.",
        max_tokens=400,
        tools=[
            AITool(name=f"tool_{i}", description="looks things up " * 30, parameters={})
            for i in range(40)
        ],
        tool_handler=AsyncMock(return_value="ok"),
        tool_search=True,
        tool_search_pinned=["tool_0"],
        skills=skills,
        script_executor=MockScriptExecutor(),
        tool_policy=ToolPolicy(deny=["run_skill_script"]),
        memory=memory,
    )

    await respond(
        agent,
        make_event(body="hello", channel_id="member", room_id="r1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    sent = provider.calls[0]
    declared = [tool.name for tool in sent.tools or []]
    assert "Agent Identity" in (sent.system_prompt or "")
    assert "run_skill_script" not in declared and "tool_1" not in declared
    expected = (
        estimate_tokens(sent.system_prompt or "")
        + sum(estimate_tool_tokens(tool) for tool in sent.tools or [])
        + 400
    )
    assert memory.footprints == [expected]


async def test_outside_a_turn_there_is_no_footprint() -> None:
    assert current_turn_footprint() is None


async def test_a_budget_aware_memory_reserves_at_least_the_measured_footprint() -> None:
    history = [make_event(body="history " * 200, room_id="r1") for _ in range(30)]
    current = make_event(body="now", room_id="r1")
    context = RoomContext(room=Room(id="r1"), recent_events=[*history, current])

    async def kept(memory: BudgetAwareMemory, footprint: int | None) -> int:
        token = _current_loop_ctx.set(_ToolLoopContext(turn_footprint=footprint))
        try:
            return len((await memory.retrieve("r1", current, context)).events)
        finally:
            _current_loop_ctx.reset(token)

    def memory(reserved: int) -> BudgetAwareMemory:
        return BudgetAwareMemory(
            SlidingWindowMemory(max_events=100),
            max_context_tokens=20_000,
            reserved_tokens=reserved,
        )

    unmeasured = await kept(memory(0), None)
    measured = await kept(memory(0), 12_000)
    declared_larger = await kept(memory(12_000), 2_000)

    assert measured < unmeasured
    assert declared_larger == measured
