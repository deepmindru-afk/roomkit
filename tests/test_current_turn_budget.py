"""The current turn occupies the window even though it is not history."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.memory import BudgetAwareMemory, MockMemoryProvider
from roomkit.memory.token_estimator import (
    estimate_event_tokens,
    estimate_message_tokens,
    history_budget,
)
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIMessage, AITool, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event


def test_current_turn_joins_the_existing_non_history_reserve() -> None:
    current = make_event(body="x" * 2000)
    summary = AIMessage(role="user", content="summary" * 100)
    assert history_budget(
        max_context_tokens=4000,
        reserved_tokens=1000,
        messages=[summary],
        current_event=current,
    ) == 3400 - 1000 - estimate_message_tokens(summary) - estimate_event_tokens(current)
    assert history_budget(max_context_tokens=100, current_event=current) == 0


async def test_larger_current_turn_leaves_less_history_and_is_never_trimmed() -> None:
    events = [make_event(body="history" * 100) for _ in range(20)]
    block = AIMessage(role="user", content="summary" * 100)
    memory = BudgetAwareMemory(
        MockMemoryProvider(events=events, messages=[block]),
        max_context_tokens=4000,
        reserved_tokens=300,
        min_events=0,
    )
    ctx = RoomContext(room=Room(id="test-room"))
    short, long = make_event(body="small"), make_event(body="x" * 8000)
    before = long.model_dump()
    small, large, small_again = await asyncio.gather(
        memory.retrieve("test-room", short, ctx),
        memory.retrieve("test-room", long, ctx),
        memory.retrieve("test-room", short, ctx),
    )
    assert 0 < len(large.events) < len(small.events)
    assert small.events == small_again.events
    assert long.model_dump() == before
    assert large.events == events[-len(large.events) :]
    assert large.messages == [block]
    carried = (
        300
        + estimate_message_tokens(block)
        + estimate_event_tokens(long)
        + sum(estimate_event_tokens(e) for e in large.events)
    )
    assert carried <= 3400


async def test_message_beyond_window_refuses_before_inner_retrieval() -> None:
    inner = MockMemoryProvider()
    inner.retrieve = AsyncMock()
    memory = BudgetAwareMemory(inner, max_context_tokens=40000)
    with pytest.raises(ProviderError, match="Shorten the message") as failed:
        await memory.retrieve(
            "test-room", make_event(body="x" * 200000), RoomContext(room=Room(id="test-room"))
        )
    assert failed.value.context_overflow is True
    assert failed.value.retryable is False
    assert failed.value.status_code is None  # No HTTP request occurred.
    inner.retrieve.assert_not_awaited()


@pytest.mark.parametrize("window", [0, -1, 100])
async def test_unknown_window_or_exact_boundary_does_not_reject(window: int) -> None:
    # Exactly 100 estimated tokens; unknown windows preserve existing behavior.
    current = make_event(body="x" * 396)
    memory = BudgetAwareMemory(MockMemoryProvider(), max_context_tokens=window)
    result = await memory.retrieve("test-room", current, RoomContext(room=Room(id="test-room")))
    assert result.events == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tools", [False, True])
async def test_every_generation_path_refuses_without_calling_provider(
    streaming: bool, tools: bool
) -> None:
    provider = MockAIProvider(streaming=streaming)
    channel = AIChannel(
        "ai",
        provider=provider,
        memory=BudgetAwareMemory(MockMemoryProvider(), max_context_tokens=40000),
        tools=[AITool(name="search", description="Search")] if tools else None,
    )
    ctx = RoomContext(room=Room(id="test-room"))
    binding = ChannelBinding(room_id="test-room", channel_id="ai", channel_type=ChannelType.AI)
    with pytest.raises(ProviderError) as failed:
        output = await channel.on_event(make_event(body="x" * 200000), binding, ctx)
        if output.response_stream is not None:
            async for _ in output.response_stream:
                pass
    assert failed.value.context_overflow is True
    assert provider.calls == []
    await channel.close()
