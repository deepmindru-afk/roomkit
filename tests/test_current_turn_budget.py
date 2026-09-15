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
from roomkit.models.event import CompositeContent, MediaContent, TextContent
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


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("composite", [False, True])
async def test_image_bytes_do_not_cost_text_tokens_or_reject_a_valid_turn(
    inline: bool, composite: bool
) -> None:
    url = "data:image/png;base64," + "a" * 400000 if inline else "https://example.test/image.png"
    media = MediaContent(url=url, mime_type="image/png", caption="A small image")
    content = CompositeContent(parts=[TextContent(body="Describe"), media]) if composite else media
    current = make_event().model_copy(update={"content": content})
    history = [make_event(body="history" * 100) for _ in range(10)]
    memory = BudgetAwareMemory(MockMemoryProvider(events=history), max_context_tokens=4000)
    provider = MockAIProvider(vision=True)
    channel = AIChannel("ai", provider=provider, memory=memory)
    binding = ChannelBinding(room_id="test-room", channel_id="ai", channel_type=ChannelType.AI)
    ctx = RoomContext(room=Room(id="test-room"), bindings=[binding])
    # Keep the history through the real channel's visibility gate.
    await channel.on_event(current, binding, ctx)
    assert len(provider.calls) == 1
    assert estimate_event_tokens(current) < 1100
    assert any("historyhistory" in str(m.content) for m in provider.calls[0].messages)
    assert url in str(provider.calls[0].messages[-1].content)
    await channel.close()


async def test_approximate_image_cost_cannot_refuse_a_small_window() -> None:
    media = MediaContent(url="https://example.test/image.png", mime_type="image/png")
    current = make_event().model_copy(update={"content": media})
    assert estimate_event_tokens(current) > 100
    assert estimate_event_tokens(current, text_only=True) == 0
    memory = BudgetAwareMemory(MockMemoryProvider(), max_context_tokens=100)
    await memory.retrieve("test-room", current, RoomContext(room=Room(id="test-room")))


async def test_oversized_text_is_refused_even_when_an_image_is_attached() -> None:
    current = make_event().model_copy(
        update={
            "content": CompositeContent(
                parts=[
                    TextContent(body="x" * 200000),
                    MediaContent(url="https://example.test/image.png", mime_type="image/png"),
                ]
            )
        }
    )
    memory = BudgetAwareMemory(MockMemoryProvider(), max_context_tokens=40000)
    with pytest.raises(ProviderError):
        await memory.retrieve("test-room", current, RoomContext(room=Room(id="test-room")))
