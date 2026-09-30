"""What a memory retrieves for one turn rides the turn's notes (RFC §20.2, RMK-334).

Knowledge passages change with every question, so they travel with the
turn's notes, after the input, each set apart as data, and the history before
the input stays the same, and cached, from one turn to the next. A memory that
wraps another carries them, and a memory that keeps to a token budget pays
for them.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.knowledge.base import KnowledgeResult, KnowledgeSource
from roomkit.memory import (
    BudgetAwareMemory,
    CompactingMemory,
    RetrievalMemory,
    SlidingWindowMemory,
    SummarizingMemory,
)
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.mock import MockMemoryProvider
from roomkit.memory.token_estimator import (
    estimate_event_tokens,
    estimate_message_tokens,
    estimate_notes_tokens,
)
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.orchestration.handoff import HandoffMemoryProvider
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond


class _Handbook(KnowledgeSource):
    """One passage, naming the question it was retrieved for."""

    async def search(
        self, query: str, *, room_id: str | None = None, limit: int = 5
    ) -> list[KnowledgeResult]:
        return [KnowledgeResult(content=f"passage for: {query}", score=1.0, source="handbook")]


def _history(count: int) -> list[RoomEvent]:
    """The channel's own earlier answers (without a binding, a channel's
    history holds only what it produced)."""
    return [
        make_event(room_id="r1", body=f"answer {n}", channel_id="ai1", channel_type=ChannelType.AI)
        for n in range(count)
    ]


async def _ask(ch: AIChannel, question: str, history: list[RoomEvent]) -> None:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    event = make_event(room_id="r1", body=question, channel_id="sms1")
    await respond(ch, event, binding, RoomContext(room=Room(id="r1"), recent_events=history))


def _text(context: AIContext) -> list[str]:
    return [str(m.content) for m in context.messages]


async def test_the_passages_follow_the_question_and_leave_the_history_alone(
    streaming: bool,
) -> None:
    provider = MockAIProvider(ai_responses=[AIResponse(content="ok")], streaming=streaming)
    memory = RetrievalMemory([_Handbook()], inner=SlidingWindowMemory())
    ch = AIChannel("ai1", provider=provider, memory=memory)

    await _ask(ch, "what about returns?", _history(2))
    await _ask(ch, "and gift cards?", _history(3))

    first, second = (_text(call) for call in provider.calls)
    # The history before the input is the same messages: the passages are not in it.
    assert first[:-1] == second[: len(first) - 1]
    assert not any("<knowledge>" in text for text in second[:-1])
    question = second[-1]
    assert question.startswith("and gift cards?")
    assert "<knowledge>\n[handbook] passage for: and gift cards?\n</knowledge>" in question


class _Inner(MemoryProvider):
    """A memory that retrieved a note for the turn, with some history."""

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        long = [
            make_event(room_id="r1", body=f"message {n} " + "word " * 300, channel_id="sms1")
            for n in range(12)
        ]
        return MemoryResult(events=long, notes=["NOTE"])


def _wrappers() -> list[Any]:
    summarizer = MockAIProvider(ai_responses=[AIResponse(content="summary")])
    return [
        BudgetAwareMemory(_Inner(), max_context_tokens=1500, min_events=1),
        CompactingMemory(_Inner(), provider=summarizer, max_context_tokens=1500, min_events=1),
        SummarizingMemory(_Inner(), provider=summarizer, max_context_tokens=1500, min_events=1),
    ]


@pytest.mark.parametrize("memory", _wrappers(), ids=["budget_aware", "compacting", "summarizing"])
async def test_a_memory_that_wraps_another_carries_its_notes(memory: MemoryProvider) -> None:
    event = make_event(room_id="r1", body="question", channel_id="sms1")

    result = await memory.retrieve("r1", event, RoomContext(room=Room(id="r1")))

    # It rebuilt the inner result (fewer events, or a summary), and kept the note.
    assert len(result.events) < 12 or result.messages
    assert result.notes == ["NOTE"]


async def test_retrieval_and_handoff_keep_the_inner_notes() -> None:
    event = make_event(room_id="r1", body="what about returns?", channel_id="sms1")
    context = RoomContext(room=Room(id="r1"))

    retrieved = await RetrievalMemory([_Handbook()], inner=_Inner()).retrieve("r1", event, context)
    handed = await HandoffMemoryProvider(_Inner()).retrieve("r1", event, context)

    assert retrieved.notes[0] == "NOTE" and "<knowledge>" in retrieved.notes[1]
    assert handed.notes == ["NOTE"]


def _budgeted(kind: str, inner: MemoryProvider, window: int) -> MemoryProvider:
    if kind == "budget_aware":
        return BudgetAwareMemory(inner, max_context_tokens=window, min_events=1)
    summarizer = MockAIProvider(ai_responses=[AIResponse(content="summary")])
    return SummarizingMemory(inner, provider=summarizer, max_context_tokens=window, min_events=1)


@pytest.mark.parametrize("kind", ["budget_aware", "summarizing"])
async def test_a_budget_pays_for_the_notes_it_carries(kind: str) -> None:
    """The notes ride the turn's input, where nothing trims them: a memory
    that keeps the turn to a window counts them before it keeps history."""
    window = 40_000
    events = [make_event(body="x" * 4_000) for _ in range(40)]
    inner = MockMemoryProvider(events=events, notes=["passage " * 6_000])
    memory = _budgeted(kind, inner, window)

    result = await memory.retrieve("r1", make_event(body="q"), RoomContext(room=Room(id="r1")))

    carried = (
        sum(estimate_message_tokens(m) for m in result.messages)
        + sum(estimate_event_tokens(e) for e in result.events)
        + estimate_notes_tokens(result.notes)
    )
    assert result.notes == inner._notes
    assert carried <= window
