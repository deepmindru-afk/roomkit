"""An emergency compaction keeps the turn's input and its notes whole (RFC §6.4, RMK-335).

A context overflow late in a long tool loop used to summarize the first half
of the messages, the turn's input among them: the participant's question and
the notes it carries (the plan, the tools already used) came back cut to 500
characters for the rest of the turn. The history before the input is what gets
summarized now, and the older rounds' long results are stored for re-reading.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels._compaction import SUMMARY_HEADER, summary_text
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    AIToolCall,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_LOOKUP = AITool(name="lookup", description="Look up a quote", parameters={})
_QUESTION = "Compare the three quotes and tell me which one to take."
_ROUNDS = 6


class _Overflowing(MockAIProvider):
    """Refuses its call number *at* as too long, once, and keeps a copy of
    the messages each call was given (the loop mutates them in place)."""

    def __init__(self, responses: list[AIResponse], *, streaming: bool, at: int) -> None:
        super().__init__(ai_responses=responses, streaming=streaming)
        self._at = at
        self._count = 0
        self.seen: list[list[AIMessage]] = []

    def _check(self, context: AIContext) -> None:
        self._count += 1
        self.seen.append([m.model_copy(deep=True) for m in context.messages])
        if self._count == self._at:
            raise ProviderError("prompt is too long", retryable=False, context_overflow=True)

    async def generate(self, context: AIContext) -> AIResponse:
        # The mock's stream goes through generate(): one check per call, both modes.
        self._check(context)
        return await super().generate(context)


def _round(call_id: str, name: str = "lookup", **arguments: Any) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments)],
    )


_DONE = AIResponse(content="done", finish_reason="stop")


async def _lookup(name: str, arguments: dict[str, Any]) -> str:
    if arguments.get("q") == "status":
        return '{"status": "SHIPPED-7731"}'
    return f"quote {arguments.get('n')} " + "x" * 3000


def _script() -> list[AIResponse]:
    """Turn 1 looks a status up (the digest of turn 2 quotes it); turn 2 runs
    six lookups, overflows, reads the first one back, and answers."""
    return [
        _round("c0", q="status"),
        _DONE,
        *(_round(f"c{n}", n=n) for n in range(1, _ROUNDS + 1)),
        _round("r1", "read_stored_result", result_id="evicted_c1"),
        _DONE,
    ]


# Turn 1 makes two calls; turn 2's call after its last round overflows.
_OVERFLOW_AT = 2 + _ROUNDS + 1


async def _turn(ch: AIChannel, body: str, history: list[RoomEvent] | None = None) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_LOOKUP.model_dump()]},
    )
    return await respond(
        ch,
        make_event(room_id="r1", body=body, channel_id="sms1"),
        binding,
        RoomContext(room=Room(id="r1"), recent_events=history or []),
    )


def _history(pairs: int) -> list[RoomEvent]:
    events: list[RoomEvent] = []
    for i in range(pairs):
        events.append(make_event(room_id="r1", body=f"earlier question {i}", channel_id="sms1"))
        events.append(
            make_event(
                room_id="r1",
                body=f"earlier answer {i}",
                channel_id="ai1",
                channel_type=ChannelType.AI,
            )
        )
    return events


def _results(messages: list[AIMessage]) -> dict[str, str]:
    return {
        part.tool_call_id: str(part.result)
        for message in messages
        if message.role == "tool" and isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
    }


def _no_two_user_messages_in_a_row(messages: list[AIMessage]) -> bool:
    return all(
        not (a.role == "user" and b.role == "user")
        for a, b in zip(messages, messages[1:], strict=False)
    )


async def _compacted_turn(streaming: bool, history: list[RoomEvent]) -> tuple[Any, LoopRun]:
    provider = _Overflowing(_script(), streaming=streaming, at=_OVERFLOW_AT)
    ch = AIChannel("ai1", provider=provider, tool_handler=_lookup)
    await _turn(ch, "what is the status?")
    run = await _turn(ch, _QUESTION, history)
    return provider, run


async def test_a_long_loop_keeps_its_input_and_notes_and_stores_older_results(
    streaming: bool,
) -> None:
    provider, run = await _compacted_turn(streaming, history=[])

    before, replay = provider.seen[_OVERFLOW_AT - 1], provider.seen[_OVERFLOW_AT]
    assert replay[0] == before[0]  # the input and its notes, as the first round built them
    assert str(replay[0].content).startswith(_QUESTION)
    assert "SHIPPED-7731" in str(replay[0].content)  # the digest of turn 1
    results = _results(replay)
    assert all("Full output saved as 'evicted_c" in results[f"c{n}"] for n in (1, 2, 3))
    assert all(results[f"c{n}"].endswith("x" * 3000) for n in (4, 5, 6))
    assert _no_two_user_messages_in_a_row(replay)
    assert run.calls[-1].name == "read_stored_result" and not run.calls[-1].failed
    assert "quote 1 xxx" in str(run.calls[-1].result)
    assert run.text == "done"


async def test_the_history_before_the_input_is_summarized_into_it(streaming: bool) -> None:
    provider, _ = await _compacted_turn(streaming, history=_history(2))

    before, replay = provider.seen[_OVERFLOW_AT - 1], provider.seen[_OVERFLOW_AT]
    input_before = next(m for m in before if _QUESTION in str(m.content))
    first = str(replay[0].content)
    assert first.startswith(SUMMARY_HEADER)
    assert "earlier answer 0" in first
    # The input follows the summary whole, its notes included.
    assert first.endswith(str(input_before.content))
    assert _no_two_user_messages_in_a_row(replay)
    assert "Full output saved as 'evicted_c1'" in _results(replay)["c1"]


def test_a_summary_never_leaves_a_delimited_result_open() -> None:
    text = "see <tool_result>\n" + "y" * 900 + "\n</tool_result> and <tool_result>\nz"
    messages = [AIMessage(role="user", content=text)]

    summary = summary_text(messages) or ""

    assert "<tool_result>" not in summary
    assert summary.count("[tool_result]") == 2


async def test_nothing_left_to_compact_before_the_input_is_refused() -> None:
    ch = AIChannel("ai1", provider=MockAIProvider())
    loop_ctx = ch._get_loop_ctx()
    messages = [AIMessage(role="user", content="q")]
    messages += [AIMessage(role="assistant", content=f"a{i}") for i in range(4)]
    loop_ctx.turn_input = messages[0]
    token = _set_loop_ctx(loop_ctx)
    try:
        with pytest.raises(ProviderError, match="nothing left to compact"):
            await ch._compact_context(AIContext(messages=messages))
    finally:
        _reset_loop_ctx(token)


def _set_loop_ctx(loop_ctx: Any) -> Any:
    from roomkit.channels.ai import _current_loop_ctx

    return _current_loop_ctx.set(loop_ctx)


def _reset_loop_ctx(token: Any) -> None:
    from roomkit.channels.ai import _current_loop_ctx

    _current_loop_ctx.reset(token)
