"""An emergency compaction keeps the turn's input and its notes whole (RFC §6.4, RMK-335).

A context overflow late in a long tool loop compacts the context once and
replays the round. The turn's input and the notes it carries (the plan, the
tools already used) stay whole: the history before the input is summarized,
and the older rounds' long results are stored for re-reading. Text the runtime
places next to a user message joins it, so no two user messages follow each
other.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import TURN_NOTES_HEADER, add_turn_note
from roomkit.channels._compaction import SUMMARY_HEADER, summary_text, with_results_stored
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._user_text import with_leading_text
from roomkit.channels.ai import AIChannel, _current_loop_ctx
from roomkit.core.hooks import SyncPipelineResult
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.fence import named_blocks
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_LOOKUP = AITool(name="lookup", description="Look up a quote", parameters={})
_QUESTION = "Compare the three quotes and tell me which one to take."
_ROUNDS = 6
_LONG = "x" * 3000


class _Overflowing(MockAIProvider):
    """Refuses its call number *at* as too long, once, and keeps a copy of
    the messages each call was given (the loop mutates them in place)."""

    def __init__(self, responses: list[AIResponse], *, streaming: bool, at: int) -> None:
        super().__init__(ai_responses=responses, streaming=streaming)
        self._at = at
        self._count = 0
        self.seen: list[list[AIMessage]] = []

    async def generate(self, context: AIContext) -> AIResponse:
        # The mock's stream goes through generate(): one check per call, both modes.
        self._count += 1
        self.seen.append([m.model_copy(deep=True) for m in context.messages])
        if self._count == self._at:
            raise ProviderError("prompt is too long", retryable=False, context_overflow=True)
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
    return f"quote {arguments.get('n')} {_LONG}"


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


def _history(count: int) -> list[RoomEvent]:
    """The channel's own earlier answers: without a binding, a channel's
    history holds only what it produced (RFC §7.5 rule 8)."""
    return [
        make_event(
            room_id="r1", body=f"earlier answer {i}", channel_id="ai1", channel_type=ChannelType.AI
        )
        for i in range(count)
    ]


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


async def _compacted_turn(
    streaming: bool, history: list[RoomEvent], hook: Any = None
) -> tuple[_Overflowing, LoopRun]:
    provider = _Overflowing(_script(), streaming=streaming, at=_OVERFLOW_AT)
    ch = AIChannel("ai1", provider=provider, tool_handler=_lookup)
    await _turn(ch, "what is the status?")
    ch._before_generation_hook = hook
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
    assert all(results[f"c{n}"].endswith(_LONG) for n in (4, 5, 6))
    assert _no_two_user_messages_in_a_row(replay)
    assert run.calls[-1].name == "read_stored_result" and not run.calls[-1].failed
    assert f"quote 1 {_LONG[:50]}" in str(run.calls[-1].result)
    assert run.text == "done"


async def test_the_history_before_the_input_is_summarized_into_it(streaming: bool) -> None:
    provider, _ = await _compacted_turn(streaming, history=_history(4))

    before, replay = provider.seen[_OVERFLOW_AT - 1], provider.seen[_OVERFLOW_AT]
    input_before = next(m for m in before if _QUESTION in str(m.content))
    first = str(replay[0].content)
    assert first.startswith(SUMMARY_HEADER)
    assert "earlier answer 0" in first
    # The input follows the summary whole, its notes included.
    assert first.endswith(str(input_before.content))
    assert _no_two_user_messages_in_a_row(replay)
    assert "Full output saved as 'evicted_c1'" in _results(replay)["c1"]


async def test_an_input_a_generation_hook_rewrote_is_the_one_kept(streaming: bool) -> None:
    async def hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
        messages = gen_event.ai_context.messages
        if messages and str(messages[-1].content).startswith(_QUESTION):
            rewritten = messages[-1].model_copy(update={"content": f"{messages[-1].content} [ok]"})
            gen_event.ai_context = gen_event.ai_context.model_copy(
                update={"messages": [*messages[:-1], rewritten]}
            )
        return SyncPipelineResult(allowed=True)

    provider, _ = await _compacted_turn(streaming, history=[], hook=hook)

    replay = provider.seen[_OVERFLOW_AT]
    assert str(replay[0].content).startswith(_QUESTION)
    assert str(replay[0].content).endswith("[ok]")
    assert "Full output saved as 'evicted_c1'" in _results(replay)["c1"]


async def test_a_note_a_generation_hook_adds_rides_the_notes_and_survives_compaction(
    streaming: bool,
) -> None:
    """RFC §6.4: the hook's block joins the channel's notes, after them, under
    their one header, and the compaction keeps it whole with the input."""

    async def hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
        messages = gen_event.ai_context.messages
        if messages and str(messages[-1].content).startswith(_QUESTION):
            gen_event.ai_context.messages = add_turn_note(messages, "It is 09:30 in Montreal.")
        return SyncPipelineResult(allowed=True)

    provider, _ = await _compacted_turn(streaming, history=[], hook=hook)

    before, replay = provider.seen[_OVERFLOW_AT - 1], provider.seen[_OVERFLOW_AT]
    assert replay[0] == before[0]  # the input and its notes, kept as the hook left them
    assert "Full output saved as 'evicted_c1'" in _results(replay)["c1"]  # it did compact
    first = str(replay[0].content)
    assert first.startswith(_QUESTION)
    assert first.count(TURN_NOTES_HEADER) == 1
    assert first.index("SHIPPED-7731") < first.index("It is 09:30 in Montreal.")
    assert first.endswith("It is 09:30 in Montreal.")


def _loop_shaped(result: str) -> list[AIMessage]:
    """An input, then three rounds of one call and its result each."""
    messages = [AIMessage(role="user", content="q")]
    for n in range(3):
        call = AIToolCallPart(id=f"c{n}", name="lookup")
        messages.append(AIMessage(role="assistant", content=[call]))
        part = AIToolResultPart(tool_call_id=f"c{n}", name="lookup", result=result)
        messages.append(AIMessage(role="tool", content=[part]))
    return messages


async def test_nothing_left_to_compact_before_the_input_is_refused() -> None:
    ch = AIChannel("ai1", provider=MockAIProvider())
    messages = _loop_shaped("short")
    loop_ctx = ch._get_loop_ctx()
    loop_ctx.turn_input = messages[0]
    token = _current_loop_ctx.set(loop_ctx)
    try:
        with pytest.raises(ProviderError, match="nothing left to compact"):
            await ch._compact_context(AIContext(messages=messages))
    finally:
        _current_loop_ctx.reset(token)


def test_a_skill_body_and_a_page_read_back_stay_whole() -> None:
    body = "RULE " * 1000

    def tool_message(name: str) -> AIMessage:
        part = AIToolResultPart(tool_call_id=f"id_{name}", name=name, result=body)
        return AIMessage(role="tool", content=[part])

    messages = [tool_message(n) for n in ("activate_skill", "read_stored_result", "lookup")]

    kept = with_results_stored(messages, ToolEviction())

    assert kept[0] is messages[0] and kept[1] is messages[1]
    assert "Full output saved as 'evicted_id_lookup'" in _results(kept)["id_lookup"]


def test_a_result_of_parts_stores_its_text_and_keeps_its_image() -> None:
    image = AIImagePart(url="https://example.com/a.png")
    part = AIToolResultPart(
        tool_call_id="c1", name="screenshot", result=[AITextPart(text=_LONG), image]
    )

    [kept] = with_results_stored([AIMessage(role="tool", content=[part])], ToolEviction())

    [stored] = kept.content
    assert isinstance(stored, AIToolResultPart) and isinstance(stored.result, list)
    text, kept_image = stored.result
    assert isinstance(text, AITextPart) and "Full output saved as" in text.text
    assert kept_image == image


def test_a_summary_joins_a_user_message_of_parts_as_its_first_part() -> None:
    image = AIImagePart(url="https://example.com/a.png")
    first = AIMessage(role="user", content=[AITextPart(text="see"), image])

    [joined] = with_leading_text("SUMMARY", [first])

    assert joined.content == [AITextPart(text="SUMMARY"), AITextPart(text="see"), image]
    assert with_leading_text("S", [AIMessage(role="user", content="")])[0].content == "S"


def test_a_summary_never_leaves_a_delimited_block_open() -> None:
    text = "see <tool_result>\n" + "y" * 900 + "\n</tool_result> and <TOOL_RESULT>\nz"
    messages = [AIMessage(role="user", content=text)]

    summary = summary_text(messages) or ""

    assert "tool_result>" not in summary.lower()
    assert summary.count("[tool_result]") == 2
    assert named_blocks("a <worker_output>\nw\n</worker_output> b") == "a [worker_output] b"


class _Summarizing(MemoryProvider):
    """A memory provider that hands back a summary as a user message."""

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        return MemoryResult(messages=[AIMessage(role="user", content="SUMMARY of before")])


async def test_a_memory_summary_joins_the_user_message_after_it(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
    ch = AIChannel("ai1", provider=provider, memory=_Summarizing())

    await _turn(ch, _QUESTION)

    [message] = provider.calls[0].messages
    assert message.role == "user"
    assert str(message.content).startswith("SUMMARY of before\n\n" + _QUESTION)
