"""Bounded retry when a generation round ends empty *after* a tool call.

Small models sometimes run a tool, get the result, then return no text instead
of a final answer. The tool loop re-prompts once (bounded by ``max_empty_retries``)
for the final answer rather than ending empty. Covers both the non-streaming
(``_run_tool_loop``) and streaming (``_run_streaming_tool_loop``) paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels._ai_loop_rules import _MALFORMED_CALL_NUDGE
from roomkit.channels.ai import _EMPTY_RETRY_NUDGE, AIChannel
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.tool_loop_modes import run_tool_loop


def _tool(content: str = "") -> AIResponse:
    return AIResponse(
        content=content,
        tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
    )


def _final(content: str = "") -> AIResponse:
    return AIResponse(content=content, tool_calls=[])


def _ctx() -> AIContext:
    return AIContext(messages=[AIMessage(role="user", content="go")])


def _nudged(context: AIContext) -> int:
    return sum(1 for m in context.messages if m.content == _EMPTY_RETRY_NUDGE)


async def test_retries_empty_after_tool_and_recovers(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_tool(), _final(""), _final("Recovered")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    run = await run_tool_loop(ch, context, streaming=streaming)
    assert run.text == "Recovered"
    assert _nudged(context) == 1  # one corrective re-prompt injected


async def test_no_retry_when_budget_zero(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_tool(), _final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=0,
    )
    context = _ctx()
    run = await run_tool_loop(ch, context, streaming=streaming)
    assert run.text == ""
    assert _nudged(context) == 0


async def test_no_retry_without_prior_tool(streaming: bool) -> None:
    # A legitimately empty turn with no tools must NOT trigger a retry.
    provider = MockAIProvider(ai_responses=[_final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=2,
    )
    context = _ctx()
    await run_tool_loop(ch, context, streaming=streaming)
    assert _nudged(context) == 0
    assert len(provider.calls) == 1  # no extra generation


async def test_bounded_gives_up_when_still_empty(streaming: bool) -> None:
    # Model stays empty: retry once (budget 1) then give up with the empty answer.
    provider = MockAIProvider(ai_responses=[_tool(), _final(""), _final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    run = await run_tool_loop(ch, context, streaming=streaming)
    assert run.text == ""
    assert _nudged(context) == 1  # exactly one retry, then give up


def _malformed() -> AIResponse:
    """A round that ended on a tool call its provider could not parse."""
    return AIResponse(content="", tool_calls=[], finish_reason="MALFORMED_FUNCTION_CALL")


def _told_malformed(context: AIContext) -> int:
    return sum(1 for m in context.messages if m.content == _MALFORMED_CALL_NUDGE)


async def test_a_malformed_call_is_told_and_retried_on_the_first_round(streaming: bool) -> None:
    """The model learns its call did not run and issues it again (RMK-314)."""
    handler = AsyncMock(return_value="ok")
    provider = MockAIProvider(ai_responses=[_malformed(), _tool(), _final("Done")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=handler,
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    run = await run_tool_loop(ch, context, streaming=streaming)
    assert run.text == "Done"
    assert _told_malformed(context) == 1
    assert handler.await_count == 1


async def test_a_malformed_call_past_the_budget_ends_the_turn_empty(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_malformed()])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=0,
    )
    run = await run_tool_loop(ch, _ctx(), streaming=streaming)
    assert run.text == ""
    assert run.reason == "empty_response"
