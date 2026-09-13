"""A tool loop cannot finish while an abandoned parallel call is still running."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.telemetry import MockTelemetryProvider, SpanKind
from roomkit.tools.external import BeforeToolDecision


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["tool_cancelled", "gate_error"])
async def test_aborted_parallel_round_joins_other_tool_finalizers(
    streaming: bool, failure: str
) -> None:
    started = asyncio.Event()
    finished = asyncio.Event()
    siblings: list[asyncio.Task[Any]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "abort":
            await started.wait()
            raise asyncio.CancelledError
        task = asyncio.current_task()
        assert task is not None
        siblings.append(task)
        started.set()
        try:
            await asyncio.Future()
        finally:
            # Joining is required; merely requesting cancellation is not enough.
            await asyncio.sleep(0)
            finished.set()
        return "unreachable"

    async def gate(event: ToolCallEvent) -> BeforeToolDecision:
        if event.name == "abort":
            await started.wait()
            raise RuntimeError("pre-execution gate failed")
        return BeforeToolDecision(allowed=True)

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id=name, name=name, arguments={}) for name in ("slow", "abort")
                ],
            )
        ],
        streaming=streaming,
    )
    channel = AIChannel("ai", provider=provider, tool_handler=handler)
    telemetry = MockTelemetryProvider()
    channel._telemetry = telemetry
    if failure == "gate_error":
        channel._before_tool_call_hook = gate
    context = AIContext(messages=[AIMessage(role="user", content="go")])

    async def run() -> None:
        if streaming:
            async for _ in channel._run_streaming_tool_loop(context):
                pass
        else:
            await channel._run_tool_loop(context)

    try:
        error = RuntimeError if failure == "gate_error" else asyncio.CancelledError
        with pytest.raises(error):
            await asyncio.wait_for(run(), 2)
        assert started.is_set()
        assert finished.is_set(), "the loop returned before its parallel tool was cleaned up"
        assert all(task.done() for task in siblings)
        assert channel.active_turns == 0
        assert telemetry.get_active_spans() == []
        tool_spans = telemetry.get_spans(SpanKind.LLM_TOOL_CALL)
        assert {span.name for span in tool_spans} == (
            {"tool.slow", "tool.abort"} if failure == "tool_cancelled" else {"tool.slow"}
        )
        assert all(span.status == "cancelled" for span in tool_spans)
    finally:
        for task in siblings:
            if not task.done():
                task.cancel()
        await asyncio.gather(*siblings, return_exceptions=True)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("phase", ["handler", "result_hook"])
async def test_cancelling_tool_turn_closes_span(streaming: bool, phase: str) -> None:
    """Caller cancellation closes the span even while the result hook runs."""
    started = asyncio.Event()
    finalized = asyncio.Event()
    telemetry = MockTelemetryProvider()

    async def pause() -> None:
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            finalized.set()

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        if phase == "handler":
            await pause()
        return "done"

    async def result_hook(event: ToolCallEvent) -> None:
        await pause()

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[AIResponse(content="", tool_calls=[AIToolCall(id="t1", name="slow")])],
    )
    channel = AIChannel("ai", provider=provider, tool_handler=handler)
    channel._telemetry = telemetry
    if phase == "result_hook":
        channel._tool_call_hook = result_hook

    async def run() -> None:
        context = AIContext(messages=[AIMessage(role="user", content="go")])
        if streaming:
            async for _ in channel._run_streaming_tool_loop(context):
                pass
        else:
            await channel._run_tool_loop(context)

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert finalized.is_set()
        assert channel.active_turns == 0
        assert telemetry.get_active_spans() == []
        spans = telemetry.get_spans(SpanKind.LLM_TOOL_CALL)
        assert len(spans) == 1
        assert spans[0].status == "cancelled"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await channel.close()
