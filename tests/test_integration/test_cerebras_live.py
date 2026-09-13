"""Opt-in Cerebras smoke tests with synthetic prompts and bounded output.

Run with CEREBRAS_API_KEY and ROOMKIT_RUN_CEREBRAS_LIVE=1 in the environment:
    uv run --extra cerebras pytest tests/test_integration/test_cerebras_live.py -q

These tests make paid API requests. No credentials or response text are logged.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from roomkit import AIChannel, CerebrasAIProvider, CerebrasConfig
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.streaming import LoopEndMarker, ToolCallEndMarker
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from tests.conftest import make_event

pytestmark = pytest.mark.skipif(
    os.environ.get("ROOMKIT_RUN_CEREBRAS_LIVE") != "1" or not os.environ.get("CEREBRAS_API_KEY"),
    reason="Set ROOMKIT_RUN_CEREBRAS_LIVE=1 and CEREBRAS_API_KEY to run live tests",
)


@pytest.fixture(params=["gpt-oss-120b", "qwen-3.8-27b"])
async def provider(request: pytest.FixtureRequest) -> AsyncIterator[CerebrasAIProvider]:
    pytest.importorskip("openai")
    instance = CerebrasAIProvider(
        CerebrasConfig(
            api_key=os.environ["CEREBRAS_API_KEY"],
            model=request.param,
            reasoning_effort="low",
            max_tokens=1024,
            timeout=30.0,
        )
    )
    try:
        yield instance
    finally:
        await instance.close()


async def test_generate(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    result = await provider.generate(
        AIContext(messages=[AIMessage(role="user", content="Reply with exactly the word pong.")])
    )
    assert "pong" in result.content.lower()
    assert result.finish_reason == "stop"
    assert result.usage["output_tokens"] > 0
    record_property("usage", result.usage)


async def test_channel_streams_dependent_tool_rounds(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    """A real model must read a tool's new value before calling the next tool."""
    code = secrets.token_hex(2)
    calls: list[str] = []
    reports: list[AIResponseEvent] = []
    verified = False

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        nonlocal verified
        calls.append(name)
        if name == "issue_code":
            return json.dumps({"code": code})
        assert name == "verify_code"
        assert arguments == {"code": code}
        verified = True
        return "VERIFIED"

    async def report(event: AIResponseEvent) -> None:
        reports.append(event)

    channel = AIChannel(
        "live-ai",
        provider=provider,
        system_prompt=(
            "First call issue_code exactly once. Read its result, then call verify_code "
            "with that exact code. After verification, answer with VERIFIED only."
        ),
        tool_handler=handler,
        tools=[
            AITool(
                name="issue_code",
                description="Issue a new code",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            AITool(
                name="verify_code",
                description="Verify the issued code",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            ),
        ],
        max_tool_rounds=4,
        tool_loop_timeout_seconds=30,
    )
    channel._after_response_hook = report
    output = await channel.on_event(
        make_event(room_id="live-room", body="Issue and verify a code now."),
        ChannelBinding(
            channel_id="live-ai",
            room_id="live-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        RoomContext(room=Room(id="live-room")),
    )
    assert output.response_stream is not None
    async with asyncio.timeout(45):
        items = [item async for item in output.response_stream]
    assert calls[0] == "issue_code" and calls[-1] == "verify_code"
    assert verified
    assert "VERIFIED" in "".join(item for item in items if isinstance(item, str))
    endings = [item for item in items if isinstance(item, LoopEndMarker)]
    assert len(endings) == 1 and endings[0].reason == "completed"
    assert 2 <= endings[0].rounds <= 4
    assert len([item for item in items if isinstance(item, ToolCallEndMarker)]) == len(calls)
    assert len(reports) == 1 and reports[0].tool_calls_count == len(calls)
    assert reports[0].usage["output_tokens"] > 0
    assert channel.active_turns == 0
    record_property("tool_calls", len(calls))
    record_property("rounds", endings[0].rounds)
    record_property("usage", reports[0].usage)


async def test_channel_stream_can_close_early_and_reuse_provider(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    channel = AIChannel(
        "live-ai",
        provider=provider,
        tools=[AITool(name="unused", description="Not needed for this request")],
    )
    output = await channel.on_event(
        make_event(room_id="live-room", body="Count from one to one hundred in words."),
        ChannelBinding(
            channel_id="live-ai",
            room_id="live-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        RoomContext(room=Room(id="live-room")),
    )
    stream = output.response_stream
    assert stream is not None
    try:
        async with asyncio.timeout(35):
            await anext(stream)
        assert channel.active_turns == 1
    finally:
        started = time.monotonic()
        async with asyncio.timeout(10):
            await stream.aclose()
    assert channel.active_turns == 0
    result = await provider.generate(
        AIContext(messages=[AIMessage(role="user", content="Reply with exactly pong.")])
    )
    assert "pong" in result.content.lower()
    record_property("close_and_reuse_ms", (time.monotonic() - started) * 1000)


async def test_stream(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    started = time.monotonic()
    first_text_ms: float | None = None
    text = ""
    done: StreamDone | None = None
    context = AIContext(
        messages=[AIMessage(role="user", content="Reply with exactly the word pong.")]
    )
    async for event in provider.generate_structured_stream(context):
        if isinstance(event, StreamTextDelta):
            if first_text_ms is None:
                first_text_ms = (time.monotonic() - started) * 1000
            text += event.text
        elif isinstance(event, StreamDone):
            done = event
    assert "pong" in text.lower()
    assert done is not None and done.finish_reason == "stop"
    assert done.usage["output_tokens"] > 0
    record_property("first_text_ms", first_text_ms)
    record_property("usage", done.usage)


async def test_streamed_tool_round(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    context = AIContext(
        system_prompt=(
            "Use the add tool to answer addition questions. "
            "After its result, answer with the number only."
        ),
        messages=[AIMessage(role="user", content="Use the add tool to add 17 and 25.")],
        tools=[
            AITool(
                name="add",
                description="Add two integers",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            )
        ],
    )
    events = [event async for event in provider.generate_structured_stream(context)]
    calls = [event for event in events if isinstance(event, StreamToolCall)]
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "add"
    assert call.arguments == {"a": 17, "b": 25}
    thinking = "".join(
        event.thinking for event in events if isinstance(event, StreamThinkingDelta)
    )
    context.messages.extend(
        [
            AIMessage(
                role="assistant",
                content=[
                    AIThinkingPart(thinking=thinking),
                    AIToolCallPart(id=call.id, name=call.name, arguments=call.arguments),
                ],
            ),
            AIMessage(
                role="tool",
                content=[
                    AIToolResultPart(tool_call_id=call.id, name=call.name, result="42"),
                ],
            ),
        ]
    )
    context.tools = []
    result = await provider.generate(context)
    assert "42" in result.content
    assert result.finish_reason == "stop"
    record_property("usage", result.usage)
