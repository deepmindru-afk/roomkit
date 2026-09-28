"""Every outcome of a tool call is bounded before the model reads it.

A refusal message, an exception's text and an ON_TOOL_CALL override are
evicted like a handler's result when oversized (RMK-259), and the hooks see
the text they are documented to see: the refusal observer the raw message,
ON_TOOL_CALL the bounded result.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels._tool_eviction import is_eviction_placeholder
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import ToolRefusedError
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIResponse,
    AITextPart,
    AIToolCall,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider

_HUGE = "<html>" + "x" * 500_000 + "</html>"


def _channel(handler: AsyncMock, *, vision: bool = False) -> tuple[AIChannel, MockAIProvider]:
    provider = MockAIProvider(
        vision=vision,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="t1", name="fetch", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ],
    )
    return AIChannel("ai1", provider=provider, tool_handler=handler), provider


async def _model_copy(ch: AIChannel, provider: MockAIProvider) -> AIToolResultPart:
    await ch._run_tool_loop(AIContext(messages=[AIMessage(role="user", content="go")]))
    tool_message = next(m for m in provider.calls[-1].messages if m.role == "tool")
    part = tool_message.content[0]
    assert isinstance(part, AIToolResultPart)
    return part


async def test_an_oversized_refusal_is_evicted_and_the_hook_sees_it_whole() -> None:
    ch, provider = _channel(AsyncMock(side_effect=ToolRefusedError(_HUGE)))
    observed: list[ToolCallEvent] = []
    ch._tool_observer_hook = AsyncMock(side_effect=observed.append)

    part = await _model_copy(ch, provider)

    assert part.is_error
    assert isinstance(part.result, str) and is_eviction_placeholder(part.result)
    assert len(part.result) < 9_000
    assert observed[0].result == _HUGE


async def test_an_oversized_exception_is_evicted() -> None:
    ch, provider = _channel(AsyncMock(side_effect=RuntimeError(_HUGE)))

    part = await _model_copy(ch, provider)

    assert part.is_error
    assert isinstance(part.result, str) and is_eviction_placeholder(part.result)
    assert _HUGE in ch._eviction._store[("", "evicted_t1")]


async def test_an_oversized_override_is_evicted_and_the_hook_input_is_unchanged() -> None:
    ch, provider = _channel(AsyncMock(return_value="small result"))
    seen: list[ToolCallEvent] = []

    async def rewrite(event: ToolCallEvent) -> str:
        seen.append(event)
        return _HUGE

    ch._tool_call_hook = rewrite

    part = await _model_copy(ch, provider)

    assert seen[0].result == "small result"
    assert isinstance(part.result, str) and is_eviction_placeholder(part.result)
    assert ch._eviction._store[("", "evicted_t1")] == _HUGE


async def test_a_small_error_reaches_the_model_unchanged() -> None:
    ch, provider = _channel(AsyncMock(side_effect=ToolRefusedError("Missing tenant header")))

    part = await _model_copy(ch, provider)

    assert part.result == "Missing tenant header"


async def test_an_oversized_part_list_override_keeps_its_images() -> None:
    ch, provider = _channel(AsyncMock(return_value="small result"), vision=True)
    image = AIImagePart(url="data:image/png;base64,AAAA", mime_type="image/png")

    async def rewrite(event: ToolCallEvent) -> list[AITextPart | AIImagePart]:
        return [AITextPart(text=_HUGE), image]

    ch._tool_call_hook = rewrite

    part = await _model_copy(ch, provider)

    assert isinstance(part.result, list)
    text, kept = part.result
    assert isinstance(text, AITextPart) and is_eviction_placeholder(text.text)
    assert kept == image
