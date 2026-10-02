"""What a provider that does not stream hands the tool loop (RFC §6.4, RMK-308).

The default ``generate_structured_stream`` reads what the provider has: a turn
without tools streams through ``generate_stream`` where the provider streams
text, any other turn wraps ``generate()`` with everything it returned kept.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIProvider,
    AIResponse,
    AITool,
    AIToolCall,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    response_stream_events,
)

_TOOL = AITool(name="lookup", description="d", parameters={"type": "object"})


class _Generating(AIProvider):
    """A provider with ``generate()`` alone, or a text stream beside it."""

    def __init__(self, response: AIResponse, *, streams_text: bool = False) -> None:
        self._response = response
        self._streams_text = streams_text

    @property
    def model_name(self) -> str:
        return "generating"

    @property
    def supports_streaming(self) -> bool:
        return self._streams_text

    async def generate(self, context: AIContext) -> AIResponse:
        return self._response

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for text in ("Hel", "lo."):
            yield text


async def _events(provider: AIProvider, context: AIContext) -> list[object]:
    return [event async for event in provider.generate_structured_stream(context)]


def _context(*, tools: bool) -> AIContext:
    return AIContext(
        messages=[AIMessage(role="user", content="hi")], tools=[_TOOL] if tools else []
    )


async def test_a_wrapped_response_keeps_its_thinking_signature_and_call_metadata() -> None:
    response = AIResponse(
        content="Looking.",
        thinking="I should look it up.",
        thinking_signature="SIG-123",
        finish_reason="tool_calls",
        usage={"input_tokens": 10, "output_tokens": 2},
        metadata={"model": "m"},
        tool_calls=[
            AIToolCall(
                id="c1", name="lookup", arguments={"q": "x"}, metadata={"thought_signature": "TS"}
            )
        ],
    )

    events = await _events(_Generating(response), _context(tools=True))

    assert events == [
        StreamThinkingDelta(thinking="I should look it up.", signature="SIG-123"),
        StreamTextDelta(text="Looking."),
        StreamToolCall(
            id="c1", name="lookup", arguments={"q": "x"}, metadata={"thought_signature": "TS"}
        ),
        StreamDone(
            finish_reason="tool_calls",
            usage={"input_tokens": 10, "output_tokens": 2},
            metadata={"model": "m"},
        ),
    ]


def test_a_signature_without_thinking_still_reaches_the_loop() -> None:
    response = AIResponse(content="Done.", thinking_signature="SIG-only")

    events = list(response_stream_events(response))

    assert events[0] == StreamThinkingDelta(thinking="", signature="SIG-only")


async def test_a_turn_without_tools_streams_the_text_a_provider_streams() -> None:
    provider = _Generating(AIResponse(content="never read"), streams_text=True)

    events = await _events(provider, _context(tools=False))

    assert events == [
        StreamTextDelta(text="Hel"),
        StreamTextDelta(text="lo."),
        StreamDone(finish_reason="stop"),
    ]


async def test_a_turn_with_tools_wraps_generate_even_where_text_streams() -> None:
    provider = _Generating(AIResponse(content="From generate."), streams_text=True)

    events = await _events(provider, _context(tools=True))

    assert events[0] == StreamTextDelta(text="From generate.")
