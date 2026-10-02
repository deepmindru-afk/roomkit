"""Tests for AIChannel's tool loop, most of them for both kinds of provider.

The tests whose subject does not depend on how the provider streams take the
``streaming`` fixture; the rest (markers, progressive delivery, stream tokens)
use a streaming provider.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import ToolRefusedError
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.streaming import ToolCallEndMarker
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import (
    AIResponse,
    AITool,
    AIToolCall,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import BeforeToolDecision
from tests.conftest import make_event
from tests.tool_loop_modes import LoopCall, respond

_SEARCH_TOOL = AITool(
    name="search",
    description="Search",
    parameters={
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
)


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "tools": [{"name": "search", "description": "Search"}],
        },
    )


def _binding_no_tools() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


class TestStreamingToolLoop:
    """Test the streaming tool loop in AIChannel."""

    async def test_single_tool_round(self, streaming: bool) -> None:
        """Provider returns tool call on round 1, text on round 2."""

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return f"Result for {name}"

        # Round 1: tool call, Round 2: text
        responses = [
            AIResponse(
                content="Let me search.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={"q": "test"}),
                ],
            ),
            AIResponse(
                content="Here are the results.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=streaming)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=tool_handler,
        )

        run = await respond(
            ch,
            make_event(body="search for test", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        # Both rounds' text reaches the reply
        assert run.said == ["Let me search.", "Here are the results."]
        assert run.text == "Here are the results."

        # Provider was called twice (two rounds)
        assert len(provider.calls) == 2

    async def test_progressive_text_delivery(self) -> None:
        """Text from round 1 is yielded before tool execution happens."""
        execution_order: list[str] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            execution_order.append("tool_executed")
            return "done"

        responses = [
            AIResponse(
                content="Thinking...",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={}),
                ],
            ),
            AIResponse(
                content="Final answer.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        output = await ch.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None

        # Consume stream, tracking when text arrives vs tools execute
        chunks: list[str] = []
        async for chunk in output.response_stream:
            if not isinstance(chunk, str):
                continue
            if not execution_order:
                # Tool hasn't been called yet — this text is from round 1
                chunks.append(f"pre_tool:{chunk}")
            else:
                chunks.append(f"post_tool:{chunk}")

        # Round 1 text arrived before tool execution
        pre_tool_text = [c for c in chunks if c.startswith("pre_tool:")]
        assert len(pre_tool_text) > 0
        assert "Thinking..." in "".join(c.split(":", 1)[1] for c in pre_tool_text)

    async def test_no_tools_single_round(self, streaming: bool) -> None:
        """Without tools, stream completes after one generation."""
        provider = MockAIProvider(responses=["Just text."], streaming=streaming)
        ch = AIChannel("ai1", provider=provider)

        run = await respond(
            ch,
            make_event(body="hi", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert run.text == "Just text."
        assert len(provider.calls) == 1

    async def test_no_tools_no_handler(self, streaming: bool) -> None:
        """Tool calls without a handler end the loop after round 1."""
        responses = [
            AIResponse(
                content="I want to call tools but can't.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={}),
                ],
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=streaming)
        # No tool_handler
        ch = AIChannel("ai1", provider=provider)

        run = await respond(
            ch,
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert run.said == ["I want to call tools but can't."]
        # Only one round because no handler
        assert len(provider.calls) == 1

    async def test_max_rounds_honored(self, streaming: bool) -> None:
        """Loop stops at max_tool_rounds even if provider keeps returning tools."""
        tool_executions = 0

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            nonlocal tool_executions
            tool_executions += 1
            return "ok"

        # Every call returns a tool call, each with distinct arguments:
        # identical repeats would trip the anti-loop repeat guard before
        # the round cap under test is reached.
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    tool_calls=[AIToolCall(id=f"tc{i}", name="search", arguments={"q": str(i)})],
                )
                for i in range(10)
            ],
            streaming=streaming,
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=tool_handler,
            max_tool_rounds=3,
        )

        run = await respond(
            ch,
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert run.reason == "max_rounds"
        # max_tool_rounds=3 → 4 generations (0,1,2,3) but only 3 tool executions
        # The last generation sees tool calls but does NOT execute them (no
        # generation would follow to use the results).
        assert len(provider.calls) == 4
        assert tool_executions == 3

    async def test_tool_execution_error_fed_back_to_llm(self, streaming: bool) -> None:
        """Tool errors are fed back as tool results instead of propagating."""

        async def broken_handler(name: str, args: dict[str, Any]) -> str:
            raise RuntimeError("Tool failed!")

        responses = [
            AIResponse(
                content="Calling tool.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={}),
                ],
            ),
            AIResponse(
                content="The tool failed, let me explain.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=broken_handler)

        run = await respond(
            ch,
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        # Both rounds' text: the error went back to the model, not up the stack
        assert run.said == ["Calling tool.", "The tool failed, let me explain."]
        assert run.calls[0].failed

    async def test_context_updated_between_rounds(self, streaming: bool) -> None:
        """Verify tool results are appended to context between rounds."""

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return "42"

        responses = [
            AIResponse(
                content="Checking.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="calculate", arguments={"x": "6*7"}),
                ],
            ),
            AIResponse(
                content="The answer is 42.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        await respond(
            ch,
            make_event(body="what is 6*7?", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        # Second call should have assistant + tool messages appended
        second_ctx = provider.calls[1]
        roles = [m.role for m in second_ctx.messages]
        assert "assistant" in roles
        assert "tool" in roles


class TestDefaultFallback:
    """AIProvider.generate_structured_stream() default wraps generate()."""

    async def test_default_wraps_generate(self) -> None:
        """Provider without override returns events from generate() result."""
        provider = MockAIProvider(responses=["Hello world"])

        from roomkit.providers.ai.base import AIContext, AIMessage

        ctx = AIContext(messages=[AIMessage(role="user", content="hi")])

        events: list[StreamEvent] = []
        async for ev in provider.generate_structured_stream(ctx):
            events.append(ev)

        assert len(events) == 2  # text delta + done
        assert isinstance(events[0], StreamTextDelta)
        assert events[0].text == "Hello world"
        assert isinstance(events[1], StreamDone)
        assert events[1].finish_reason == "stop"

    async def test_default_wraps_generate_with_tool_calls(self) -> None:
        """Default fallback includes tool calls from generate()."""
        ai_resp = AIResponse(
            content="Let me check.",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[
                AIToolCall(id="tc1", name="search", arguments={"q": "test"}),
            ],
        )
        provider = MockAIProvider(ai_responses=[ai_resp])

        from roomkit.providers.ai.base import AIContext, AIMessage

        ctx = AIContext(messages=[AIMessage(role="user", content="search")])

        events: list[StreamEvent] = []
        async for ev in provider.generate_structured_stream(ctx):
            events.append(ev)

        assert len(events) == 3  # text delta + tool call + done
        assert isinstance(events[0], StreamTextDelta)
        assert events[0].text == "Let me check."
        assert isinstance(events[1], StreamToolCall)
        assert events[1].name == "search"
        assert isinstance(events[2], StreamDone)


class TestStreamingWithNoToolsBinding:
    """Streaming provider without tools still uses simple streaming path."""

    async def test_no_tools_uses_streaming_response(self) -> None:
        provider = MockAIProvider(responses=["streamed"], streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(
            make_event(body="hi", channel_id="sms1"),
            _binding_no_tools(),
            _ctx(),
        )

        assert output.response_stream is not None
        chunks = [chunk async for chunk in output.response_stream]
        assert "".join(c for c in chunks if isinstance(c, str)) == "streamed"

    async def test_tools_uses_streaming_tool_loop(self) -> None:
        provider = MockAIProvider(responses=["with tools"], streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(
            make_event(body="hi", channel_id="sms1"),
            _binding(),  # has tools
            _ctx(),
        )

        assert output.response_stream is not None
        chunks = [chunk async for chunk in output.response_stream]
        assert "".join(c for c in chunks if isinstance(c, str)) == "with tools"


class TestToolCallEphemeralEvents:
    """Tool calls emit ephemeral events instead of inline XML."""

    async def test_tool_calls_yield_stream_markers(self) -> None:
        """Streaming tool loop yields structured markers instead of ephemeral events."""
        from roomkit.models.streaming import ToolCallStartMarker

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return f"Result for {name}"

        responses = [
            AIResponse(
                content="Let me search.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={"q": "test"}),
                ],
            ),
            AIResponse(
                content="Here are the results.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        output = await ch.on_event(
            make_event(body="search for test", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None
        all_items = [item async for item in output.response_stream]

        # Text deltas are strings
        text = "".join(c for c in all_items if isinstance(c, str))
        assert "Let me search." in text
        assert "Here are the results." in text

        # No XML fragments in the text
        assert "<invoke" not in text
        assert "<result>" not in text

        # Exactly one START and one END marker
        starts = [m for m in all_items if isinstance(m, ToolCallStartMarker)]
        ends = [m for m in all_items if isinstance(m, ToolCallEndMarker)]
        assert len(starts) == 1
        assert len(ends) == 1

        # Validate START marker
        assert starts[0].tool_name == "search"
        assert starts[0].tool_id == "tc1"
        assert starts[0].arguments == {"q": "test"}

        # Validate END marker
        assert ends[0].tool_name == "search"
        assert ends[0].tool_id == "tc1"
        assert ends[0].status == "completed"
        assert ends[0].duration_ms >= 0

    async def test_the_call_reports_its_request_and_what_executed(self, streaming: bool) -> None:
        """Persistence keeps the request and the execution distinct, in both
        loops: the start carries what the model asked for, the end what the
        handler ran with after a BEFORE_TOOL_USE rewrite."""
        seen: list[dict[str, Any]] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(args)
            return "ok"

        async def rewrite(_event: Any) -> BeforeToolDecision:
            return BeforeToolDecision(allowed=True, arguments={"q": "redacted-value"})

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "token"})],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ],
            streaming=streaming,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler, tools=[_SEARCH_TOOL])
        ch._before_tool_call_hook = rewrite

        run = await respond(ch, make_event(body="search", channel_id="sms1"), _binding(), _ctx())

        assert seen == [{"q": "redacted-value"}]
        assert run.calls[0].requested == {"q": "token"}
        assert run.calls[0].arguments == {"q": "redacted-value"}

    async def test_a_provider_that_does_not_stream_yields_the_same_rows(self) -> None:
        """A provider read through generate() yields each round's text and its
        calls' start and end, in order, like any other."""

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return f"Result for {name}"

        responses = [
            AIResponse(
                content="Checking.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="calculate", arguments={"x": "6*7"}),
                ],
            ),
            AIResponse(
                content="The answer is 42.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        # Non-streaming provider. The tool has to be declared: an undeclared
        # name is refused before the handler, and the call this test means to
        # observe would be a refusal — which is what it used to assert as
        # "completed", back when the status was guessed from the result body.
        provider = MockAIProvider(ai_responses=responses, streaming=False)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=tool_handler,
            tools=[
                AITool(
                    name="calculate",
                    description="Calculate",
                    parameters={"type": "object", "properties": {"x": {"type": "string"}}},
                )
            ],
        )

        run = await respond(
            ch, make_event(body="what is 6*7?", channel_id="sms1"), _binding(), _ctx()
        )

        assert run.said == ["Checking.", "The answer is 42."]
        [call] = run.calls
        assert call.name == "calculate"
        assert call.requested == {"x": "6*7"}
        assert not call.failed


class TestStreamingTokenAccumulation:
    """Span attributes should sum tokens across all streaming rounds."""

    async def test_span_accumulates_tokens_across_rounds(self) -> None:
        """Two rounds should have summed input/output tokens on the span."""
        from roomkit.telemetry.base import Attr
        from roomkit.telemetry.mock import MockTelemetryProvider

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return "result"

        responses = [
            AIResponse(
                content="Round 1 text.",
                finish_reason="tool_calls",
                usage={"input_tokens": 100, "output_tokens": 50},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={}),
                ],
            ),
            AIResponse(
                content="Round 2 text.",
                finish_reason="stop",
                usage={"input_tokens": 200, "output_tokens": 75},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        telemetry = MockTelemetryProvider()
        ch._telemetry = telemetry

        output = await ch.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )
        assert output.response_stream is not None
        async for _ in output.response_stream:
            pass

        # Find the LLM_GENERATE span
        llm_spans = [s for s in telemetry.spans if s.name == "llm.generate"]
        assert len(llm_spans) == 1
        span = llm_spans[0]

        # Tokens should be summed: 100+200=300 input, 50+75=125 output
        assert span.attributes.get(Attr.LLM_INPUT_TOKENS) == 300
        assert span.attributes.get(Attr.LLM_OUTPUT_TOKENS) == 125

    async def test_cache_counters_reach_the_after_response_hook(self) -> None:
        """Every counter a round reports is summed, not just input/output.

        The hook payload is where a consumer prices a turn. Forwarding only
        input/output made a cached prefix indistinguishable from fresh input,
        and the two are billed an order of magnitude apart.
        """

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return "result"

        responses = [
            AIResponse(
                content="Round 1.",
                finish_reason="tool_calls",
                usage={
                    "input_tokens": 200,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 1_000,
                },
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={})],
            ),
            AIResponse(
                content="Round 2.",
                finish_reason="stop",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 1_000,
                },
            ),
        ]

        captured: list[AIResponseEvent] = []

        async def after_response(event: AIResponseEvent) -> None:
            captured.append(event)

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
        ch._after_response_hook = after_response

        output = await ch.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )
        assert output.response_stream is not None
        async for _ in output.response_stream:
            pass

        assert captured
        assert captured[0].usage == {
            "input_tokens": 300,
            "output_tokens": 15,
            "cache_read_input_tokens": 1_800,
            "cache_creation_input_tokens": 1_000,
        }


class TestToolOutcome:
    """Both loops state a call's outcome the same way.

    The end's status and error (``ToolCallEndMarker`` on the stream, the
    TOOL_CALL_END event otherwise) are read off ``is_error``, not matched
    against the result text.
    """

    @staticmethod
    async def _outcome(handler: Any, *, streaming: bool) -> LoopCall:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ],
            streaming=streaming,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=handler)
        run = await respond(ch, make_event(body="search", channel_id="sms1"), _binding(), _ctx())
        assert len(run.calls) == 1
        return run.calls[0]

    async def test_a_refusal_is_a_failed_call_in_the_handlers_words(self, streaming: bool) -> None:
        async def declines(name: str, args: dict[str, Any]) -> str:
            raise ToolRefusedError(f"Error: Tool '{name}' is temporarily unavailable.")

        end = await self._outcome(declines, streaming=streaming)

        assert end.failed
        assert end.error == "Error: Tool 'search' is temporarily unavailable."

    async def test_a_handler_that_raised_is_a_failed_call(self, streaming: bool) -> None:
        async def boom(name: str, args: dict[str, Any]) -> str:
            raise RuntimeError("upstream down")

        end = await self._outcome(boom, streaming=streaming)

        assert end.failed
        assert end.error is not None
        assert end.error == '{"error": "Tool \'search\' failed (RuntimeError)"}'

    async def test_a_served_call_is_a_completed_call(self, streaming: bool) -> None:
        async def ok(name: str, args: dict[str, Any]) -> str:
            return "ok"

        end = await self._outcome(ok, streaming=streaming)

        assert not end.failed
        assert end.error is None
