"""Mock AI provider for testing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AIToolCall,
    ModelInfo,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.ai.response_schema import (
    check_schema_answer,
    refuse_streamed_schema,
    schema_for_generate,
)

_MOCK_MODELS = [
    ModelInfo(id="mock", display_name="Mock", context_window=8192, supports_vision=False),
    ModelInfo(
        id="mock-vision", display_name="Mock Vision", context_window=8192, supports_vision=True
    ),
]


class MockAIProvider(AIProvider):
    """Round-robin response provider for tests.

    With ``response_schema=True`` it honours :attr:`AIContext.response_schema`
    the way a real provider must (RFC §6.7): the scripted answer has to be a
    JSON document, and an :class:`AIResponse` scripted with
    ``finish_reason="refusal"`` or ``"length"`` raises the matching
    :class:`~roomkit.providers.ai.response_schema.ResponseSchemaError`.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        vision: bool = False,
        ai_responses: list[AIResponse] | None = None,
        streaming: bool = False,
        tool_call_delta_chunks: int = 0,
        response_schema: bool = False,
    ) -> None:
        self.responses = responses or ["Hello from AI"]
        self._ai_responses = ai_responses
        self.calls: list[AIContext] = []
        self._index = 0
        self._vision = vision
        self._streaming = streaming
        self._tool_call_delta_chunks = tool_call_delta_chunks
        self._response_schema = response_schema

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def supports_vision(self) -> bool:
        return self._vision

    @property
    def supports_streaming(self) -> bool:
        return self._streaming

    @property
    def supports_structured_streaming(self) -> bool:
        return self._streaming

    @property
    def supports_response_schema(self) -> bool:
        return self._response_schema

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Fixed two-entry catalog for exercising model-discovery code."""
        return list(_MOCK_MODELS)

    async def generate(self, context: AIContext) -> AIResponse:
        # Refused before the call is recorded: a real provider sends nothing.
        schema_for_generate(context, supported=self._response_schema, provider="mock")
        self.calls.append(context)
        response = self._next_response()
        if context.response_schema is not None:
            check_schema_answer(
                response.content,
                schema=context.response_schema,
                provider="mock",
                refusal="refusal" if response.finish_reason == "refusal" else None,
                truncated=response.finish_reason == "length",
            )
        return response

    def _next_response(self) -> AIResponse:
        """The next scripted response, round-robin."""
        if self._ai_responses:
            resp = self._ai_responses[self._index % len(self._ai_responses)]
            self._index += 1
            return resp
        content = self.responses[self._index % len(self.responses)]
        self._index += 1
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text from generate() as a single delta."""
        refuse_streamed_schema(context, provider="mock")
        response = await self.generate(context)
        if response.content:
            yield response.content

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from generate() result."""
        refuse_streamed_schema(context, provider="mock")
        response = await self.generate(context)
        if response.thinking:
            yield StreamThinkingDelta(thinking=response.thinking)
        if response.content:
            yield StreamTextDelta(text=response.content)
        for tc in response.tool_calls:
            for delta in self._tool_call_deltas(tc):
                yield delta
            yield StreamToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
        yield StreamDone(
            finish_reason=response.finish_reason,
            usage=response.usage,
            metadata=response.metadata,
        )

    def _tool_call_deltas(self, tool_call: AIToolCall) -> list[StreamToolCallDelta]:
        """Split a call's arguments into ``tool_call_delta_chunks`` fragments.

        Real providers deliver a tool call's arguments fragment by fragment;
        this reproduces that for tests of the composition events. Zero chunks
        (the default) emits none, so a test that does not ask for them sees a
        plain complete call.
        """
        chunks = self._tool_call_delta_chunks
        if chunks <= 0:
            return []
        payload = json.dumps(tool_call.arguments)
        size = max(1, -(-len(payload) // chunks))
        fragments = [payload[i : i + size] for i in range(0, len(payload), size)] or [""]
        return [
            StreamToolCallDelta(id=tool_call.id, name=tool_call.name, arguments_delta=fragment)
            for fragment in fragments
        ]
