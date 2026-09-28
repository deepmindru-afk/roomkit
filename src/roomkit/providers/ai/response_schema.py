"""Constrained JSON output: the error, and the checks every provider shares.

Each provider translates :attr:`AIContext.response_schema` into its own request
field. What is common to all of them lives here, so every provider reports it
the same way (RFC §6.7): when a call cannot carry a schema at all, and when an
answer came back without the JSON document it was constrained to.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, Literal

from roomkit.providers.ai.base import (
    AIContext,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.json_schema import schema_mismatch
from roomkit.providers.utils import _aclose_stream

ResponseSchemaFailure = Literal["unsupported", "refusal", "truncated", "invalid_json"]


class ResponseSchemaError(ProviderError):
    """A response schema was set and no answer satisfying it can be returned.

    Never retryable: the same request fails the same way.

    Attributes:
        reason: ``"unsupported"`` when the call cannot carry a schema: the
            provider does not support one, the turn also has tools, or a
            streaming method received it. Raised before any request is sent.
            ``"refusal"`` when the model declined to answer. ``"truncated"``
            when the output cap cut the document. ``"invalid_json"`` when the
            text is not a JSON document satisfying the schema, from a server
            that accepted the constraint and did not apply it.
    """

    def __init__(self, message: str, *, reason: ResponseSchemaFailure, provider: str = "") -> None:
        super().__init__(message, retryable=False, provider=provider)
        self.reason: ResponseSchemaFailure = reason


def schema_for_generate(
    context: AIContext, *, supported: bool, provider: str, with_tools: bool = False
) -> dict[str, Any] | None:
    """The schema a call must send, or ``None`` when there is none.

    Args:
        context: The turn's context.
        supported: The provider's :attr:`~AIProvider.supports_response_schema`.
        provider: The provider's name, carried by the error.
        with_tools: The provider's
            :attr:`~AIProvider.supports_response_schema_with_tools`.

    Raises:
        ResponseSchemaError: ``unsupported``, when the provider cannot honour a
            schema, or cannot honour one in a turn that also carries tools.
    """
    schema = context.response_schema
    if schema is None:
        return None
    if not supported:
        raise ResponseSchemaError(
            f"{provider} does not support a response schema; check "
            "supports_response_schema before setting one",
            reason="unsupported",
            provider=provider,
        )
    if context.tools and not with_tools:
        raise ResponseSchemaError(
            f"{provider} cannot combine a response schema with tools in the same "
            "turn; check supports_response_schema_with_tools before setting both",
            reason="unsupported",
            provider=provider,
        )
    return schema


def check_schema_answer(
    content: str,
    *,
    schema: Mapping[str, Any],
    provider: str,
    refusal: str | None = None,
    truncated: bool = False,
) -> None:
    """Refuse a constrained answer that did not deliver its JSON document.

    The document is checked against the schema itself, not only parsed: a
    server that takes the constraint and ignores it (an OpenAI-compatible
    proxy routing to an upstream without structured output, say) answers
    well-formed JSON of another shape, which must not pass for an answer.

    Args:
        content: The answer's text, reasoning already split out.
        schema: The turn's response schema.
        provider: The provider's name, carried by the error.
        refusal: Why the model declined, when it did: the provider's refusal
            text or its refusal or safety stop reason.
        truncated: Whether the output cap cut the answer.

    Raises:
        ResponseSchemaError: ``refusal``, ``truncated`` or ``invalid_json``.
    """
    if refusal is not None:
        raise ResponseSchemaError(
            f"the model declined to answer: {refusal}", reason="refusal", provider=provider
        )
    if truncated:
        raise ResponseSchemaError(
            "the output cap cut the JSON answer; raise max_tokens",
            reason="truncated",
            provider=provider,
        )
    try:
        document = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ResponseSchemaError(
            f"the answer is not a JSON document ({exc})",
            reason="invalid_json",
            provider=provider,
        ) from exc
    mismatch = schema_mismatch(schema, document)
    if mismatch is not None:
        raise ResponseSchemaError(
            f"the answer does not satisfy the response schema ({mismatch})",
            reason="invalid_json",
            provider=provider,
        )


async def checked_stream(
    events: AsyncIterator[StreamEvent],
    context: AIContext,
    *,
    provider: str,
    refusal: Callable[[StreamDone], str | None],
    truncated: Callable[[StreamDone], bool],
) -> AsyncIterator[StreamEvent]:
    """Pass a provider's stream through, checking a constrained answer before
    its done event (RFC §6.7).

    The text deltas go out as they come: partial JSON, provisional until the
    end. When the stream reaches :class:`StreamDone` without a tool call, the
    whole text is checked like a ``generate()`` answer, and the error replaces
    the done event when it fails. Without a schema, the stream is untouched.

    Args:
        events: The provider's own event stream, closed when this one is.
        context: The turn's context, whose ``response_schema`` is checked.
        provider: The provider's name, carried by the error.
        refusal: Reads, from the done event, why the model declined, if it did.
        truncated: Reads, from the done event, whether the answer was cut.
    """
    schema = context.response_schema
    text: list[str] = []
    tool_called = False
    try:
        async for event in events:
            if schema is not None:
                if isinstance(event, StreamTextDelta):
                    text.append(event.text)
                elif isinstance(event, StreamToolCall):
                    tool_called = True
                elif isinstance(event, StreamDone) and not tool_called:
                    check_schema_answer(
                        "".join(text),
                        schema=schema,
                        provider=provider,
                        refusal=refusal(event),
                        truncated=truncated(event),
                    )
            yield event
    finally:
        await _aclose_stream(events)
