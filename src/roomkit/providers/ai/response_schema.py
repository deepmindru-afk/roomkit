"""Constrained JSON output: the error, and the checks every provider shares.

Each provider translates :attr:`AIContext.response_schema` into its own request
field. What is common to all of them lives here, so every provider reports it
the same way (RFC §6.7): when a call cannot carry a schema at all, and when an
answer came back without the JSON document it was constrained to.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from roomkit.providers.ai.base import AIContext, ProviderError
from roomkit.providers.ai.json_schema import schema_mismatch

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
    context: AIContext, *, supported: bool, provider: str
) -> dict[str, Any] | None:
    """The schema a ``generate()`` call must send, or ``None`` when there is none.

    Args:
        context: The turn's context.
        supported: The provider's :attr:`~AIProvider.supports_response_schema`.
        provider: The provider's name, carried by the error.

    Raises:
        ResponseSchemaError: ``unsupported``, when the provider cannot honour a
            schema, or when the turn also carries tools.
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
    if context.tools:
        raise ResponseSchemaError(
            "a response schema cannot be combined with tools in the same turn",
            reason="unsupported",
            provider=provider,
        )
    return schema


def refuse_streamed_schema(context: AIContext, *, provider: str) -> None:
    """Refuse a schema on a streaming call, which this contract does not cover yet.

    Raises:
        ResponseSchemaError: ``unsupported``, when the context carries a schema.
    """
    if context.response_schema is not None:
        raise ResponseSchemaError(
            "a response schema is honoured by generate() only, not by a streaming call",
            reason="unsupported",
            provider=provider,
        )


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
