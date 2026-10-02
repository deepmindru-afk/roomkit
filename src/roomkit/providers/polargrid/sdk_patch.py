"""A patch of polargrid-sdk: what its chat requests send, and what a streamed
chat reports.

Two defects, both measured on polargrid-sdk 0.10.0 against edge yul-01 on
2026-10-02, and PolarGrid does not plan to change its SDK:

- Its body builder replaces a falsy value with its own default
  (``request.temperature or 0.7``, ``top_p or 0.9``, ``max_tokens or 150``):
  a ``temperature`` of 0 goes out as 0.7.
- It can neither ask for a streamed chat's usage nor read it: its
  ``ChatCompletionRequest`` has no ``stream_options`` and its
  ``ChatCompletionChunk`` no ``usage``, so a turn streamed through the SDK
  reports no tokens, while the server sends the usage last when asked.

:func:`chat_completion` and :func:`chat_completion_stream` are the client's
own methods with those values put back and the usage asked for and read: the
SDK still checks the request, builds its body, authenticates and parses every
answer, through its private ``_build_chat_completion_body``, ``_make_request``,
``_convert_chat_completion_response`` and ``_stream_post``. Remove this module
and call the client's methods again when ``test_the_sdk_still_*`` in
``tests/test_providers/test_polargrid_sdk_patch.py`` fail: the SDK then sends
what it is given and hands the stream's usage over itself. ``pyproject.toml``
caps polargrid-sdk below its next minor version, since the patch calls the
SDK's private methods.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger("roomkit.providers.polargrid")


@dataclass(frozen=True)
class UsageChunk:
    """The stream's usage, handed over as a chunk with no choices."""

    usage: Any
    choices: tuple[()] = ()


def _chunk(sdk: ModuleType, raw: dict[str, Any]) -> Any | None:
    """*raw* as the SDK's ``ChatCompletionChunk``, or ``None`` for a line its
    type does not hold, which the SDK skips (the closing ``pg_metadata`` line
    among them)."""
    try:
        return sdk.ChatCompletionChunk(**raw)
    except PydanticValidationError:
        logger.debug("PolarGrid stream line skipped: %s", raw)
        return None


def _usage(sdk: ModuleType, raw: dict[str, Any]) -> UsageChunk | None:
    """The usage *raw* carries, or ``None``: usage is a report, and one the
    SDK's ``TokenUsage`` cannot read must not fail a turn already streamed."""
    try:
        return UsageChunk(usage=sdk.TokenUsage(**raw["usage"]))
    except (PydanticValidationError, TypeError):
        logger.warning("PolarGrid usage line unreadable, usage dropped: %s", raw["usage"])
        return None


def _request(sdk: ModuleType, request: dict[str, Any]) -> Any:
    """*request* as the SDK's ``ChatCompletionRequest``, refused as it refuses
    one: its messages too are read inside the model."""
    try:
        return sdk.ChatCompletionRequest(**request)
    except PydanticValidationError as exc:
        raise sdk.ValidationError(str(exc)) from exc


# What the SDK's body builder replaces when it reads the value as unset.
_GIVEN_AS_IS = ("max_tokens", "temperature", "top_p")


def _body(
    sdk: ModuleType, client: Any, request: dict[str, Any], *, stream: bool
) -> dict[str, Any]:
    """The body the SDK builds for *request*, checked as it checks one, with
    the values its builder would replace sent as given."""
    parsed = _request(sdk, request)
    client._validate_chat_completion_request(parsed)
    body = client._build_chat_completion_body(parsed, stream_override=stream)
    for key in _GIVEN_AS_IS:
        if request.get(key) is not None:
            body[key] = request[key]
    return body


async def chat_completion(sdk: ModuleType, client: Any, request: dict[str, Any]) -> Any:
    """``client.chat_completion(request)``, sending what it was given."""
    body = _body(sdk, client, request, stream=False)
    response = await client._make_request("/v1/chat/completions", method="POST", body=body)
    return client._convert_chat_completion_response(response)


async def chat_completion_stream(
    sdk: ModuleType, client: Any, request: dict[str, Any]
) -> AsyncIterator[Any]:
    """``client.chat_completion_stream(request)``, sending what it was given,
    then the usage the server sends last, as a :class:`UsageChunk`."""
    body = _body(sdk, client, request, stream=True)
    body["stream_options"] = {"include_usage": True}
    async for raw in client._stream_post("/v1/chat/completions", body):
        if "error" in raw:
            raise sdk.NetworkError(raw["error"].get("message", "stream error"), None, None)
        chunk = _chunk(sdk, raw)
        if chunk is not None:
            yield chunk
        usage = _usage(sdk, raw) if raw.get("usage") else None
        if usage is not None:
            yield usage
