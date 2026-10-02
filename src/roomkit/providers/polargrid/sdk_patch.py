"""A patch of polargrid-sdk, so that a streamed chat reports its usage.

polargrid-sdk can neither ask for a streamed chat's usage nor read it: its
``ChatCompletionRequest`` has no ``stream_options`` and its
``ChatCompletionChunk`` no ``usage`` (measured on 0.10.0), so every streamed
turn reported no tokens. PolarGrid's server sends the usage on the stream's
last chunk when ``stream_options.include_usage`` is set (measured on
2026-10-02, edge yul-01, qwen-3.8-27b), and PolarGrid does not plan to change
its SDK (2026-10-02).

:func:`chat_completion_stream` is the client's own ``chat_completion_stream``
with the usage asked for and read: the SDK still checks the request, builds
its body, authenticates and parses every chunk, through its private
``_build_chat_completion_body`` and ``_stream_post``. Remove this module and
call ``client.chat_completion_stream`` again if a release can ask for the
usage and carries it on a chunk: ``test_the_sdk_still_has_no_streamed_usage``
in ``tests/test_providers/test_polargrid_sdk_patch.py`` fails on that release.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from pydantic import ValidationError as PydanticValidationError


@dataclass(frozen=True)
class UsageChunk:
    """The stream's usage, handed over as a chunk with no choices."""

    usage: Any
    choices: tuple[()] = ()


def _chunk(sdk: ModuleType, raw: dict[str, Any]) -> Any | None:
    """*raw* as the SDK's ``ChatCompletionChunk``, or ``None`` for a line its
    type does not hold, which the SDK skips."""
    try:
        return sdk.ChatCompletionChunk(**raw)
    except PydanticValidationError:
        return None


def _request(sdk: ModuleType, request: dict[str, Any]) -> Any:
    """*request* as the SDK's ``ChatCompletionRequest``, refused as it refuses one."""
    fields = dict(request)
    fields["messages"] = [
        sdk.Message(**message) if isinstance(message, dict) else message
        for message in fields.get("messages") or []
    ]
    try:
        return sdk.ChatCompletionRequest(**fields)
    except PydanticValidationError as exc:
        raise sdk.ValidationError(str(exc)) from exc


async def chat_completion_stream(
    sdk: ModuleType, client: Any, request: dict[str, Any]
) -> AsyncIterator[Any]:
    """``client.chat_completion_stream(request)``, then the usage the server
    sends last, as a :class:`UsageChunk`."""
    parsed = _request(sdk, request)
    client._validate_chat_completion_request(parsed)
    body = client._build_chat_completion_body(parsed, stream_override=True)
    body["stream_options"] = {"include_usage": True}
    async for raw in client._stream_post("/v1/chat/completions", body):
        if "error" in raw:
            raise sdk.NetworkError(raw["error"].get("message", "stream error"), None, None)
        chunk = _chunk(sdk, raw)
        if chunk is not None:
            yield chunk
        if raw.get("usage"):
            yield UsageChunk(usage=sdk.TokenUsage(**raw["usage"]))
