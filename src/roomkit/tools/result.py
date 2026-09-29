"""What a tool handler's answer reads as for a model (RFC §21.4).

A handler answers with text or with a list of content parts, which it may give
as mappings naming their type; anything else it returns is serialized as JSON,
the same on every channel, and so is a result a SYNC ON_TOOL_CALL hook supplies
in its place. A value outside the contract must neither fail the turn nor reach
the model as Python's printing of it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from roomkit.models.tool_call import ToolCallEvent, ToolCallVerdict, chained_call_event
from roomkit.providers.ai.base import AIImagePart, AITextPart

logger = logging.getLogger("roomkit.tools")

ToolResult = str | list[AITextPart | AIImagePart]

_PARTS: TypeAdapter[list[AITextPart | AIImagePart]] = TypeAdapter(list[AITextPart | AIImagePart])
_PART_TYPES = frozenset({"text", "image"})


def as_tool_result(value: Any) -> ToolResult:
    """*value* as the model reads it: text and content parts as they are,
    anything else as JSON (a mapping, a list of values, a number, ``None``)."""
    if isinstance(value, str):
        return value
    parts = _content_parts(value)
    if parts is not None:
        return parts
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def tool_call_verdict(hook_result: Any, event: ToolCallEvent) -> ToolCallVerdict:
    """ON_TOOL_CALL's SYNC chain on *event*, as the verdict the channel applies.

    A BLOCK is told apart from a rewrite, since a blocked call must not keep
    its structured copy. Otherwise the verdict is the event the chain left
    (``fold_tool_call_rewrite``): its structured copy, and its result as the
    model reads it where a hook replaced it. A replacement counts whatever
    its value (RFC §9.3): a hook that clears the result has the model read
    ``null``, never the original.
    """
    if not hook_result.allowed:
        reason = json.dumps({"error": hook_result.reason or "blocked"})
        return ToolCallVerdict(result=reason, blocked=True)
    final = chained_call_event(hook_result, event)
    copy = final.structured_content
    if copy is not None and not isinstance(copy, Mapping):
        # A MODIFY skips the fold's check; the same rule applies to it.
        logger.warning(
            "ON_TOOL_CALL hook left a structured_content of type %s, not a mapping; "
            "the call's structured copy is dropped",
            type(copy).__name__,
        )
        copy = None
    replaced = final.result is not event.result
    return ToolCallVerdict(
        result=as_tool_result(final.result) if replaced else None,
        replaces_structured=copy is not event.structured_content,
        structured_content=dict(copy) if copy is not None else None,
    )


def is_unknown_tool_answer(result: Any) -> bool:
    """Whether *result* is the answer by which a handler says a tool is not
    its to serve (``{"error": "Unknown tool: ..."}``), the one
    :func:`~roomkit.tools.compose.compose_tool_handlers` passes a call on for.
    """
    if not isinstance(result, str):
        return False  # A multimodal result is always a handled tool
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(parsed, dict):
        error = parsed.get("error", "")
        return isinstance(error, str) and error.lower().startswith("unknown tool")
    return False


def unserved_tool_error(name: str) -> str:
    """The failure a call reports when no handler and no hook served it."""
    return json.dumps({"error": f"No handler for tool {name}"})


def _content_parts(value: Any) -> list[AITextPart | AIImagePart] | None:
    """*value* as content parts, when every item is one or a mapping naming
    its part type; ``None`` otherwise."""
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(item, AITextPart | AIImagePart) for item in value):
        return value
    if not all(
        isinstance(item, AITextPart | AIImagePart)
        or (isinstance(item, dict) and item.get("type") in _PART_TYPES)
        for item in value
    ):
        return None
    try:
        return _PARTS.validate_python(value)
    except ValidationError:
        return None


def _json_default(value: Any) -> Any:
    """JSON for the values ``json`` does not know, never their Python repr."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", errors="replace")
    return str(value)
