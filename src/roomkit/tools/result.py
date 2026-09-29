"""What a tool handler's answer reads as for a model (RFC §21.4).

A handler answers with text or with a list of content parts; anything else it
returns is serialized as JSON, the same on every channel, and so is a result a
SYNC ON_TOOL_CALL hook supplies in its place. A value outside the contract must
neither fail the turn nor reach the model as Python's printing of it.
"""

from __future__ import annotations

import json
from typing import Any

from roomkit.providers.ai.base import AIImagePart, AITextPart

ToolResult = str | list[AITextPart | AIImagePart]


def as_tool_result(value: Any) -> ToolResult:
    """*value* as the model reads it: text and content parts as they are,
    anything else as JSON (a mapping, a list of values, a number, ``None``)."""
    if isinstance(value, str):
        return value
    if (
        isinstance(value, list)
        and value
        and all(isinstance(part, AITextPart | AIImagePart) for part in value)
    ):
        return value
    return json.dumps(value, default=str)


class UnservedToolCallError(Exception):
    """Raised by a channel's dispatcher when nothing serves a call.

    Not a refusal: ON_TOOL_CALL's hooks may still serve the call (RFC §9.3).
    Raised rather than returned, so a handler keeps its contract (it answers
    with a result) and the wrappers orchestration puts around the dispatcher
    carry it through unchanged.
    """


def unserved_tool_error(name: str) -> str:
    """The failure a call reports when no handler and no hook served it."""
    return json.dumps({"error": f"No handler for tool {name}"})
