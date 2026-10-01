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
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from roomkit.models.tool_call import (
    ToolCallEvent,
    ToolCallVerdict,
    chained_call_event,
    renderable_copy,
)
from roomkit.providers.ai.base import AIImagePart, AITextPart

ToolResult = str | list[AITextPart | AIImagePart]

_PARTS: TypeAdapter[list[AITextPart | AIImagePart]] = TypeAdapter(list[AITextPart | AIImagePart])
# The fields each part type may carry: a mapping with any other key is data
# that happens to name a type, never a part (RFC §21.4).
_PART_FIELDS = {
    "text": frozenset(AITextPart.model_fields),
    "image": frozenset(AIImagePart.model_fields),
}


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
    # A MODIFY skips the fold's check; the same rule applies to it.
    copy = renderable_copy(final.structured_content)
    replaced = final.result is not event.result
    return ToolCallVerdict(
        result=as_tool_result(final.result) if replaced else None,
        replaces_structured=final.structured_content is not event.structured_content,
        structured_content=copy,
    )


@dataclasses.dataclass(frozen=True)
class VerdictReading:
    """A call's outcome once ON_TOOL_CALL's verdict applies to what served it."""

    result: ToolResult
    """What the model reads."""
    failed: bool
    """A hook blocked the call, or nothing served it."""
    served: bool
    """A handler or a hook served the call."""
    blocked: bool
    """A hook withheld the result; the model reads the block's reason."""
    replaced: bool
    """A hook's result stands in place of the handler's."""


def read_tool_call_verdict(
    name: str, verdict: ToolCallVerdict | ToolResult | None, served: ToolResult | None
) -> VerdictReading:
    """The one reading of an ON_TOOL_CALL verdict, on every channel (RFC §9.3).

    *served* is what the handler answered, ``None`` when nothing served the
    call. A block withholds the result and fails the call; a hook's result,
    given as a verdict or bare, replaces the handler's, or serves a call
    nothing served; a call that neither a handler nor a hook served failed.
    """
    if verdict is not None and not isinstance(verdict, ToolCallVerdict):
        verdict = ToolCallVerdict(result=verdict)  # a bare override
    if verdict is not None and verdict.blocked:
        reason = verdict.result or json.dumps({"error": "blocked"})
        return VerdictReading(reason, failed=True, served=False, blocked=True, replaced=True)
    if verdict is not None and verdict.result is not None:
        override = as_tool_result(verdict.result)
        return VerdictReading(override, failed=False, served=True, blocked=False, replaced=True)
    if served is None:
        unserved = unserved_tool_error(name)
        return VerdictReading(unserved, failed=True, served=False, blocked=False, replaced=False)
    return VerdictReading(served, failed=False, served=True, blocked=False, replaced=False)


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


def tool_failure(name: str, exc: BaseException) -> str:
    """What the model reads of a call that raised: the tool's failure and the
    exception's class, never its message (RFC §9.3).

    The message can hold anything the failing code held (a connection string
    with its password, a path, a record); it goes to the log and to the
    observers (:func:`failure_detail`). A handler that wants the model to read
    its words raises :class:`~roomkit.core.exceptions.ToolRefusedError`.
    """
    return json.dumps({"error": f"Tool '{name}' failed ({type(exc).__name__})"})


def failure_detail(exc: BaseException) -> str:
    """A failure as logs and observers read it, never the model: class and message."""
    return f"{type(exc).__name__}: {exc}"


def hook_errors_detail(hook_result: Any) -> str | None:
    """What the hooks that failed said, for logs and observers only."""
    errors = hook_result.hook_errors
    if not errors:
        return None
    return "; ".join(f"{e['hook']}: {e['error']}" for e in errors)


def pre_execution_denial(name: str) -> str:
    """What the model reads of a call BEFORE_TOOL_USE refused, a failed hook's
    included: never the hook's error (RFC §9.3)."""
    return f"Tool '{name}' denied by pre-execution hook."


def before_tool_use_detail(hook_result: Any) -> str | None:
    """The error of a BEFORE_TOOL_USE hook that failed closed, for the observers
    only; ``None`` when the call was allowed or a hook deliberately blocked it."""
    return hook_errors_detail(hook_result) if hook_result.failed_closed else None


@dataclasses.dataclass(frozen=True)
class GateRefusal:
    """Why the pre-execution gate refused a call.

    *body* is what the model reads; *detail*, the error of a BEFORE_TOOL_USE
    hook that failed closed, is for the log and the observers only
    (``ToolCallEvent.error_detail``).
    """

    body: str
    detail: str | None = None


def unserved_tool_error(name: str) -> str:
    """The failure a call reports when no handler and no hook served it."""
    return json.dumps({"error": f"No handler for tool {name}"})


def _content_parts(value: Any) -> list[AITextPart | AIImagePart] | None:
    """*value* as content parts, when every item is one or a mapping in a
    part's exact shape; ``None`` otherwise."""
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(item, AITextPart | AIImagePart) for item in value):
        return value
    if not all(isinstance(item, AITextPart | AIImagePart) or _part_shaped(item) for item in value):
        return None
    try:
        return _PARTS.validate_python(value)
    except ValidationError:
        return None


def _part_shaped(item: Any) -> bool:
    """Whether *item* is a mapping with a part's type and only that part's fields."""
    if not isinstance(item, dict):
        return False
    fields = _PART_FIELDS.get(item.get("type"))
    return fields is not None and set(item) <= fields


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
