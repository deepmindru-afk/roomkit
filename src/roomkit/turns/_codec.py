"""What a turn's objects become on the wire, and back (RFC §10.6.3, §10.6.8).

A stream item (a text delta or one of the stream markers), a channel's output
and an exception each have one wire form. Decoding builds new objects: nothing
the executor holds is handed to the owner.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import traceback
from typing import Any

from pydantic_core import to_jsonable_python

from roomkit.core.exceptions import RemoteTurnError
from roomkit.models.channel import ChannelOutput
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import (
    LoopEndMarker,
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.providers.ai.base import AIImagePart, AITextPart
from roomkit.turns.wire import ErrorRecord, WireOutput

# The packages whose exception classes an owner rebuilds by type. Anything
# else comes back as a RemoteTurnError naming the class: a name read off the
# wire imports nothing and reaches only a class of an already loaded module.
_REBUILT_PACKAGES = ("roomkit", "builtins")


# -- Stream items ---------------------------------------------------------------


def encode_item(item: Any, *, start: int | None = None) -> dict[str, Any]:
    """The wire form of one stream item; *start* is the sequence number of the
    frame that carried a call end's start."""
    if isinstance(item, str):
        return {"kind": "text", "text": item}
    if isinstance(item, ThinkingDeltaMarker):
        return {"kind": "thinking", "thinking": item.thinking}
    if isinstance(item, ToolCallStartMarker):
        return {
            "kind": "tool_start",
            "tool_name": item.tool_name,
            "tool_id": item.tool_id,
            "arguments": item.arguments,
            "ran_with": item.ran_with,
        }
    if isinstance(item, ToolCallEndMarker):
        return {"kind": "tool_end", "start": start, **encode_end(item)}
    if isinstance(item, SegmentBreakMarker):
        return {"kind": "segment_break"}
    if isinstance(item, LoopEndMarker):
        return {"kind": "loop_end", **dataclasses.asdict(item)}
    raise TypeError(f"A turn's stream cannot carry a {type(item).__name__} across the backend")


def decode_item(data: dict[str, Any]) -> Any:
    """A new stream item from its wire form (a call end without its start link)."""
    kind = data["kind"]
    if kind == "text":
        return data["text"]
    if kind == "thinking":
        return ThinkingDeltaMarker(thinking=data["thinking"])
    if kind == "tool_start":
        return ToolCallStartMarker(
            tool_name=data["tool_name"],
            tool_id=data["tool_id"],
            arguments=data["arguments"],
            ran_with=data["ran_with"],
        )
    if kind == "tool_end":
        return decode_end(data)
    if kind == "segment_break":
        return SegmentBreakMarker()
    if kind == "loop_end":
        fields = {f.name for f in dataclasses.fields(LoopEndMarker)}
        return LoopEndMarker(**{k: v for k, v in data.items() if k in fields})
    raise ValueError(f"Unknown stream item kind {kind!r}")


def encode_end(end: ToolCallEndMarker) -> dict[str, Any]:
    """A call end's fields, its result in a form that keeps its type."""
    return {
        "tool_name": end.tool_name,
        "tool_id": end.tool_id,
        "arguments": end.arguments,
        "result": _encode_result(end.result),
        "status": end.status,
        "duration_ms": end.duration_ms,
        "error": end.error,
        "structured_content": end.structured_content,
        "outcome": end.outcome,
        "refused_but_ran": end.refused_but_ran,
    }


def decode_end(data: dict[str, Any]) -> ToolCallEndMarker:
    """A new call end from its fields."""
    return ToolCallEndMarker(
        tool_name=data["tool_name"],
        tool_id=data["tool_id"],
        arguments=data["arguments"],
        result=_decode_result(data["result"]),
        status=data["status"],
        duration_ms=data["duration_ms"],
        error=data["error"],
        structured_content=data["structured_content"],
        outcome=data["outcome"],
        refused_but_ran=data["refused_but_ran"],
    )


def _encode_result(result: Any) -> dict[str, Any]:
    """A tool result: text, content parts, or a JSON value."""
    if (
        isinstance(result, list)
        and result
        and all(isinstance(part, AITextPart | AIImagePart) for part in result)
    ):
        return {"parts": [part.model_dump(mode="json") for part in result]}
    return {"value": result}


def _decode_result(data: dict[str, Any]) -> Any:
    if "parts" in data:
        return [
            AITextPart.model_validate(p)
            if p.get("type") == "text"
            else AIImagePart.model_validate(p)
            for p in data["parts"]
        ]
    return data["value"]


# -- Channel output -------------------------------------------------------------


def record_payload(record: Any) -> dict[str, Any]:
    """A response-metadata record as it crosses: a deep copy JSON carries,
    a value JSON has no form for written as its text. One rule for the
    record wherever it crosses, at the output and on every later change."""
    return to_jsonable_python(dict(record), fallback=str)


def encode_output(output: ChannelOutput) -> WireOutput:
    """A channel's output without its stream; its error as a record."""
    return WireOutput(
        responded=output.responded,
        response_events=output.response_events,
        response_metadata=record_payload(output.response_metadata),
        tasks=output.tasks,
        observations=output.observations,
        metadata_updates=output.metadata_updates,
        error=None if output.error is None else error_record(output.error),
    )


def decode_output(wire: WireOutput) -> ChannelOutput:
    """A new channel output; its record is the one the turn's frames update."""
    return ChannelOutput(
        responded=wire.responded,
        response_events=wire.response_events,
        response_metadata=ResponseMetadata(wire.response_metadata),
        tasks=wire.tasks,
        observations=wire.observations,
        metadata_updates=wire.metadata_updates,
        error=None if wire.error is None else rebuild_error(wire.error),
    )


# -- Errors ---------------------------------------------------------------------


def error_record(exc: BaseException) -> ErrorRecord:
    """*exc* as it crosses, with the chain of its causes."""
    causes: list[ErrorRecord] = []
    cause = exc.__cause__
    while cause is not None and len(causes) < 16:
        causes.append(_one_record(cause))
        cause = cause.__cause__
    record = _one_record(exc)
    record.causes = causes
    record.traceback = "".join(traceback.format_exception(exc))
    return record


def _one_record(exc: BaseException) -> ErrorRecord:
    cls = type(exc)
    return ErrorRecord(
        kind=f"{cls.__module__}:{cls.__qualname__}",
        message=str(exc),
        args=[a if _is_json(a) else str(a) for a in exc.args],
        attributes={k: v for k, v in vars(exc).items() if _is_json(v)},
    )


def rebuild_error(record: ErrorRecord) -> BaseException:
    """One new exception per record, of the record's class when the owner
    rebuilds it, with its causes chained."""
    exc = _rebuild_one(record)
    current = exc
    for cause_record in record.causes:
        cause = _rebuild_one(cause_record)
        current.__cause__ = cause
        current = cause
    if record.traceback:
        exc.add_note(f"Raised on the turn's executor:\n{record.traceback.rstrip()}")
    return exc


def _rebuild_one(record: ErrorRecord) -> BaseException:
    cls = _exception_class(record.kind)
    if cls is None:
        return RemoteTurnError(record.message, kind=record.kind)
    try:
        exc = cls.__new__(cls, *record.args)
    except Exception:
        return RemoteTurnError(record.message, kind=record.kind)
    exc.args = tuple(record.args)
    exc.__dict__.update(record.attributes)
    return exc


def _exception_class(kind: str) -> type[Exception] | None:
    """The class *kind* names: an ``Exception`` defined at the top of a
    loaded module of RoomKit or Python's builtins; ``None`` for any other
    (an interpreter exit or a cancellation is never raised from a name)."""
    module_name, _, name = kind.partition(":")
    if module_name.split(".", 1)[0] not in _REBUILT_PACKAGES or not name.isidentifier():
        return None
    target = getattr(sys.modules.get(module_name), name, None)
    if not isinstance(target, type) or not issubclass(target, Exception):
        return None
    return target if target.__module__ == module_name else None


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
