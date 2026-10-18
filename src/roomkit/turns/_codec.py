"""What a turn's objects become on the wire, and back (RFC §10.6.3).

A stream item (a text delta or one of the stream markers) and a channel's
output each have one wire form (an exception's is ``_error_codec``'s).
Decoding builds new objects: nothing the executor holds is handed to the
owner.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from pydantic_core import to_jsonable_python

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
from roomkit.turns._error_codec import ErrorClasses, error_record
from roomkit.turns.wire import WireOutput

# -- Stream items ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _ItemKind:
    """One kind of stream item: its type, and its wire form both ways."""

    name: str
    type: type
    encode: Callable[[Any, int | None], dict[str, Any]]
    """The item's fields; the second argument is a call end's start frame."""
    decode: Callable[[dict[str, Any]], Any]


def _encode_start(item: ToolCallStartMarker, start: int | None) -> dict[str, Any]:
    return {
        "tool_name": item.tool_name,
        "tool_id": item.tool_id,
        "arguments": item.arguments,
        "ran_with": item.ran_with,
    }


def _decode_start(data: dict[str, Any]) -> ToolCallStartMarker:
    return ToolCallStartMarker(
        tool_name=data["tool_name"],
        tool_id=data["tool_id"],
        arguments=data["arguments"],
        ran_with=data["ran_with"],
    )


def _decode_loop_end(data: dict[str, Any]) -> LoopEndMarker:
    fields = {f.name for f in dataclasses.fields(LoopEndMarker)}
    return LoopEndMarker(**{k: v for k, v in data.items() if k in fields})


_ITEM_KINDS = (
    _ItemKind("text", str, lambda item, start: {"text": item}, lambda data: data["text"]),
    _ItemKind(
        "thinking",
        ThinkingDeltaMarker,
        lambda item, start: {"thinking": item.thinking},
        lambda data: ThinkingDeltaMarker(thinking=data["thinking"]),
    ),
    _ItemKind("tool_start", ToolCallStartMarker, _encode_start, _decode_start),
    _ItemKind(
        "tool_end",
        ToolCallEndMarker,
        lambda item, start: {"start": start, **encode_end(item)},
        lambda data: decode_end(data),
    ),
    _ItemKind(
        "segment_break",
        SegmentBreakMarker,
        lambda item, start: {},
        lambda data: SegmentBreakMarker(),
    ),
    _ItemKind(
        "loop_end", LoopEndMarker, lambda item, start: dataclasses.asdict(item), _decode_loop_end
    ),
)
_KIND_OF_TYPE = {kind.type: kind for kind in _ITEM_KINDS}
_KIND_NAMED = {kind.name: kind for kind in _ITEM_KINDS}


def encode_item(item: Any, *, start: int | None = None) -> dict[str, Any]:
    """The wire form of one stream item; *start* is the sequence number of the
    frame that carried a call end's start."""
    kind = _KIND_OF_TYPE.get(type(item)) or next(
        (k for k in _ITEM_KINDS if isinstance(item, k.type)), None
    )
    if kind is None:
        raise TypeError(f"A turn's stream cannot carry a {type(item).__name__} across the backend")
    return {"kind": kind.name, **kind.encode(item, start)}


def decode_item(data: dict[str, Any]) -> Any:
    """A new stream item from its wire form (a call end without its start link)."""
    kind = _KIND_NAMED.get(data["kind"])
    if kind is None:
        raise ValueError(f"Unknown stream item kind {data['kind']!r}")
    return kind.decode(data)


def encode_end(end: ToolCallEndMarker) -> dict[str, Any]:
    """A call end's fields, its result in a form that keeps its type."""
    return {
        "tool_name": end.tool_name,
        "tool_id": end.tool_id,
        "arguments": end.arguments,
        "result": encode_result(end.result),
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
        result=decode_result(data["result"]),
        status=data["status"],
        duration_ms=data["duration_ms"],
        error=data["error"],
        structured_content=data["structured_content"],
        outcome=data["outcome"],
        refused_but_ran=data["refused_but_ran"],
    )


def encode_result(result: Any) -> dict[str, Any]:
    """A tool result: text, content parts, or a JSON value."""
    if (
        isinstance(result, list)
        and result
        and all(isinstance(part, AITextPart | AIImagePart) for part in result)
    ):
        return {"parts": [part.model_dump(mode="json") for part in result]}
    return {"value": result}


def decode_result(data: dict[str, Any]) -> Any:
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


def decode_output(wire: WireOutput, errors: ErrorClasses) -> ChannelOutput:
    """A new channel output, its error of a class *errors* rebuilds; its
    record is the one the turn's frames update."""
    return ChannelOutput(
        responded=wire.responded,
        response_events=wire.response_events,
        response_metadata=ResponseMetadata(wire.response_metadata),
        tasks=wire.tasks,
        observations=wire.observations,
        metadata_updates=wire.metadata_updates,
        error=None if wire.error is None else errors.rebuild(wire.error),
    )
