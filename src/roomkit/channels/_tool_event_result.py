"""What a ``TOOL_CALL_END`` event keeps of a tool's result.

The event is persisted, broadcast and handed to the hooks of the event
pipeline, so a screenshot's base64 must not ride along without bound. The
model's own copy of the result is not this: it keeps every image, and later
turns replay only a result's text.

A result reaches the event in more than one shape: content parts
(:class:`AIImagePart`), or JSON, the shape an ACP agent's tool output and a
structured copy take. Both are walked with one budget per event.
"""

from __future__ import annotations

import re
from typing import Any

from roomkit.providers.ai.base import AIImagePart, AITextPart

# Data-URL or base64 characters of binary payloads an event keeps: the bound
# the MCP provider puts on structuredContent, the other payload these events
# carry.
TOOL_EVENT_IMAGE_MAX_CHARS = 512 * 1024

# An RFC 2397 data URI's header, matched on the first characters only: text
# that merely starts with "data:" (an SSE log, "data: {...}") is not one.
_DATA_URI = re.compile(r"data:([\w.+-]+/[\w.+-]+)?(?:;[\w.+-]+=[^;,\s]*)*(?:;base64)?,", re.ASCII)
_DATA_URI_HEADER_MAX = 256

# JSON deeper than this is kept as it is: no binary block hides that deep in
# a real tool result, and the walk must not recurse without bound.
_MAX_DEPTH = 32


class _Budget:
    """The payload characters an event has kept so far."""

    def __init__(self) -> None:
        self.used = 0

    def keep(self, size: int) -> bool:
        if self.used + size > TOOL_EVENT_IMAGE_MAX_CHARS:
            return False
        self.used += size
        return True


def _note(kind: str, mime: str | None, size: int) -> str:
    """The note standing for a payload the event does not keep."""
    label = f"{kind} {mime}" if mime else kind
    decoded_kb = max(1, size * 3 // 4 // 1024)
    return f"[{label}, {decoded_kb} KB, not kept in the event]"


def _binary_block(node: dict[str, Any]) -> tuple[str, str | None, str] | None:
    """``(kind, mime, payload)`` of a JSON block carrying binary data, or ``None``.

    An image or audio block (MCP and ACP put the base64 in ``data``, Anthropic
    in ``source.data``, a stored :class:`AIImagePart` in ``url``) or a blob
    resource (``blob`` beside its ``uri``, MCP's ``BlobResourceContents``). A
    ``data`` or ``blob`` field in any other dict is somebody's text.
    """
    kind = node.get("type")
    if kind in ("image", "audio"):
        raw_source = node.get("source")
        source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
        payload = node.get("data") or node.get("url") or source.get("data")
        mime = node.get("mimeType") or node.get("mime_type") or source.get("media_type")
    elif isinstance(node.get("blob"), str) and isinstance(node.get("uri"), str):
        kind, payload, mime = "resource", node["blob"], node.get("mimeType")
    else:
        return None
    if not isinstance(payload, str):
        return None
    return str(kind), mime if isinstance(mime, str) else None, payload


def _walk(node: Any, budget: _Budget, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return node
    if isinstance(node, AIImagePart):
        if budget.keep(len(node.url)):
            return node
        return AITextPart(text=_note("image", node.mime_type, len(node.url)))
    if isinstance(node, str):
        header = _DATA_URI.match(node, 0, _DATA_URI_HEADER_MAX)
        if header is None or budget.keep(len(node)):
            return node
        return _note("data URI", header.group(1), len(node))
    if isinstance(node, list):
        return [_walk(item, budget, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node
    block = _binary_block(node)
    if block is None:
        return {key: _walk(value, budget, depth + 1) for key, value in node.items()}
    kind, mime, payload = block
    if budget.keep(len(payload)):
        return node
    note = _note(kind, mime, len(payload))
    if kind == "resource":
        # Still a resource: its uri, and the note as its text.
        return {"uri": node["uri"], "mimeType": "text/plain", "text": note}
    return {"type": "text", "text": note}


def tool_event_payload(result: Any, structured_content: Any) -> tuple[Any, Any]:
    """*result* and *structured_content* as a ``TOOL_CALL_END`` event keeps them.

    Each binary payload (an image part, an image, audio or blob block in
    JSON, a data URI) is kept if the payloads kept so far, its own included,
    stay within :data:`TOOL_EVENT_IMAGE_MAX_CHARS`, one budget for both
    fields; one that would pass the bound becomes a note naming its kind,
    type and size, and a later, smaller one may still be kept. Everything
    else is returned as it is.

    The structured copy is walked first: it is the one written for UI
    surfaces, where a note in place of an image breaks the payload's schema,
    while in the result a note is one more text part.
    """
    budget = _Budget()
    structured = _walk(structured_content, budget)
    return _walk(result, budget), structured


def tool_event_result(result: Any) -> Any:
    """*result* as a ``TOOL_CALL_END`` event with no structured copy keeps it."""
    return tool_event_payload(result, None)[0]
