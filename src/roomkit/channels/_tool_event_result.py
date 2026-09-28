"""What a ``TOOL_CALL_END`` event keeps of a tool's result.

The event is persisted, broadcast and handed to the hooks of the event
pipeline, so a screenshot's base64 must not ride along without bound. The
model's own copy of the result is not this: it keeps every image, and later
turns replay only a result's text.
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import AIImagePart, AITextPart

# Data-URL characters of images an event keeps: the bound the MCP provider
# puts on structuredContent, the other payload these events carry.
TOOL_EVENT_IMAGE_MAX_CHARS = 512 * 1024


def _left_out(part: AIImagePart) -> AITextPart:
    """The note standing for an image the event does not keep."""
    kind = f"image {part.mime_type}" if part.mime_type else "image"
    decoded_kb = max(1, len(part.url) * 3 // 4 // 1024)
    return AITextPart(text=f"[{kind}, {decoded_kb} KB, not kept in the event]")


def tool_event_result(result: Any) -> Any:
    """*result* as a ``TOOL_CALL_END`` event keeps it.

    Each image is kept if the data URLs kept so far, its own included, stay
    within :data:`TOOL_EVENT_IMAGE_MAX_CHARS`; one that would pass the bound
    becomes a note naming its type and size, and a later, smaller one may
    still be kept. A result that is not a content-part list is returned as it
    is.
    """
    if not isinstance(result, list):
        return result
    kept: list[Any] = []
    used = 0
    for part in result:
        if not isinstance(part, AIImagePart):
            kept.append(part)
        elif used + len(part.url) > TOOL_EVENT_IMAGE_MAX_CHARS:
            kept.append(_left_out(part))
        else:
            used += len(part.url)
            kept.append(part)
    return kept
