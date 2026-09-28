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


def tool_event_result(result: Any) -> Any:
    """*result* as a ``TOOL_CALL_END`` event keeps it.

    Images are kept in order while their data URLs add up to
    :data:`TOOL_EVENT_IMAGE_MAX_CHARS`; each one past it becomes a note naming
    its type and size. A result that is not a content-part list is returned
    as it is.
    """
    if not isinstance(result, list):
        return result
    kept: list[Any] = []
    used = 0
    for part in result:
        if isinstance(part, AIImagePart):
            size = len(part.url)
            if used + size > TOOL_EVENT_IMAGE_MAX_CHARS:
                mime = part.mime_type or "image"
                note = f"[image {mime}, {size // 1024} KB, not kept in the event]"
                kept.append(AITextPart(text=note))
                continue
            used += size
        kept.append(part)
    return kept
