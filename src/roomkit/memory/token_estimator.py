"""Fast approximate token estimation for context budget management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    MediaContent,
    TextContent,
    VideoContent,
)
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.utils import extract_event_text as _transport_text

if TYPE_CHECKING:
    from roomkit.models.event import RoomEvent


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ~ 4 characters for English text."""
    return len(text) // 4 + 1


def estimate_tool_tokens(tool: AITool) -> int:
    """Estimate tokens for a single tool definition sent to the model.

    Counts the name, description, and the JSON schema of the parameters —
    the parts a provider serializes into the request's tool list.
    """
    total = estimate_tokens(tool.name) + estimate_tokens(tool.description)
    if tool.parameters:
        total += estimate_tokens(json.dumps(tool.parameters))
    return total


def estimate_message_tokens(message: AIMessage) -> int:
    """Estimate tokens for a complete message including role overhead."""
    overhead = 4  # role, delimiters
    if isinstance(message.content, str):
        return overhead + estimate_tokens(message.content)
    total = overhead
    for part in message.content:
        if isinstance(part, AITextPart):
            total += estimate_tokens(part.text)
        elif isinstance(part, AIToolCallPart):
            args_str = (
                json.dumps(part.arguments)
                if isinstance(part.arguments, dict)
                else str(part.arguments)
            )
            total += estimate_tokens(part.name) + estimate_tokens(args_str)
        elif isinstance(part, AIToolResultPart):
            total += estimate_tokens(part.as_text())
        elif isinstance(part, AIImagePart):
            total += 1000  # rough estimate for vision tokens
    return total


def extract_event_text(event: RoomEvent) -> str:
    """The text of ``event`` as the memory layer reads it.

    The transports' extraction (:func:`roomkit.providers.utils.extract_event_text`),
    with one difference: content that carries no text, a media attachment, a
    location, a tool call, is rendered with ``str()`` rather than dropped,
    because it still costs tokens and still belongs in a summary.
    """
    text = _transport_text(event)
    if text or isinstance(event.content, TextContent):
        return text
    return str(event.content)


def estimate_event_tokens(event: RoomEvent, *, text_only: bool = False) -> int:
    """Estimate a RoomEvent without billing media payloads as text.

    ``text_only`` omits approximate vision costs and non-text metadata. A
    rejection based on message length cannot rely on the flat image estimate,
    which may overstate what a small image actually costs.
    """
    content = event.content
    if isinstance(content, CompositeContent):
        return sum(
            estimate_event_tokens(event.model_copy(update={"content": part}), text_only=text_only)
            for part in content.parts
        )
    if isinstance(content, (MediaContent, AudioContent, VideoContent)):
        parts: list[AITextPart | AIImagePart] = []
        text = getattr(content, "caption", None) or getattr(content, "transcript", None)
        if text:
            parts.append(AITextPart(text=text))
        if isinstance(content, MediaContent) and not text_only:
            parts.append(AIImagePart(url=content.url, mime_type=content.mime_type))
        return estimate_message_tokens(AIMessage(role="user", content=parts)) if parts else 0
    if text_only:
        text = _transport_text(event)
        return estimate_tokens(text) if text else 0
    return estimate_tokens(extract_event_text(event))


def estimate_context_tokens(context: AIContext) -> int:
    """Estimate total tokens for an AIContext."""
    total = 0
    if context.system_prompt:
        total += estimate_tokens(context.system_prompt)
    for msg in context.messages:
        total += estimate_message_tokens(msg)
    if context.tools:
        for tool in context.tools:
            total += estimate_tool_tokens(tool)
    return total


def history_budget(
    *,
    max_context_tokens: int,
    safety_margin_ratio: float = 0.15,
    reserved_tokens: int = 0,
    messages: list[AIMessage] | None = None,
    current_event: RoomEvent | None = None,
) -> int:
    """Tokens the conversation history may occupy, once the rest of the prompt is paid for.

    A context window holds the system prompt, the tool
    schemas, whatever pre-built messages the memory layer injects, and the
    history plus the current turn. A trimmer measuring only the history
    can return a result that overflows the very window it was given — which is
    what ``max_context_tokens * (1 - safety_margin_ratio)`` computed alone did.

    So the arithmetic is explicit here, in one place:

    - ``max_context_tokens * (1 - safety_margin_ratio)`` is what the *whole*
      prompt may occupy; the margin is headroom for the model's reply. The
      default duplicates the ``0.15`` the provider constructors declare, and
      stays anyway: it shipped in the public signature (0.57.0), so dropping
      it is an API break, not a cleanup.
    - ``reserved_tokens`` is the non-history part the caller knows about and the
      trimmer cannot see — system prompt and tool schemas. 0 declares "nothing
      besides what is passed here occupies the window".
    - ``messages`` are the injected blocks the trimmer passes through untouched.
      They are not trimmable, so they are subtracted rather than cut.
    - ``current_event`` is the turn the channel appends after retrieving memory.
      It is not history and cannot be trimmed. Its cost is local to this call,
      never added to the wrapper's shared ``reserved_tokens``.

    What remains is the history's, and nothing else's.
    """
    prompt_budget = int(max_context_tokens * (1 - safety_margin_ratio))
    injected = sum(estimate_message_tokens(m) for m in messages or ())
    current = estimate_event_tokens(current_event) if current_event is not None else 0
    return max(0, prompt_budget - reserved_tokens - injected - current)
