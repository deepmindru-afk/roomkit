"""Text the runtime joins to a user message (RFC §6.4).

The turn's notes, a compaction's summary and a memory provider's summary are
text the runtime places next to a user message. Some chat formats refuse two
user messages in a row, so that text joins the user message it sits next to
rather than forming one of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.providers.ai.base import AIMessage, AITextPart

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart


def joined(message: AIMessage, text: str, *, before: bool) -> AIMessage:
    """*message* with *text* joined before or after its own content: a text
    content stays text, a list gains a text part."""
    content = message.content
    if isinstance(content, str):
        parts = [text, content] if before else [content, text]
        joined_content: str | list[_ContentPart] = "\n\n".join(p for p in parts if p)
    elif before:
        joined_content = [AITextPart(text=text), *content]
    else:
        joined_content = [*content, AITextPart(text=text)]
    return message.model_copy(update={"content": joined_content})


def with_leading_text(text: str | None, messages: list[AIMessage]) -> list[AIMessage]:
    """*text* ahead of *messages*, joined to the first when it is a user message."""
    if text is None:
        return messages
    first = messages[0] if messages else None
    if first is None or first.role != "user":
        return [AIMessage(role="user", content=text), *messages]
    return [joined(first, text, before=True), *messages[1:]]
