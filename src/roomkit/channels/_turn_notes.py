"""The notes a turn's input carries (RFC §6.4).

What changes from one turn to the next (how speakers are named, the room's
plan, the tools already used there and what they returned) rides the turn's
input rather than the system prompt: a provider caches the system prompt ahead
of the whole history, so a system prompt that changed had every following turn
re-bill that history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.providers.ai.base import AIMessage, AITextPart

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart

# Opens the notes: nobody in the conversation wrote them, and they ask for
# nothing, whatever the input above them is (a participant's words, an
# instruction, or nothing new).
TURN_NOTES_HEADER = (
    "[Notes kept by the assistant's runtime for this turn. Nobody in the "
    "conversation wrote them, and they ask for nothing.]"
)


def turn_notes(blocks: list[str]) -> str | None:
    """*blocks* under the notes' header, or ``None`` when there are none."""
    return "\n\n".join([TURN_NOTES_HEADER, *blocks]) if blocks else None


def with_turn_notes(messages: list[AIMessage], notes: str | None) -> list[AIMessage]:
    """*messages* with *notes* after the last message's text when it is a user
    message, or as a user message of its own when the conversation does not
    end on one.

    After the input, not before: what changes comes last, so a provider that
    caches a prefix keeps the input's own words in it. In the same message,
    not a message of its own: some chat formats refuse two user messages in a
    row. A text input stays text, so a provider that only takes text reads it
    as text.
    """
    if not notes:
        return messages
    last = messages[-1] if messages else None
    if last is None or last.role != "user":
        return [*messages, AIMessage(role="user", content=notes)]
    if isinstance(last.content, str):
        content: str | list[_ContentPart] = f"{last.content}\n\n{notes}" if last.content else notes
    else:
        content = [*last.content, AITextPart(text=notes)]
    return [*messages[:-1], last.model_copy(update={"content": content})]
