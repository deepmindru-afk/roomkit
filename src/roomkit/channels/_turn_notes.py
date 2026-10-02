"""The notes a turn's input carries (RFC §6.4).

What changes from one turn to the next (how speakers are named, the room's
plan, the tools already used there and what they returned, what the memory
retrieved for the turn) rides the turn's input rather than the system prompt
or the history: a provider caches the system prompt ahead
of the whole history, so a system prompt that changed had every following turn
re-bill that history.

A ``BEFORE_AI_GENERATION`` hook adds to them with :func:`add_turn_note`, and a
reader that shows the input apart from its notes (a debug view) separates them
with :func:`split_turn_notes`.
"""

from __future__ import annotations

from roomkit.channels._user_text import joined
from roomkit.providers.ai.base import AIMessage, AITextPart

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
    return [*messages[:-1], joined(last, notes, before=False)]


def turn_input(messages: list[AIMessage]) -> AIMessage | None:
    """The turn's input among *messages* as its first round is built: the last
    message when it is a user message, the one that carries the turn's notes."""
    last = messages[-1] if messages else None
    return last if last is not None and last.role == "user" else None


def add_turn_note(messages: list[AIMessage], block: str) -> list[AIMessage]:
    """*messages* with *block* added to the turn's notes (RFC §6.4).

    The block joins the section the channel opened, under its one header,
    when the turn's input carries it, and opens the section as
    :func:`with_turn_notes` does otherwise. Either way the notes read exactly
    as if they had been assembled at once (the same header, the blocks joined
    by a blank line, a text input still text, an input with images keeping
    its notes in one text part), so the prefix a provider caches is the same.

    For a ``BEFORE_AI_GENERATION`` hook::

        event.ai_context.messages = add_turn_note(event.ai_context.messages, block)
    """
    last = turn_input(messages)
    if last is None or not _carries_notes(last):
        return with_turn_notes(messages, turn_notes([block]))
    return [*messages[:-1], _with_block(last, block)]


def split_turn_notes(text: str) -> tuple[str, str]:
    """*text* as the turn's input and its notes, cut where the header opens
    them; the notes are empty when the text carries none.

    The header is the notes' only mark, and it always opens a paragraph, so
    the cut is at its last occurrence that does: an input that quotes the
    header as a paragraph of its own is the one this misreads.
    """
    at = _notes_at(text)
    if at < 0:
        return text, ""
    return text[:at].removesuffix("\n\n"), text[at:]


def _notes_at(text: str) -> int:
    """Where the turn's notes open in *text*, or ``-1``: the header's last
    occurrence at the start of a paragraph, as the channel places it."""
    at = text.rfind(f"\n\n{TURN_NOTES_HEADER}")
    if at >= 0:
        return at + 2
    return 0 if text.startswith(TURN_NOTES_HEADER) else -1


def _carries_notes(message: AIMessage) -> bool:
    """Whether *message* ends on the turn's notes."""
    content = message.content
    if isinstance(content, str):
        return _notes_at(content) >= 0
    tail = content[-1] if content else None
    return isinstance(tail, AITextPart) and tail.text.startswith(TURN_NOTES_HEADER)


def _with_block(message: AIMessage, block: str) -> AIMessage:
    """*message*, whose content ends on the turn's notes, with *block* after them."""
    content = message.content
    if isinstance(content, str):
        return message.model_copy(update={"content": f"{content}\n\n{block}"})
    notes = content[-1]
    assert isinstance(notes, AITextPart)  # _carries_notes  # noqa: S101
    tail = AITextPart(text=f"{notes.text}\n\n{block}")
    return message.model_copy(update={"content": [*content[:-1], tail]})
