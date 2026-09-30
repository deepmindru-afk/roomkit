"""What an emergency compaction makes of a turn's messages (RFC §6.4).

A provider that refuses a turn's context as too long mid-loop gets one
compacted replay. The turn's input and its notes stay whole: the history
before the input is summarized, and the results of the turn's older tool
rounds are stored for re-reading like any large result, a preview in their
place. Every call keeps its result, and no two user messages follow each
other.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from roomkit.channels._tool_eviction import ToolEviction, is_eviction_placeholder
from roomkit.providers.ai.base import AIMessage, AITextPart, AIToolResultPart

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart

# A result of an older round longer than this is stored at compaction, with a
# preview this long: its placeholder costs less than what it replaces.
_STORED_OVER_CHARS = 2000
_STORED_PREVIEW_CHARS = 1000

# How much of each summarized message the summary quotes.
_SUMMARY_MESSAGE_CHARS = 500
_SUMMARY_PART_CHARS = 200

SUMMARY_HEADER = "[Context compacted — earlier conversation summary]"

# A delimited tool result, closed or cut: the summary names it rather than
# quoting a block its truncation could leave open.
_FENCED_RESULT = re.compile(r"<tool_result>.*?(?:</tool_result>|\Z)", re.DOTALL)


def compaction_cut(messages: list[AIMessage], turn_input: AIMessage | None) -> tuple[int, int]:
    """Where the summarized messages end, and where the shortened ones end.

    The first half is summarized, never parting a call from its result. When
    the turn's input falls in that half, what precedes the input is
    summarized instead, and the rounds after it, up to the half, shortened.
    """
    half = _past_results(messages, len(messages) // 2)
    at = next((i for i, message in enumerate(messages) if message is turn_input), None)
    if at is None or at >= half:
        return half, half
    return at, half


def _past_results(messages: list[AIMessage], index: int) -> int:
    """*index*, moved past the tool results there, which belong to the call before."""
    while index < len(messages) and messages[index].role == "tool":
        index += 1
    return index


def summary_text(messages: list[AIMessage]) -> str | None:
    """The summary of *messages*, one line each, or ``None`` when there are none."""
    if not messages:
        return None
    lines = [f"[{message.role}]: {_quoted(message)}" for message in messages]
    return "\n".join([SUMMARY_HEADER, *lines])


def _quoted(message: AIMessage) -> str:
    """What the summary quotes of *message*: its text, cut short, a delimited
    tool result named rather than quoted."""
    if isinstance(message.content, str):
        text = message.content
    else:
        text = " ".join(
            _FENCED_RESULT.sub("[tool_result]", part.text)[:_SUMMARY_PART_CHARS]
            if isinstance(part, AITextPart)
            else f"[{part.type}]"
            for part in message.content
        )
    return _FENCED_RESULT.sub("[tool_result]", text)[:_SUMMARY_MESSAGE_CHARS]


def with_results_stored(messages: list[AIMessage], eviction: ToolEviction) -> list[AIMessage]:
    """*messages* with each long tool result stored for re-reading, a preview
    in its place."""
    return [
        message.model_copy(
            update={"content": [_stored(part, eviction) for part in message.content]}
        )
        if message.role == "tool" and isinstance(message.content, list)
        else message
        for message in messages
    ]


def _stored(part: object, eviction: ToolEviction) -> object:
    """*part* with its result stored when it is a long one, as it is otherwise."""
    if not isinstance(part, AIToolResultPart):
        return part
    result = part.result
    if isinstance(result, str):
        if len(result) <= _STORED_OVER_CHARS or is_eviction_placeholder(result):
            return part
        stored = eviction.evict(result, part.tool_call_id, _STORED_PREVIEW_CHARS)
        return part.model_copy(update={"result": stored})
    text = "\n".join(p.text for p in result if isinstance(p, AITextPart))
    if len(text) <= _STORED_OVER_CHARS:
        return part
    parts = eviction.evict_parts(result, part.tool_call_id, _STORED_PREVIEW_CHARS)
    return part.model_copy(update={"result": parts})


def with_summary(summary: str | None, messages: list[AIMessage]) -> list[AIMessage]:
    """*summary* ahead of *messages*, joined to the first when it is a user
    message: some chat formats refuse two user messages in a row."""
    if summary is None:
        return messages
    first = messages[0] if messages else None
    if first is None or first.role != "user":
        return [AIMessage(role="user", content=summary), *messages]
    if isinstance(first.content, str):
        content: str | list[_ContentPart] = f"{summary}\n\n{first.content}"
    else:
        content = [AITextPart(text=summary), *first.content]
    return [first.model_copy(update={"content": content}), *messages[1:]]
