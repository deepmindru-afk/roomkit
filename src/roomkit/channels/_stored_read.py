"""Reading a stored tool result back: a page of it, or a search in it (RFC §21.5).

A result too large for the model's context is stored and previewed
(``ToolEviction``); what the model reads of it afterwards comes from here,
always bounded to *budget* chars so the answer is never stored again.
"""

from __future__ import annotations

import json
from typing import Any

# The lines shown around each line a search finds.
_SEARCH_CONTEXT_LINES = 2


def page(full_result: str, offset: int, limit: int, budget: int) -> str:
    """The JSON page of *full_result* from line *offset*, at most *limit*
    lines and bounded to *budget* chars."""
    lines = paginable_lines(full_result, budget)
    total_lines = len(lines)

    shown: list[str] = []
    used = 0
    for line in lines[offset : offset + limit]:
        if shown and used + len(line) > budget:
            break
        shown.append(line)
        used += len(line) + 1

    has_more = (offset + len(shown)) < total_lines
    envelope: dict[str, Any] = {
        "content": "\n".join(shown),
        "offset": offset,
        "lines_returned": len(shown),
        "total_lines": total_lines,
        "has_more": has_more,
        "next_offset": offset + len(shown) if has_more else None,
    }
    if has_more:
        # Small models read `content` and skip the pagination fields, then
        # assert "X is not in the result" off one page. Make the partiality
        # impossible to miss and the consequence explicit.
        envelope["warning"] = (
            f"PARTIAL CONTENT — lines {offset + 1}-{offset + len(shown)} of "
            f"{total_lines}. Never conclude that something is absent or that "
            f"this is the complete result until you have read EVERY page; "
            f"continue with offset={offset + len(shown)}."
        )
    # Text as it is: an escaped \uXXXX is six chars, and a page of non-ASCII
    # text would come back past the threshold and be stored again.
    return json.dumps(envelope, ensure_ascii=False)


def search(full_result: str, query: str, offset: int, budget: int) -> str:
    """The JSON matches of *query* in *full_result*, case aside, from its
    *offset*-th match on, bounded as a page is.

    *query* is one line, matched as written and never as a pattern, within
    each line of the result as stored (RFC §21.5): a search that finds
    nothing then says the text is absent from the whole result.
    """
    if "\n" in query or "\r" in query:
        return json.dumps({"error": "query must be one line: a search finds text within a line"})
    lines = full_result.splitlines()
    needle = query.casefold()
    hits = [n for n, line in enumerate(lines) if needle in line.casefold()]
    shown, content = _numbered_windows(lines, hits[offset:], needle, budget - len(query))
    has_more = offset + shown < len(hits)
    envelope: dict[str, Any] = {
        "query": query,
        "content": content,
        "total_matches": len(hits),
        "matches_returned": shown,
        "has_more": has_more,
        "next_offset": offset + shown if has_more else None,
    }
    note = _search_note(len(hits), offset, shown)
    if note:
        envelope["warning" if has_more else "note"] = note
    return json.dumps(envelope, ensure_ascii=False)


def _search_note(total: int, offset: int, shown: int) -> str | None:
    """What the model must not miss about a search's answer, if anything."""
    if total == 0:
        return "No line contains it: the search covered the whole result, so it is absent."
    if offset >= total:
        return (
            f"offset {offset} is past the last match: a search's offset counts matches "
            f"(total_matches={total}), not lines."
        )
    if offset + shown < total:
        return (
            f"PARTIAL CONTENT — matches {offset + 1}-{offset + shown} of {total}. "
            f"Continue with offset={offset + shown} to see the others."
        )
    return None


def paginable_lines(text: str, budget: int) -> list[str]:
    """Lines of ``text``, with lines longer than ``budget`` split into
    budget-sized chunks so every line fits within one page."""
    lines: list[str] = []
    for line in text.splitlines():
        if len(line) <= budget:
            lines.append(line)
        else:
            lines.extend(line[i : i + budget] for i in range(0, len(line), budget))
    return lines


def _numbered_windows(
    lines: list[str], hits: list[int], needle: str, budget: int
) -> tuple[int, str]:
    """How many of *hits* fit in *budget* chars, and their lines with
    ``_SEARCH_CONTEXT_LINES`` around each, numbered from 1; windows that
    overlap are merged, a gap between two is marked ``...``, and a line longer
    than its share of the budget is cut around *needle*."""
    share = max(80, budget // (2 * _SEARCH_CONTEXT_LINES + 1) - 16)
    shown, used, last = 0, 0, -1
    out: list[str] = []
    for hit in hits:
        window = range(
            max(hit - _SEARCH_CONTEXT_LINES, last + 1),
            min(hit + _SEARCH_CONTEXT_LINES + 1, len(lines)),
        )
        block = [f"{n + 1}: {_excerpt(lines[n], needle, share)}" for n in window]
        if out and window and window.start > last + 1:
            block.insert(0, "...")
        size = sum(len(line) + 1 for line in block)
        if used + size > budget:
            break
        out.extend(block)
        used, shown = used + size, shown + 1
        last = max(last, window.stop - 1)
    return shown, "\n".join(out)


def _excerpt(line: str, needle: str, share: int) -> str:
    """*line*, or *share* chars of it around *needle* (its head when it holds
    none), a cut end marked ``[…]``."""
    if len(line) <= share:
        return line
    at = line.casefold().find(needle)
    start = max(0, min(at - share // 2, len(line) - share)) if at >= 0 else 0
    head = "[…]" if start > 0 else ""
    tail = "[…]" if start + share < len(line) else ""
    return f"{head}{line[start : start + share]}{tail}"


# The read-back tool as the model reads it (RFC §21.5).
REREAD_DESCRIPTION = (
    "Read a previously evicted large tool result. "
    "Supports line-based pagination via offset and limit; pages "
    "are size-bounded, so follow next_offset until has_more is "
    "false to read everything. To find something in it, pass "
    "query (one line of text): you get the lines that contain it "
    "with their neighbours and line numbers, searched across the "
    "whole result, so no match means it is not there."
)
REREAD_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "result_id": {
            "type": "string",
            "description": "The evicted result ID shown in the preview.",
        },
        "query": {
            "type": "string",
            "description": (
                "One line of text to find, case aside (matched as written, "
                "not a pattern). Returns only the lines that contain it, with "
                "the lines around them."
            ),
        },
        "offset": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": (
                "Line number to start reading from; with query, the match "
                "to start from (next_offset)."
            ),
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "default": 800,
            "description": "Maximum number of lines to return (pages only).",
        },
    },
    "required": ["result_id"],
}
