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

    The text is matched as it is, never as a pattern (RFC §21.5), and the
    search covers the whole result, so no match says the text is absent.
    """
    if not query.strip():
        return json.dumps({"error": "query must not be empty"})
    lines = paginable_lines(full_result, budget)
    needle = query.casefold()
    hits = [n for n, line in enumerate(lines) if needle in line.casefold()]
    shown, content = _numbered_windows(lines, hits[offset:], budget)
    has_more = offset + shown < len(hits)
    envelope: dict[str, Any] = {
        "query": query,
        "content": content,
        "total_matches": len(hits),
        "matches_returned": shown,
        "has_more": has_more,
        "next_offset": offset + shown if has_more else None,
    }
    if not hits:
        envelope["note"] = "No line contains it: the search covered the whole result."
    return json.dumps(envelope)


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


def _numbered_windows(lines: list[str], hits: list[int], budget: int) -> tuple[int, str]:
    """How many of *hits* fit in *budget* chars, and their lines with
    ``_SEARCH_CONTEXT_LINES`` around each, numbered from 1; windows that
    overlap are merged, and a gap between two is marked ``...``."""
    shown, used, last = 0, 0, -1
    out: list[str] = []
    for hit in hits:
        window = range(
            max(hit - _SEARCH_CONTEXT_LINES, last + 1),
            min(hit + _SEARCH_CONTEXT_LINES + 1, len(lines)),
        )
        block = [f"{n + 1}: {lines[n]}" for n in window]
        size = sum(len(line) + 1 for line in block)
        if shown and used + size > budget:
            break
        if out and window and window.start > last + 1:
            out.append("...")
        out.extend(block)
        used, shown = used + size, shown + 1
        last = max(last, window.stop - 1)
    return shown, "\n".join(out)


# The read-back tool as the model reads it (RFC §21.5).
REREAD_DESCRIPTION = (
    "Read a previously evicted large tool result. "
    "Supports line-based pagination via offset and limit; pages "
    "are size-bounded, so follow next_offset until has_more is "
    "false to read everything. To find something in it, pass "
    "query: you get the lines that contain it with their "
    "neighbours and line numbers, searched across the whole "
    "result, so no match means it is not there."
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
                "Text to find, case aside (matched as written, not a "
                "pattern). Returns only the lines that contain it, with "
                "the lines around them."
            ),
        },
        "offset": {
            "type": "integer",
            "default": 0,
            "description": (
                "Line number to start reading from; with query, the match "
                "to start from (next_offset)."
            ),
        },
        "limit": {
            "type": "integer",
            "default": 800,
            "description": "Maximum number of lines to return.",
        },
    },
    "required": ["result_id"],
}
