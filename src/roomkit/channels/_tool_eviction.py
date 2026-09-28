"""Large tool result eviction and paginated re-reading."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Any

from roomkit.providers.ai.base import AIImagePart, AITextPart, AITool

logger = logging.getLogger("roomkit.channels.ai")

_MAX_EVICTED = 50

# Chars reserved per read_stored_result page for the JSON envelope — fixed
# keys plus the partial-page warning prose (~350 chars escaped, worst case) —
# and a margin, so worst-case escaping of the content still leaves the page
# under the re-eviction bound (4 * threshold_tokens chars). See handle_read.
_PAGE_ENVELOPE_CHARS = 512

# Ceiling of the preview that stands in for an evicted result, in chars. The
# budget also shrinks with the threshold (see _preview_budget), so a preview
# never carries as much as the result it replaces.
_PREVIEW_MAX_CHARS = 8000
_PREVIEW_HEAD_LINES = 5
_PREVIEW_TAIL_LINES = 5

# How every eviction placeholder starts. The usage memory recognises one by it:
# a placeholder is what TOOL_CALL_END persists for an evicted result, and its
# ``evicted_…`` id dies with the process, so it must not be replayed as data.
EVICTION_PLACEHOLDER_PREFIX = "Result too large ("


def is_eviction_placeholder(text: str) -> bool:
    """Whether *text* is the stand-in :class:`ToolEviction` gave an oversized result."""
    return text.startswith(EVICTION_PLACEHOLDER_PREFIX)


def _omission_marker(count: int) -> str:
    return f"[... {count} lines omitted ...]"


def _truncation_marker(count: int) -> str:
    return f" [... {count} chars truncated ...]"


def _clip_line(line: str, room: int) -> str | None:
    """*line* cut to fit *room* chars, the cut stated in a marker; ``None``
    when the marker leaves no room for any of the line."""
    # Sized for the widest count, so the real marker is never longer.
    keep = room - len(_truncation_marker(len(line)))
    if keep < 1:
        return None
    return line[:keep] + _truncation_marker(len(line) - keep)


def _fit_lines(lines: list[str], budget: int) -> list[str]:
    """The leading *lines* that fit in *budget* chars once joined by newlines.

    The first line that does not fit is clipped, if its marker leaves room for
    some of it, and ends the run: the lines after it are not shown.
    """
    kept: list[str] = []
    used = 0
    for line in lines:
        sep = 1 if kept else 0
        room = budget - used - sep
        if len(line) > room:
            clipped = _clip_line(line, room)
            if clipped is not None:
                kept.append(clipped)
            break
        kept.append(line)
        used += sep + len(line)
    return kept


def _preview(result: str, budget: int) -> str:
    """Head/tail preview of *result* within *budget* chars.

    A bound on lines alone would let one giant line (minified HTML, a JSON
    blob) reach the provider whole, which is what eviction exists to prevent.
    The head and the tail split the lines, so even a short result shows its
    last line. The head takes at most half the budget, or what the whole
    tail leaves; the tail takes the rest, filled from the last line. A line
    that does not fit is clipped with a marker, and every line not shown is
    counted in the omission marker.
    """
    lines = result.splitlines()
    # Reserved at its widest, which also covers the newline joining head and
    # tail when nothing is omitted.
    reserve = len(_omission_marker(len(lines))) + 4
    if budget < reserve:
        return ""
    head_n = min(_PREVIEW_HEAD_LINES, (len(lines) + 1) // 2)
    head_src = lines[:head_n]
    tail_src = lines[max(head_n, len(lines) - _PREVIEW_TAIL_LINES) :]
    lines_budget = budget - reserve
    whole_tail = len("\n".join(tail_src))
    head = _fit_lines(head_src, max(lines_budget // 2, lines_budget - whole_tail))
    head_text = "\n".join(head)
    tail = _fit_lines(tail_src[::-1], lines_budget - len(head_text))[::-1]
    tail_text = "\n".join(tail)

    omitted = len(lines) - len(head) - len(tail)
    if not omitted:
        return "\n".join(head + tail)
    return "\n\n".join(part for part in (head_text, _omission_marker(omitted), tail_text) if part)


class ToolEviction:
    """Stores large tool results and provides paginated re-reading.

    When a tool result exceeds ``threshold_tokens``, the full result is
    stored in a FIFO-bounded buffer and replaced with a head/tail preview
    bounded in lines and in chars.
    The ``read_stored_result`` tool definition is injected into the AI
    context so the agent can paginate back through the full output.

    The store is scoped per room: the eviction buffer lives on a channel
    object shared by every room the channel serves, so an unscoped buffer
    would leak one conversation's tool output into another (and inject
    the re-read tool into rooms that evicted nothing). The room comes
    from the tool-loop context; paths outside a loop share one fallback
    scope.
    """

    def __init__(self, threshold_tokens: int = 5000) -> None:
        self.threshold_tokens = threshold_tokens
        self._store: OrderedDict[tuple[str, str], str] = OrderedDict()

    @staticmethod
    def _room_scope() -> str:
        from roomkit.channels.ai import _current_loop_ctx

        ctx = _current_loop_ctx.get()
        return (ctx.room_id if ctx is not None else None) or ""

    @property
    def has_evicted(self) -> bool:
        room = self._room_scope()
        return any(key[0] == room for key in self._store)

    def _estimate(self, text: str) -> int:
        return len(text) // 4 + 1

    def maybe_evict(self, result: str, tool_call_id: str = "") -> str:
        """Evict large results to the store, returning a preview."""
        estimated = self._estimate(result)
        if estimated <= self.threshold_tokens:
            return result

        result_id = f"evicted_{tool_call_id}" if tool_call_id else f"evicted_{id(result)}"
        self._store[(self._room_scope(), result_id)] = result
        while len(self._store) > _MAX_EVICTED:
            self._store.popitem(last=False)

        return (
            f"{EVICTION_PLACEHOLDER_PREFIX}{estimated} tokens). Full output saved as "
            f"'{result_id}'. Use read_stored_result to read it with pagination.\n\n"
            f"Preview:\n{_preview(result, self._preview_budget())}"
        )

    def maybe_evict_parts(
        self, parts: list[AITextPart | AIImagePart], tool_call_id: str = ""
    ) -> list[AITextPart | AIImagePart]:
        """Evict the text of a content-part result, keeping its images in place.

        The text parts are measured joined, so many medium parts cannot add up
        past the threshold unseen. Over it, they are stored as one text and
        replaced by a single placeholder part where the first text part was.
        """
        text = "\n".join(p.text for p in parts if isinstance(p, AITextPart))
        if self._estimate(text) <= self.threshold_tokens:
            return parts
        placeholder: AITextPart | None = AITextPart(text=self.maybe_evict(text, tool_call_id))
        kept: list[AITextPart | AIImagePart] = []
        for part in parts:
            if not isinstance(part, AITextPart):
                kept.append(part)
            elif placeholder is not None:
                kept.append(placeholder)
                placeholder = None
        return kept

    def _preview_budget(self) -> int:
        """Chars the preview may use: the ceiling, or half of what evicts (the
        estimator is len // 4 tokens) when the threshold is small."""
        return min(_PREVIEW_MAX_CHARS, 2 * self.threshold_tokens)

    def handle_read(self, arguments: dict[str, Any]) -> str:
        """Paginate a previously evicted result.

        Pages are size-bounded below the eviction threshold: a page that grew
        past it would itself be evicted on return, re-stored under a new id,
        and the agent would chase evicted results forever. Lines longer than
        the page budget (single-line JSON tool results) are split into chunks
        so they paginate instead of coming back whole.
        """
        result_id = arguments.get("result_id", "")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 800)

        room = self._room_scope()
        full_result = self._store.get((room, result_id))
        if full_result is None:
            available = [rid for scope, rid in self._store if scope == room]
            return json.dumps({"error": f"Result '{result_id}' not found", "available": available})

        # Char budget per page. The page returns as a JSON string (the content
        # re-escaped, wrapped in an envelope) and is re-measured against
        # threshold_tokens on the way back, so it must never itself be evicted
        # — else the agent chases evicted results forever. The estimator is
        # len // 4 tokens, so the page must stay under 4 * threshold_tokens
        # chars. JSON escaping can double the content (already-escaped tool
        # output re-escapes worst-case ~2x) and the envelope adds a fixed
        # overhead, so bound the raw content at 2 * threshold_tokens minus an
        # envelope allowance: worst case 2*budget + envelope stays under
        # 4 * threshold_tokens.
        budget = max(1, self.threshold_tokens * 2 - _PAGE_ENVELOPE_CHARS)
        lines = self._paginable_lines(full_result, budget)
        total_lines = len(lines)

        page: list[str] = []
        used = 0
        for line in lines[offset : offset + limit]:
            if page and used + len(line) > budget:
                break
            page.append(line)
            used += len(line) + 1

        has_more = (offset + len(page)) < total_lines
        envelope: dict[str, Any] = {
            "content": "\n".join(page),
            "offset": offset,
            "lines_returned": len(page),
            "total_lines": total_lines,
            "has_more": has_more,
            "next_offset": offset + len(page) if has_more else None,
        }
        if has_more:
            # Small models read `content` and skip the pagination fields, then
            # assert "X is not in the result" off one page. Make the partiality
            # impossible to miss and the consequence explicit.
            envelope["warning"] = (
                f"PARTIAL CONTENT — lines {offset + 1}-{offset + len(page)} of "
                f"{total_lines}. Never conclude that something is absent or that "
                f"this is the complete result until you have read EVERY page; "
                f"continue with offset={offset + len(page)}."
            )
        return json.dumps(envelope)

    @staticmethod
    def _paginable_lines(text: str, budget: int) -> list[str]:
        """Lines of ``text``, with lines longer than ``budget`` split into
        budget-sized chunks so every line fits within one page."""
        lines: list[str] = []
        for line in text.splitlines():
            if len(line) <= budget:
                lines.append(line)
            else:
                lines.extend(line[i : i + budget] for i in range(0, len(line), budget))
        return lines

    @staticmethod
    def tool_definition() -> AITool:
        """Return the AITool definition for read_stored_result."""
        return AITool(
            name="read_stored_result",
            description=(
                "Read a previously evicted large tool result. "
                "Supports line-based pagination via offset and limit; pages "
                "are size-bounded, so follow next_offset until has_more is "
                "false to read everything."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "The evicted result ID shown in the preview.",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Line number to start reading from.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 800,
                        "description": "Maximum number of lines to return.",
                    },
                },
                "required": ["result_id"],
            },
        )
