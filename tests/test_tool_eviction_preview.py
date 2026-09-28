"""ToolEviction preview bound.

The preview that stands in for an evicted result is what reaches the provider;
the full text stays stored for ``read_stored_result``. The preview is bounded
in lines (5 head, 5 tail) and in chars, so one giant line cannot carry the
whole result to the provider (RMK-258).
"""

from __future__ import annotations

import re

import pytest

from roomkit.channels._tool_eviction import ToolEviction, is_eviction_placeholder

_PREVIEW_LABEL = "\n\nPreview:\n"
_TRUNCATED = re.compile(r" \[\.\.\. (\d+) chars truncated \.\.\.\]")


def _preview(placeholder: str) -> str:
    return placeholder.split(_PREVIEW_LABEL, 1)[1]


def _budget(threshold_tokens: int) -> int:
    """The documented bound: 8000 chars, or half of what evicts if smaller."""
    return min(8000, 2 * threshold_tokens)


def _evict(result: str, threshold_tokens: int = 5000) -> tuple[ToolEviction, str]:
    ev = ToolEviction(threshold_tokens=threshold_tokens)
    return ev, ev.maybe_evict(result, "tc1")


class TestGiantLine:
    def test_giant_line_in_a_long_result_is_clipped(self) -> None:
        """A 12-line result whose third line is 500 KB."""
        lines = [f"line {i}" for i in range(12)]
        lines[2] = "x" * 500_000
        result = "\n".join(lines)

        ev, out = _evict(result)
        preview = _preview(out)

        assert len(preview) <= _budget(5000)
        # The head uses what the short tail leaves, not only half the budget.
        assert len(preview) > _budget(5000) - 100
        match = _TRUNCATED.search(preview)
        assert match is not None
        kept = preview[: match.start()].split("\n")[-1]
        assert set(kept) == {"x"}
        assert len(kept) + int(match.group(1)) == 500_000
        # The tail still shows the end of the result, and the skipped lines
        # are counted.
        assert preview.endswith("line 11")
        assert "lines omitted" in preview
        # Only the preview is bounded: the stored text is whole.
        assert ev._store[("", "evicted_tc1")] == result

    def test_giant_line_in_a_short_result_is_marked(self) -> None:
        """A result of ten lines or fewer is clipped with a marker too."""
        result = "first\n" + "y" * 100_000 + "\nlast"

        _, out = _evict(result)
        preview = _preview(out)

        assert len(preview) <= _budget(5000)
        assert preview.startswith("first\n")
        assert _TRUNCATED.search(preview) is not None

    def test_short_result_still_shows_its_last_line(self) -> None:
        """A giant line does not hide the end of a result of a few lines."""
        result = "HEADER\n" + "x" * 50_000 + "\nSTATUS: done"

        _, out = _evict(result, threshold_tokens=100)
        preview = _preview(out)

        assert len(preview) <= _budget(100)
        assert preview.startswith("HEADER\n")
        assert preview.endswith("STATUS: done")

    def test_single_line_uses_the_whole_budget(self) -> None:
        """With no tail to share with, the head gets the whole budget."""
        _, out = _evict("z" * 100_000)
        preview = _preview(out)

        assert len(preview) <= _budget(5000)
        assert len(preview) > _budget(5000) - 50

    def test_giant_last_line_does_not_hide_the_end(self) -> None:
        """The tail is filled from the end: the last line is shown first."""
        lines = [f"row {i}" for i in range(30)]
        lines[-3] = "w" * 200_000
        _, out = _evict("\n".join(lines))
        preview = _preview(out)

        assert len(preview) <= _budget(5000)
        assert preview.endswith("row 28\nrow 29")


class TestShortLines:
    def test_short_lines_keep_the_head_tail_format(self) -> None:
        """Results made of ordinary lines keep the head/tail format."""
        lines = [f"line {i} " + "x" * 50 for i in range(5000)]
        _, out = _evict("\n".join(lines))

        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-5:])
        assert _preview(out) == f"{head}\n\n[... 4990 lines omitted ...]\n\n{tail}"

    def test_small_threshold_shrinks_the_preview(self) -> None:
        """At 1000 tokens (eviction past 4000 chars), the preview of a
        5000-char result of a few lines is shorter than the result."""
        result = "\n".join("v" * 999 for _ in range(5))
        _, out = _evict(result, threshold_tokens=1000)

        assert len(_preview(out)) <= _budget(1000)
        assert len(_preview(out)) < len(result)


def _layouts() -> dict[str, str]:
    giant = "g" * 150_000
    short = [f"line {i}" for i in range(40)]
    return {
        "one-giant-line": giant,
        "giant-first": "\n".join([giant, *short]),
        "giant-middle": "\n".join([*short[:20], giant, *short[20:]]),
        "giant-last": "\n".join([*short, giant]),
        "giant-in-tail": "\n".join([*short, giant, "end"]),
        "all-giant": "\n".join([giant[:20_000]] * 12),
        "exactly-ten": "\n".join([giant[:15_000]] * 10),
        "eleven": "\n".join([giant[:15_000]] * 11),
        "blank-lines": "\n" * 100_000,
        "crlf": "\r\n".join([giant[:30_000]] * 4),
        "many-short": "\n".join(f"r{i}" for i in range(60_000)),
    }


@pytest.mark.parametrize("threshold_tokens", [1, 10, 100, 1000, 2000, 5000, 20_000])
@pytest.mark.parametrize("layout", sorted(_layouts()))
def test_preview_never_exceeds_the_budget(layout: str, threshold_tokens: int) -> None:
    """Whatever the line layout and the threshold, the preview stays within
    the budget, and the stored text is the tool's result, untouched."""
    result = _layouts()[layout]
    ev, out = _evict(result, threshold_tokens)

    assert is_eviction_placeholder(out)
    assert len(_preview(out)) <= _budget(threshold_tokens)
    assert ev._store[("", "evicted_tc1")] == result
