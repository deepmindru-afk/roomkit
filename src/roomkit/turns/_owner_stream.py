"""The owner's copies of what a turn's stream yielded (RFC §10.6.4).

The reader reads items the executor sent, possibly well ahead of it. What the
channel changed on them afterwards comes in frames of its own: the record, a
call's fixed arguments, a call's early end. They are set on the owner's own
copies, the record as the reader reaches the item it was sent before. The
reads are counted too, a joined burst as the deltas it carries, so each pull
grants the window again and tells the executor how far the reader is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.turns._codec import carries_deltas, decode_end, decode_item
from roomkit.turns.wire import Pull

if TYPE_CHECKING:
    from roomkit.models.response_metadata import ResponseMetadata

PULL_WINDOW = 64
"""How many stream items a pull grants the executor at once (RFC §10.6.4):
text and thinking deltas run ahead of their reader up to this many; any
other item waits for a later pull, which the reader sends when it reads on."""


def _deltas_in(item: dict[str, Any]) -> int:
    """How many deltas a delta item carries: one, or a burst's number."""
    deltas = item.get("deltas", 1)
    if type(deltas) is not int or deltas < 1:
        raise ValueError(f"a burst carries a positive number of deltas, not {deltas!r}")
    return deltas


def _cut_end(start: ToolCallStartMarker) -> ToolCallEndMarker:
    """The end of a call its turn's cut left open."""
    return ToolCallEndMarker(
        tool_name=start.tool_name,
        tool_id=start.tool_id,
        arguments=start.ran_with if start.ran_with is not None else start.arguments,
        status="failed",
        error="cancelled",
        outcome="cancelled",
    )


class StreamCopies:
    """The owner's copies of one turn's stream items, and its reader's pace.

    Decoding raises ``KeyError``, ``TypeError`` or ``ValueError`` on a
    payload it cannot read; the turn fails it as an invalid frame.
    """

    def __init__(self) -> None:
        self.record: ResponseMetadata | None = None
        """The turn's one record, the output's, updated in place."""
        # The items the executor may still send unasked, and how many the
        # reader read.
        self._granted = 0
        self._read = 0
        # Records the executor sent ahead of the reader, by frame number.
        self._records: list[tuple[int, dict[str, Any]]] = []
        self._starts: dict[int, ToolCallStartMarker] = {}
        self._early_ends: dict[int, ToolCallEndMarker] = {}

    def pull(self) -> Pull | None:
        """The pull granting the window again once half of it was read: one
        per many deltas, never one per item; ``None`` while enough remains."""
        if self._granted > PULL_WINDOW // 2:
            return None
        pull = Pull(count=PULL_WINDOW - self._granted, read=self._read)
        self._granted = PULL_WINDOW
        return pull

    def count_read(self, item: dict[str, Any]) -> None:
        """Count an item the reader reads: past one that is no delta, the
        executor waits for a pull; a burst counts as the deltas it carries."""
        if carries_deltas(item):
            deltas = _deltas_in(item)
            self._granted -= deltas
            self._read += deltas
        else:
            self._granted = 0
            self._read += 1

    def item(self, seq: int, data: dict[str, Any]) -> Any:
        """A new item, a call's end the same object as its start's ``ended``."""
        if data["kind"] != "tool_end":
            item = decode_item(data)
            if isinstance(item, ToolCallStartMarker):
                self._starts[seq] = item
            return item
        start_seq = data.get("start")
        end = self._early_ends.pop(start_seq, None) if start_seq is not None else None
        if end is None:
            end = decode_end(data)
        start = self._starts.pop(start_seq, None) if start_seq is not None else None
        if start is not None:
            start.ended = end
        return end

    def open_call_ends(self) -> list[ToolCallEndMarker]:
        """The end of every call the stream started and never ended: its own
        when the executor sent it early, else a cut one."""
        ends = []
        for seq in sorted(self._starts):
            start = self._starts.pop(seq)
            end = self._early_ends.pop(seq, None) or start.ended or _cut_end(start)
            start.ended = end
            ends.append(end)
        return ends

    def record_sent(self, seq: int, record: dict[str, Any]) -> None:
        """A record frame: it applies as the reader reaches the item after it."""
        self._records.append((seq, record))

    def apply_records(self, *, before: int | None = None) -> None:
        """Apply the records sent before frame *before* (every one when
        ``None``, once the reader stops): the record a reader reads is the
        one sent before the item it reads, however far the executor read
        ahead, until the turn's end."""
        while self._records and (before is None or self._records[0][0] < before):
            _, record = self._records.pop(0)
            if self.record is not None:
                self.record.clear()
                self.record.update(record)

    def args_fixed(self, start: int, arguments: Any) -> None:
        """The arguments the call started at frame *start* runs with."""
        if (marker := self._starts.get(start)) is not None:
            marker.ran_with = arguments

    def ended_early(self, start: int, end: ToolCallEndMarker) -> None:
        """The end of the call started at frame *start*, known before the item
        carrying it is read."""
        self._early_ends[start] = end
        if (marker := self._starts.get(start)) is not None:
            marker.ended = end
