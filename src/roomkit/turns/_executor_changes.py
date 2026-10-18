"""What a turn's channel changes on what it already yielded (RFC §10.6.4).

The channel keeps writing into objects the owner holds copies of: its
response-metadata record, a call's arguments once its gate fixed them, a
call's end before the end itself is read. Before each item the executor
sends, it sends what changed, so the owner's copies read as the channel's own.
Before a delta, a record that counts its writes counts as changed only once
one of its keys was written: a stream of deltas that leaves it alone costs no
copy of it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, NoReturn

from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.turns._codec import encode_end, encode_item, is_delta, record_payload
from roomkit.turns.wire import Item, Record, ToolArgs, ToolEnded

if TYPE_CHECKING:
    from roomkit.turns.wire import Frame


def _writes_of(record: Mapping[str, Any]) -> int | None:
    """The writes *record* took; ``None`` for a mapping that does not count
    them (a bare dict a channel set past the output's validation), which is
    then compared before every item."""
    return record.writes if isinstance(record, ResponseMetadata) else None


class ChannelChanges:
    """The changes of one turn's channel, sent before the items they precede."""

    def __init__(
        self,
        send: Callable[[Frame], Awaitable[int]],
        refuse: Callable[[str, Exception], Awaitable[NoReturn]],
    ) -> None:
        self._send = send
        self._refuse = refuse
        self._live: Mapping[str, Any] | None = None
        # The record as last sent, and the writes it had taken when copied.
        self._record: dict[str, Any] = {}
        self._writes: int | None = None
        # The calls the stream started and has not ended, by their start's frame.
        self._open: dict[int, ToolCallStartMarker] = {}
        self._args_sent: dict[int, Any] = {}
        self._ends_sent: set[int] = set()

    def watch(self, record: Mapping[str, Any]) -> None:
        """Watch the channel's live record, its output's."""
        self._live = record

    def sent(self, record: dict[str, Any]) -> None:
        """The record as the output carried it: a change is one from it."""
        self._record = record
        if self._live is not None:
            self._writes = _writes_of(self._live)

    async def report(self, *, before: Any = None) -> None:
        """Send what the channel changed since the last item: its record, a
        call's fixed arguments, a call's end (but the one *before* is)."""
        await self._report_record(before)
        for seq, start in self._open.items():
            if start.ran_with is not None and self._args_sent.get(seq) != start.ran_with:
                self._args_sent[seq] = start.ran_with
                await self._send(ToolArgs(start=seq, arguments=start.ran_with))
            ended = start.ended
            if ended is not None and ended is not before and seq not in self._ends_sent:
                self._ends_sent.add(seq)
                await self._send(ToolEnded(start=seq, end=encode_end(ended)))

    async def _report_record(self, before: Any) -> None:
        """Send the record when it differs from the last one sent. Before a
        delta, only a write to one of its keys since says it may: a copy per
        delta would cost the record's size on each (RFC §10.6.4)."""
        live = self._live
        if live is None:
            return
        writes = _writes_of(live)
        if is_delta(before) and writes is not None and writes == self._writes:
            return
        self._writes = writes
        snapshot = record_payload(live)
        if snapshot != self._record:
            self._record = snapshot
            await self._send(Record(record=snapshot))

    async def send_item(self, item: Any) -> None:
        """Send one item, a call's end naming the frame of its start."""
        start = self._start_of(item) if isinstance(item, ToolCallEndMarker) else None
        try:
            data = encode_item(item, start=start)
        except TypeError as exc:
            await self._refuse("stream item", exc)
        seq = await self._send(Item(item=data))
        if isinstance(item, ToolCallStartMarker):
            self._open[seq] = item

    def _start_of(self, end: ToolCallEndMarker) -> int | None:
        """The start *end* closes: the one it rides, else the earliest open
        under its id (two calls under one id are two calls, RFC §6.4)."""
        seq = next((s for s, start in self._open.items() if start.ended is end), None)
        if seq is None:
            seq = next(
                (s for s, start in self._open.items() if start.tool_id == end.tool_id), None
            )
        if seq is not None:
            del self._open[seq]
        return seq
