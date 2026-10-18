"""What a turn's channel changes on what it already yielded (RFC §10.6.4).

The channel keeps writing into objects the owner holds copies of: its
response-metadata record, a call's arguments once its gate fixed them, a
call's end before the end itself is read. Before each item the executor
sends, it sends what changed, so the owner's copies read as the channel's own.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, NoReturn

from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.turns._codec import encode_end, encode_item, record_payload
from roomkit.turns.wire import Item, Record, ToolArgs, ToolEnded

if TYPE_CHECKING:
    from roomkit.turns.wire import Frame


class ChannelChanges:
    """The changes of one turn's channel, sent before the items they precede."""

    def __init__(
        self,
        send: Callable[[Frame], Awaitable[int]],
        refuse: Callable[[str, Exception], Awaitable[NoReturn]],
    ) -> None:
        self._send = send
        self._refuse = refuse
        self._live: Any = None
        self._record: dict[str, Any] = {}
        # The calls the stream started and has not ended, by their start's frame.
        self._open: dict[int, ToolCallStartMarker] = {}
        self._args_sent: dict[int, Any] = {}
        self._ends_sent: set[int] = set()

    def watch(self, record: Any) -> None:
        """Watch the channel's live record, its output's."""
        self._live = record

    def sent(self, record: dict[str, Any]) -> None:
        """The record as the output carried it: a change is one from it."""
        self._record = record

    async def report(self, *, before: Any = None) -> None:
        """Send what the channel changed since the last item: its record, a
        call's fixed arguments, a call's end (but the one *before* is)."""
        if self._live is not None:
            snapshot = record_payload(self._live)
            if snapshot != self._record:
                self._record = snapshot
                await self._send(Record(record=snapshot))
        for seq, start in self._open.items():
            if start.ran_with is not None and self._args_sent.get(seq) != start.ran_with:
                self._args_sent[seq] = start.ran_with
                await self._send(ToolArgs(start=seq, arguments=start.ran_with))
            ended = start.ended
            if ended is not None and ended is not before and seq not in self._ends_sent:
                self._ends_sent.add(seq)
                await self._send(ToolEnded(start=seq, end=encode_end(ended)))

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
