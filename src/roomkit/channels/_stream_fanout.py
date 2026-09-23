"""Fan-out of one async stream to several independent readers.

A streamed AI response is read once — reading it drives persistence
upstream — but every voice session in the room must hear all of it.
:class:`StreamFanOut` reads the source in a single producer coroutine and
copies each item into one :class:`StreamBranch` per reader.  A reader that
stops early (barge-in, transport error) closes its branch without cutting
the others off.

The source is pulled on demand, one item each time a reader has run dry,
so it advances at the pace of the fastest reader: with a single reader it
is read exactly as that reader alone would read it.
"""

from __future__ import annotations

from asyncio import Event, Queue
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class _Failure:
    """The source raised: every branch re-raises the same error."""

    error: Exception


class _End:
    """The source is exhausted (or the producer stopped)."""


_END = _End()


class StreamBranch[T]:
    """One reader's copy of a :class:`StreamFanOut` source."""

    def __init__(self, demand: Event) -> None:
        self._queue: Queue[T | _Failure | _End] = Queue()
        self._closed = False
        self._demand = demand

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Stop receiving items; the other branches are unaffected."""
        self._closed = True
        # Wake the producer so it notices when every branch is closed.
        self._demand.set()

    def _put(self, item: T | _Failure | _End) -> None:
        if not self._closed:
            self._queue.put_nowait(item)

    def __aiter__(self) -> StreamBranch[T]:
        return self

    async def __anext__(self) -> T:
        if self._closed:
            raise StopAsyncIteration
        if self._queue.empty():
            self._demand.set()
        item = await self._queue.get()
        if isinstance(item, _End):
            self._closed = True
            raise StopAsyncIteration
        if isinstance(item, _Failure):
            self._closed = True
            raise item.error
        return item


class StreamFanOut[T]:
    """Copy one async stream to *readers* branches, reading it only once."""

    def __init__(self, source: AsyncIterator[T], readers: int) -> None:
        self._source = source
        self._demand = Event()
        self.branches: list[StreamBranch[T]] = [StreamBranch(self._demand) for _ in range(readers)]
        self.error: Exception | None = None

    def _all_closed(self) -> bool:
        return all(branch.closed for branch in self.branches)

    async def run(self) -> None:
        """Pump the source into every open branch until it ends.

        Pulls the next item only when a reader asks for one, and stops once
        every branch is closed: a source nobody reads any more is left
        where it is, as a single reader would leave it.  A source error is
        kept in :attr:`error` and handed to every open branch rather than
        raised here: each reader surfaces it on its own path.
        """
        try:
            while not self._all_closed():
                await self._demand.wait()
                self._demand.clear()
                if self._all_closed():
                    return
                try:
                    item = await anext(self._source)
                except StopAsyncIteration:
                    return
                for branch in self.branches:
                    branch._put(item)
        except Exception as exc:
            self.error = exc
            for branch in self.branches:
                branch._put(_Failure(exc))
        finally:
            for branch in self.branches:
                branch._put(_END)
