"""Fan-out of one async stream to several independent readers.

A streamed AI response is read once — reading it drives persistence
upstream — but every voice session in the room must hear all of it.
:class:`StreamFanOut` reads the source in a single producer coroutine and
copies each item into one :class:`StreamBranch` per reader.  A reader that
stops early (barge-in, transport error) closes its branch without cutting
the others off.
"""

from __future__ import annotations

from asyncio import Queue
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

    def __init__(self) -> None:
        self._queue: Queue[T | _Failure | _End] = Queue()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Stop receiving items; the other branches are unaffected."""
        self._closed = True

    def _put(self, item: T | _Failure | _End) -> None:
        if not self._closed:
            self._queue.put_nowait(item)

    def __aiter__(self) -> StreamBranch[T]:
        return self

    async def __anext__(self) -> T:
        if self._closed:
            raise StopAsyncIteration
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
        self.branches: list[StreamBranch[T]] = [StreamBranch() for _ in range(readers)]

    async def run(self) -> None:
        """Pump the source into every open branch until it ends.

        Stops pulling as soon as every branch is closed, so a source nobody
        reads any more is left where it is, as a single reader would leave
        it.  A source error is handed to every open branch rather than
        raised here: each reader surfaces it on its own path.
        """
        try:
            while not all(branch.closed for branch in self.branches):
                try:
                    item = await anext(self._source)
                except StopAsyncIteration:
                    return
                for branch in self.branches:
                    branch._put(item)
        except Exception as exc:
            for branch in self.branches:
                branch._put(_Failure(exc))
        finally:
            for branch in self.branches:
                branch._put(_END)
