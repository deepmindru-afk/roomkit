"""What a process writes to the other processes' inboxes on the Redis turn backend.

Each destination has one writer task: what is posted to a node reaches its
inbox in the order it was posted, a batch per round trip, whatever task posted
it and whether that task is cancelled meanwhile. Posting never waits, so a
message is numbered and queued in one step, and two messages of one turn can
never reach the other side in the other order (RFC §10.6.3). A batch Redis
refuses is not retried: each message's sender is told, and its turn fails
rather than wait for what will not come. A writer left idle goes, and the
next message to its node starts another; closing writes what is queued
(a turn's ``end`` among it) within a bound, and tells the senders of the rest.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.turns._redis_inbox import Envelope

logger = logging.getLogger("roomkit.turns")

_BATCH = 256

Failed = Callable[[Exception], None]
"""Told when Redis refused the batch a message was in."""

_Queued = tuple["Envelope", Failed | None]
# Put on each queue on closing: its writer stops once it wrote what came before.
_CLOSING = None


class Outbox:
    """The writes of this process, in order per destination node.

    Args:
        client: The ``redis.asyncio`` client.
        inbox_key: The inbox stream of a node.
        ttl_ms: How long an inbox nobody writes to lives, and a writer idle.
    """

    def __init__(self, client: Any, inbox_key: Callable[[str], str], ttl_ms: int) -> None:
        self._client = client
        self._inbox_key = inbox_key
        self._ttl_ms = ttl_ms
        self._queues: dict[str, asyncio.Queue[_Queued | None]] = {}
        self._writers: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def post(self, node: str, envelope: Envelope, failed: Failed | None = None) -> None:
        """Queue *envelope* for *node*'s inbox, after what was posted to it
        before; *failed* is told if Redis refuses it."""
        if self._closed:
            if failed is not None:
                failed(ConnectionError("The turn backend closed"))
            return
        queue = self._queues.get(node)
        if queue is None:
            queue = self._queues[node] = asyncio.Queue()
            # A context of its own: the writes belong to no turn's trace.
            self._writers[node] = asyncio.get_running_loop().create_task(
                self._write(node, queue),
                name=f"roomkit-turns-outbox-{node}",
                context=contextvars.Context(),
            )
        queue.put_nowait((envelope, failed))

    async def close(self, drain: float) -> None:
        """Take no more messages, write what is queued within *drain*
        seconds, and tell the senders of what is left."""
        self._closed = True
        for queue in self._queues.values():
            queue.put_nowait(_CLOSING)
        writers = list(self._writers.values())
        if writers:
            _, late = await asyncio.wait(writers, timeout=drain)
            for writer in late:
                writer.cancel()
            await asyncio.gather(*writers, return_exceptions=True)
        left = ConnectionError("The turn backend closed before writing it")
        for queue in self._queues.values():
            _tell([item for item in _emptied(queue) if item is not _CLOSING], left)
        self._writers.clear()

    async def _write(self, node: str, queue: asyncio.Queue[_Queued | None]) -> None:
        key = self._inbox_key(node)
        while True:
            try:
                async with asyncio.timeout(self._ttl_ms / 1000):
                    first = await queue.get()
            except TimeoutError:
                if queue.empty():
                    # Idle: the writer goes; the next message starts another.
                    del self._queues[node], self._writers[node]
                    return
                continue
            batch, closing = _batch(first, queue)
            if batch:
                await self._send(key, node, batch)
            if closing:
                return

    async def _send(self, key: str, node: str, batch: list[_Queued]) -> None:
        try:
            pipe = self._client.pipeline(transaction=False)
            for envelope, _ in batch:
                pipe.xadd(key, envelope.fields())
            pipe.pexpire(key, self._ttl_ms)
            await pipe.execute()
        except asyncio.CancelledError:
            _tell(batch, ConnectionError("The turn backend closed before writing it"))
            raise
        except Exception as exc:
            logger.warning("Writing %d turn message(s) to %s failed: %s", len(batch), node, exc)
            _tell(batch, exc)


def _batch(
    first: _Queued | None, queue: asyncio.Queue[_Queued | None]
) -> tuple[list[_Queued], bool]:
    """*first* and what follows it in *queue*, up to a batch, and whether
    the closing mark ended it."""
    batch: list[_Queued] = []
    item = first
    while item is not _CLOSING:
        batch.append(item)
        if len(batch) == _BATCH or queue.empty():
            return batch, False
        item = queue.get_nowait()
    return batch, True


def _emptied(queue: asyncio.Queue[_Queued | None]) -> list[_Queued | None]:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _tell(batch: list[_Queued], error: Exception) -> None:
    for envelope, failed in batch:
        if failed is None:
            continue
        try:
            failed(error)
        except Exception:
            logger.exception("Telling the sender of a %s message it failed failed", envelope.kind)
