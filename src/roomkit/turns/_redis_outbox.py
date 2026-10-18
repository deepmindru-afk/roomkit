"""What a process writes to the other processes' inboxes on the Redis turn backend.

Each destination has one writer task: what is posted to a node reaches its
inbox in the order it was posted, a batch per round trip, whatever task posted
it and whether that task is cancelled meanwhile. Posting never waits, so a
message is numbered and queued in one step, and two messages of one turn can
never reach the other side in the other order (RFC §10.6.3). A batch Redis
refuses is not retried: each message's sender is told, and its turn fails
rather than wait for what will not come.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.turns._redis_inbox import Envelope

logger = logging.getLogger("roomkit.turns")

_BATCH = 256

Failed = Callable[[Exception], None]
"""Told when Redis refused the batch a message was in."""


class Outbox:
    """The writes of this process, in order per destination node.

    Args:
        client: The ``redis.asyncio`` client.
        inbox_key: The inbox stream of a node.
        ttl_ms: How long an inbox nobody writes to lives.
    """

    def __init__(self, client: Any, inbox_key: Callable[[str], str], ttl_ms: int) -> None:
        self._client = client
        self._inbox_key = inbox_key
        self._ttl_ms = ttl_ms
        self._queues: dict[str, asyncio.Queue[tuple[Envelope, Failed | None]]] = {}
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
            self._writers[node] = asyncio.get_running_loop().create_task(
                self._write(node, queue), name=f"roomkit-turns-outbox-{node}"
            )
        queue.put_nowait((envelope, failed))

    async def close(self) -> None:
        """Stop writing; what is still queued is not sent."""
        self._closed = True
        for writer in self._writers.values():
            writer.cancel()
        await asyncio.gather(*self._writers.values(), return_exceptions=True)
        self._writers.clear()

    async def _write(
        self, node: str, queue: asyncio.Queue[tuple[Envelope, Failed | None]]
    ) -> None:
        key = self._inbox_key(node)
        while True:
            batch = [await queue.get()]
            while len(batch) < _BATCH and not queue.empty():
                batch.append(queue.get_nowait())
            try:
                pipe = self._client.pipeline(transaction=False)
                for envelope, _ in batch:
                    pipe.xadd(key, envelope.fields())
                pipe.pexpire(key, self._ttl_ms)
                await pipe.execute()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Writing %d turn message(s) to %s failed: %s", len(batch), node, exc
                )
                _tell(batch, exc)


def _tell(batch: list[tuple[Envelope, Failed | None]], error: Exception) -> None:
    for envelope, failed in batch:
        if failed is None:
            continue
        try:
            failed(error)
        except Exception:
            logger.exception("Telling the sender of a %s message it failed failed", envelope.kind)
