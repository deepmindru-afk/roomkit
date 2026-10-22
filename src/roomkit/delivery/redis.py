"""Redis Streams delivery backend.

Uses Redis Streams with consumer groups for persistent, distributed
delivery across multiple worker processes.

Requires Redis server 6.2+ and ``redis>=5.0.2``::

    pip install roomkit[redis]

Usage::

    from roomkit.delivery import RedisDeliveryBackend

    kit = RoomKit(
        delivery_backend=RedisDeliveryBackend("redis://localhost:6379"),
    )
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit._redis import client_options
from roomkit.core.task_utils import cancel_and_wait
from roomkit.delivery._redis_scripts import PROMOTE, RENEW, TRANSITION
from roomkit.delivery.base import (
    DeliveryBackend,
    DeliveryItem,
    DeliveryItemStatus,
    check_retry_delays,
    retry_delay,
)
from roomkit.delivery.worker import run_worker_loop

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.delivery.redis")

# Shortest wait for a retry about to fall due: a pass costs a round trip.
_MIN_BLOCK_MS = 10

_READ_SLICE_S = 1.0
"""The longest a read waits for new work: ``dequeue()`` waits in slices of
it, so no read outlasts its client's call bound, on any redis-py version."""


def _read_block_ms(remaining: float, next_due_ms: int | None) -> int | None:
    """How long one read for new work waits, in ms: until the wait's
    *remaining* seconds end, a retry falls due (*next_due_ms*), or a read
    slice ends, whichever comes first; ``None`` reads without waiting."""
    if remaining <= 0:
        return None
    block = min(remaining, _READ_SLICE_S) * 1000
    # XREADGROUP does not wake for the delayed set: a retry falling due
    # first cuts the read short, and the next pass queues it.
    if next_due_ms is not None:
        block = min(block, max(_MIN_BLOCK_MS, next_due_ms))
    return max(1, int(block))


class RedisDeliveryBackend(DeliveryBackend):
    """Redis Streams delivery backend with consumer groups.

    Items are serialized as JSON and stored in a Redis Stream.
    Consumer groups distribute items across workers automatically.

    Every call gives up after 6 s (5 s, and the second a read for new work
    waits) on a connection a NAT or a load balancer forgot (half-open), and
    is tried again on a new one: the worker loop's next pass, the next
    renewal. A live worker keeps its leases across such a connection while
    *claim_idle_seconds* stays above three times that bound (18 s; 60 s by
    default).

    Args:
        url: Redis connection URL (ignored if *client* is provided). Its
            query's options (``?socket_timeout=...``) take precedence over
            the backend's bounds; one under a second fails idle reads.
        client: Inject an existing ``redis.asyncio.Redis`` instance; it
            keeps its own options (redis-py before 8.0 bounds no call): give
            it a ``socket_timeout`` over a second.
        stream_prefix: Namespace for stream keys.
        group_name: Consumer group name — all workers in the same
            group share the workload.
        max_dead_letter_size: Approximate cap on the dead-letter stream
            (uses ``MAXLEN ~``).
        claim_idle_seconds: Reclaim abandoned work after this idle interval.
            Live workers renew their pending entries every third of this interval.
            Delivery is at-least-once; destinations should handle duplicate IDs.
        retry_delay: Seconds before the first retry; each next one waits
            twice as long, up to *max_retry_delay*. A retry waits in a sorted
            set (``{stream_prefix}:delayed``) on the Redis server's clock, and
            a worker queues it again once due; one waiting in ``dequeue()``
            stops waiting when it falls due. 0 retries at once.
        max_retry_delay: Cap on the wait before a retry.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        client: Any = None,
        stream_prefix: str = "roomkit:delivery",
        group_name: str = "roomkit-workers",
        max_dead_letter_size: int = 10_000,
        claim_idle_seconds: float = 60.0,
        retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
    ) -> None:
        if not math.isfinite(claim_idle_seconds) or claim_idle_seconds <= 0:
            raise ValueError("claim_idle_seconds must be finite and positive")
        check_retry_delays(retry_delay, max_retry_delay)
        self._retry_delay = retry_delay
        self._max_retry_delay = max_retry_delay
        try:
            import redis.asyncio as _aioredis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisDeliveryBackend. "
                "Install it with: pip install roomkit[redis]"
            ) from exc

        self._client: Any
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = _aioredis.from_url(url, **client_options(_READ_SLICE_S))
            self._owns_client = True

        self._pending_key = f"{stream_prefix}:pending"
        self._dl_key = f"{stream_prefix}:dead_letter"
        self._delayed_key = f"{stream_prefix}:delayed"
        self._group = group_name
        self._max_dl = max_dead_letter_size
        self._worker_id = uuid4().hex[:12]
        self._worker_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._claim_idle_ms = max(1, math.ceil(claim_idle_seconds * 1000))
        self._claim_cursor: str | bytes = "0-0"

        # Maps DeliveryItem.id → Redis stream entry ID
        self._entry_ids: dict[str, str] = {}
        # Maps DeliveryItem.id → DeliveryItem (for nack/dead_letter)
        self._items: dict[str, DeliveryItem] = {}

    # -- ABC implementation -----------------------------------------------

    async def enqueue(self, item: DeliveryItem) -> None:
        item.status = DeliveryItemStatus.PENDING
        await self._client.xadd(self._pending_key, {"data": item.model_dump_json()})
        logger.debug("Enqueued %s for room %s", item.id, item.room_id)

    async def dequeue(
        self,
        worker_id: str,
        batch_size: int = 1,
        timeout: float = 5.0,
    ) -> list[DeliveryItem]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        remaining = timeout
        while True:
            next_due_ms = await self._promote_due(batch_size)
            block = _read_block_ms(remaining, next_due_ms)
            resp = await self._claim_or_read(worker_id, batch_size, block)
            remaining = deadline - loop.time()
            if resp or remaining <= 0:
                break
        if not resp:
            return []

        items = await self._take_entries(worker_id, resp)
        if items and (self._heartbeat_task is None or self._heartbeat_task.done()):
            self._heartbeat_task = asyncio.create_task(
                self._renew_pending(), name="redis-delivery-heartbeat"
            )
        return items

    async def _promote_due(self, batch_size: int) -> int | None:
        """Queue again the retries whose delay has passed.

        Returns the milliseconds until the next waiting retry is due, 0 when
        one is due already, None when none waits.
        """
        result = await self._client.eval(
            PROMOTE, 2, self._delayed_key, self._pending_key, batch_size
        )
        return max(0, int(result[1])) if len(result) > 1 else None

    async def _claim_or_read(self, worker_id: str, batch_size: int, block: int | None) -> Any:
        """Reclaim abandoned entries, else read new ones, waiting up to *block* ms."""
        # Scan the PEL before reading new work, advancing the cursor even
        # when this scan finds no sufficiently idle entry.
        claimed = await self._client.xautoclaim(
            self._pending_key,
            self._group,
            worker_id,
            self._claim_idle_ms,
            start_id=self._claim_cursor,
            count=batch_size,
        )
        self._claim_cursor = claimed[0]
        if claimed[1]:
            return [(self._pending_key, claimed[1])]
        return await self._client.xreadgroup(
            self._group,
            worker_id,
            {self._pending_key: ">"},
            count=batch_size,
            block=block,
        )

    async def _take_entries(self, worker_id: str, resp: Any) -> list[DeliveryItem]:
        """Track the entries *resp* carries as *worker_id*'s items."""
        items: list[DeliveryItem] = []
        # resp format: [[stream_name, [(entry_id, {field: value}), ...]]]
        for _stream_name, entries in resp:
            for entry_id, fields in entries:
                eid = entry_id if isinstance(entry_id, str) else entry_id.decode()
                raw = fields.get(b"data") or fields.get("data")
                if raw is None:
                    logger.warning("Entry %s has no data field, skipping", eid)
                    await self._client.xack(self._pending_key, self._group, eid)
                    continue

                data = raw if isinstance(raw, str) else raw.decode()
                item = DeliveryItem.model_validate_json(data)
                item.status = DeliveryItemStatus.IN_PROGRESS
                item.worker_id = worker_id
                self._entry_ids[item.id] = eid
                self._items[item.id] = item
                items.append(item)
        return items

    async def _renew_pending(self) -> None:
        while True:
            await asyncio.sleep(self._claim_idle_ms / 3000)
            for item_id, entry_id in list(self._entry_ids.items()):
                item = self._items.get(item_id)
                if item is None:
                    continue
                try:
                    await self._client.eval(
                        RENEW, 1, self._pending_key, self._group, entry_id, item.worker_id
                    )
                except Exception:
                    logger.warning("Failed to renew delivery %s", item_id, exc_info=True)

    async def _transition(
        self,
        item_id: str,
        replacement: DeliveryItem | None = None,
        delay_ms: int | None = None,
    ) -> None:
        """Settle a claimed entry: drop it (ack), or replace it on the
        dead-letter stream, or, after *delay_ms*, on the pending stream."""
        entry_id = self._entry_ids.get(item_id)
        item = self._items.get(item_id)
        if entry_id is None or item is None:
            return
        dead = replacement is not None and replacement.status == DeliveryItemStatus.DEAD_LETTER
        if dead:
            target = self._dl_key
        elif delay_ms is not None:
            target = self._delayed_key
        else:
            target = self._pending_key
        await self._client.eval(
            TRANSITION,
            2,
            self._pending_key,
            target,
            self._group,
            entry_id,
            item.worker_id,
            replacement.model_dump_json() if replacement is not None else "",
            self._max_dl if dead else 0,
            "" if dead or delay_ms is None else delay_ms,
        )
        # A network failure leaves these intact so the transition can be retried.
        # A zero result means it was already completed or another worker owns it.
        if self._entry_ids.get(item_id) == entry_id:
            self._entry_ids.pop(item_id, None)
            self._items.pop(item_id, None)

    async def ack(self, item_id: str) -> None:
        await self._transition(item_id)

    async def nack(self, item_id: str, error: str | None = None) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        retry_count = item.retry_count + 1
        dead = retry_count > item.max_retries
        replacement = item.model_copy(
            update={
                "retry_count": retry_count,
                "error": (error or "max retries exceeded") if dead else error,
                "status": DeliveryItemStatus.DEAD_LETTER if dead else DeliveryItemStatus.PENDING,
                "worker_id": None,
            }
        )
        if dead:
            await self._transition(item_id, replacement)
            return
        delay = retry_delay(retry_count, self._retry_delay, self._max_retry_delay)
        await self._transition(item_id, replacement, delay_ms=round(delay * 1000))

    async def dead_letter(self, item_id: str, error: str) -> None:
        item = self._items.get(item_id)
        if item is not None:
            await self._transition(
                item_id,
                item.model_copy(
                    update={
                        "status": DeliveryItemStatus.DEAD_LETTER,
                        "error": error,
                        "worker_id": None,
                    }
                ),
            )

    async def get_queue_depth(self) -> int:
        """Pending items, those waiting out a retry delay included."""
        queued = await self._client.xlen(self._pending_key)
        waiting = await self._client.zcard(self._delayed_key)
        return int(queued) + int(waiting)

    async def get_dead_letter_items(self, limit: int = 50) -> list[DeliveryItem]:
        entries = await self._client.xrevrange(self._dl_key, "+", "-", count=limit)
        items: list[DeliveryItem] = []
        for _entry_id, fields in entries:
            raw = fields.get(b"data") or fields.get("data")
            if raw is None:
                continue
            data = raw if isinstance(raw, str) else raw.decode()
            items.append(DeliveryItem.model_validate_json(data))
        return items

    # -- Lifecycle --------------------------------------------------------

    async def start(self, kit: RoomKit) -> None:
        """Create consumer group and start the worker loop."""
        if self._worker_task is not None:
            return
        try:
            # id="0" reads from the beginning — required so items enqueued
            # before start() are not silently dropped.
            await self._client.xgroup_create(self._pending_key, self._group, id="0", mkstream=True)
            logger.info("Created consumer group %s", self._group)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            # Group already exists — that's fine
            logger.debug("Consumer group %s already exists", self._group)

        self._worker_task = asyncio.create_task(
            run_worker_loop(
                self,
                kit,
                self._worker_id,
                batch_size=5,
                poll_timeout=2.0,
            ),
            name="redis-delivery-worker",
        )
        logger.info(
            "Redis delivery worker started (id=%s, group=%s)",
            self._worker_id,
            self._group,
        )

    async def close(self) -> None:
        """Stop the worker loop and close the connection if we own it.

        In-flight items remain in the Redis PEL (Pending Entries List)
        and will be reclaimed by another consumer or on restart.
        """
        await self._cancel_worker_task()
        await cancel_and_wait(self._heartbeat_task)
        self._heartbeat_task = None

        # Clear in-process tracking (items remain in Redis PEL for recovery)
        self._entry_ids.clear()
        self._items.clear()

        if self._owns_client:
            await self._client.aclose()

        logger.info("Redis delivery worker stopped")
