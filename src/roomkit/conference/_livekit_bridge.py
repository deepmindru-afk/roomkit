"""The ordered, bounded fanout of one bot session's control-plane events.

LiveKit calls its handlers synchronously and discards whatever they return, while
the framework's fanout is awaited — so handlers only enqueue, and a single
consumer task awaits the emissions in the order they were enqueued. Scheduling a
task per event instead would let a track's arrival overtake its publisher's, and
a roster asked to open a lane for a participant it has never seen has no good
answer.

The bridge is bounded, and says when it is full rather than dropping anything:
what a full bridge means is the session's to decide. It knows nothing of rooms,
sessions or departures. The pumps are deliberately *not* on it: see
``_livekit_media``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.core.task_utils import cancel_and_wait

logger = logging.getLogger("roomkit.conference.livekit")

# How many control-plane events the bridge holds before declaring the session
# unhealthy. State events (active speaker, connection quality) coalesce to one
# entry per key and never accumulate; lifecycle events carry facts the roster
# and the lanes must not miss, so past this bound the session *ends* — through
# the same ``bot_session_ended`` contract as a dropped connection — rather
# than lose an arrival or a track silently. The channel's supervisor re-joins,
# and the new session's catch-up rebuilds a consistent view of the *current*
# state; what happened entirely inside the outage window is discarded with
# the session, counted, and named in the report (RFC 12.10.3).
MAX_QUEUED_EVENTS = 512

Emit = Callable[..., Awaitable[None]]


class EventBridge:
    """Queue a session's events in order and emit them from one consumer task."""

    def __init__(self, room_id: str) -> None:
        self._room_id = room_id
        # Bounded: the consumer awaits the framework's fanout — identity
        # resolution, hooks — so a participant generating events faster than
        # the fanout returns would otherwise grow this without limit. State
        # events never queue more than one entry each (see `put_state`).
        self._events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._pending_state: dict[Any, tuple[Emit, tuple[Any, ...]]] = {}
        self._consumer: asyncio.Task[None] | None = None

    @property
    def queued(self) -> int:
        """How many entries wait for the consumer."""
        return self._events.qsize()

    def start(self) -> None:
        """Start the consumer task."""
        self._consumer = asyncio.create_task(self._consume())

    def put(self, emit: Emit, *args: Any) -> bool:
        """Queue a lifecycle event — an arrival, a departure, a track — or refuse it.

        These carry facts the roster and the lanes must not miss, so they are
        never coalesced and never silently dropped: a bridge that can no longer
        hold one answers ``False``, and the caller ends the session instead.
        """
        if self._events.qsize() >= MAX_QUEUED_EVENTS:
            return False
        self._events.put_nowait(("event", (emit, args)))
        return True

    def put_state(self, key: Any, emit: Emit, *args: Any) -> bool:
        """Queue a state event, keeping only the latest value per key, or refuse it.

        Active speaker and connection quality are *states*, not facts: only
        the current value matters, and a consumer that fell behind should say
        the newest one, not replay the history. One marker per key sits in
        the queue; further updates replace the stored value in place, so a
        participant flapping quality cannot grow the queue at all.
        """
        already_queued = key in self._pending_state
        self._pending_state[key] = (emit, args)
        if already_queued:
            return True
        if self._events.qsize() >= MAX_QUEUED_EVENTS:
            return False
        self._events.put_nowait(("state", key))
        return True

    async def stop(self) -> int:
        """Stop the consumer, and drop what it will never deliver; say how much.

        A departing session's remaining events describe a conference the
        framework has stopped listening to, and emitting them after the bot has
        gone would announce arrivals into a room that is being torn down. The
        count is returned because on the unhealthy path it is part of the
        report: these were facts, and they are being discarded.
        """
        await cancel_and_wait(self._consumer)
        self._consumer = None
        undelivered = 0
        while not self._events.empty():
            self._events.get_nowait()
            undelivered += 1
        self._pending_state.clear()
        return undelivered

    async def _consume(self) -> None:
        """Await the queued emissions, in order, until cancelled.

        A subscriber that raises is the backend's own fanout problem and is
        logged there; anything that escapes it would otherwise kill this task
        and take the room's whole event stream with it, so it is caught here too.
        """
        while True:
            kind, payload = await self._events.get()
            if kind == "state":
                entry = self._pending_state.pop(payload, None)
                if entry is None:
                    continue
                emit, args = entry
            else:
                emit, args = payload
            try:
                await emit(*args)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Emitting a LiveKit conference event for room %s failed", self._room_id
                )
