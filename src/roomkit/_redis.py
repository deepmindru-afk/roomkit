"""The bounds of the Redis clients the delivery queue, the status bus and the
realtime backend open, and how their pub/sub readers tell a dead connection
from a quiet one.

A connection a NAT or a load balancer forgot (half-open) answers nothing and
raises nothing. redis-py before 8.0 sets no bound of its own, so a call on it
waits until TCP gives up, about 15 minutes; a pub/sub read never notices on
any version, since a read past its own timeout just finds no message, and
redis-py's ``health_check_interval`` writes its ping without waiting for the
answer. The turn backend bounds its connections from its own silence bound
(``turns/_redis_clients.py``, RFC §10.6.8).
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("roomkit.redis")

CALL_BOUND_S = 5.0
"""How long a call waits for its peer before it gives the connection up:
redis-py 8's own default, applied on every version."""

PING_AFTER_S = 5.0
"""How long a pub/sub connection may stay silent before its reader pings it."""

PONG_WAIT_S = 5.0
"""How long a reader waits for any word after its ping before it drops the
connection."""


def client_options(read_block: float = 0.0) -> dict[str, float]:
    """The options that bound every call of a client at :data:`CALL_BOUND_S`,
    *read_block* more for one whose blocking reads the server holds that long."""
    return {"socket_timeout": CALL_BOUND_S + read_block, "socket_connect_timeout": CALL_BOUND_S}


class PubSubWatch:
    """Reads a pub/sub connection and keeps it alive.

    Silent for :data:`PING_AFTER_S`, the connection is pinged; with no word
    :data:`PONG_WAIT_S` later, it is dropped, and the next read opens it
    again, subscribed again to its channels (redis-py does that). What was
    published meanwhile is lost, as for any connection lost.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._heard = time.monotonic()
        self._pinged: float | None = None

    async def read(self, pubsub: Any, timeout: float) -> dict[str, Any] | None:
        """The next message of *pubsub* within *timeout*; ``None`` when none
        came, the answer to a ping included."""
        await self._probe(pubsub)
        message = await pubsub.get_message(ignore_subscribe_messages=False, timeout=timeout)
        if message is None:
            return None
        self._heard, self._pinged = time.monotonic(), None
        return None if message.get("type") == "pong" else message

    async def _probe(self, pubsub: Any) -> None:
        """Ping *pubsub* once it has been silent too long, or drop its
        connection when the ping went unanswered."""
        now = time.monotonic()
        if not pubsub.subscribed or pubsub.connection is None:
            # Nothing to hear from: the silence starts once subscribed.
            self._heard, self._pinged = now, None
            return
        if self._pinged is None:
            if now - self._heard >= PING_AFTER_S:
                await pubsub.ping()
                self._pinged = now
            return
        if now - self._pinged >= PONG_WAIT_S:
            logger.warning(
                "Redis %s connection silent for %.0fs; reconnecting", self._name, now - self._heard
            )
            await pubsub.connection.disconnect()
            self._heard, self._pinged = now, None
