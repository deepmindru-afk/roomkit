"""A process's inbox on Redis for the turn backend (RFC §10.6).

Each process that runs a :class:`~roomkit.turns.redis.RedisTurnBackend` has
one stream, its inbox, which every other process writes to (through its
outbox, :mod:`~roomkit.turns._redis_outbox`) and it alone reads, with one
blocking read for all its turns, on a connection of its own. An entry carries one message
of the turn protocol as its JSON (a request, a frame, a control message, a
channel control, a message outside a turn, an answer), with the transport's
fields only: its kind, the node that sent it (the answer's address), the
executor it is for, and a reference tying an answer to its question. What
was read is deleted; the inbox of a process that stopped expires.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("roomkit.turns")

READ_BLOCK_S = 1.0
"""How long the server holds the inbox's blocking read at most, in seconds:
its connection waits that much longer than any other call (RFC §10.6.8)."""
_READ_COUNT = 256
_RETRY_DELAY = 1.0


@dataclass(frozen=True, slots=True)
class Envelope:
    """One inbox entry: a protocol message and its transport fields."""

    kind: str
    """``request``, ``frame``, ``control``, ``released``, ``channel``,
    ``owner`` (a room call), ``work`` (work after a turn nothing answers)
    or ``answer``."""
    sender: str
    """The node that sent it, where an answer goes."""
    body: str
    """The protocol message, as JSON."""
    target: str = ""
    """The executor it is for, at the receiving node."""
    ref: str = ""
    """What ties an answer to its question."""

    def fields(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "sender": self.sender,
            "body": self.body,
            "target": self.target,
            "ref": self.ref,
        }

    @classmethod
    def parse(cls, fields: dict[Any, Any]) -> Envelope:
        """The envelope of a stream entry's fields, bytes or text."""
        text = {_text(key): _text(value) for key, value in fields.items()}
        return cls(
            kind=text["kind"],
            sender=text["sender"],
            body=text["body"],
            target=text.get("target", ""),
            ref=text.get("ref", ""),
        )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else bytes(value).decode()


class Inbox:
    """The reader of this process's inbox.

    Args:
        client: The ``redis.asyncio`` client its blocking read holds a
            connection of.
        key: This process's inbox stream.
        handle: Called with each entry read, in order, on the reader's task;
            it must not wait.
        owns_client: Whether the reader closes *client* when it closes.
        block: How long the server holds each read, in seconds.
        timeout_errors: What a read its bound cut raises: read again at once.
    """

    def __init__(
        self,
        client: Any,
        key: str,
        handle: Callable[[Envelope], None],
        *,
        owns_client: bool,
        block: float = READ_BLOCK_S,
        timeout_errors: tuple[type[Exception], ...] = (TimeoutError,),
    ) -> None:
        self._client = client
        self._key = key
        self._handle = handle
        self._owns_client = owns_client
        # Redis reads a block of 0 as forever.
        self._block_ms = max(1, round(block * 1000))
        self._timeout_errors = timeout_errors
        self._reader: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start reading this process's inbox, once."""
        if self._reader is None:
            self._reader = asyncio.get_running_loop().create_task(
                self._read(), name=f"roomkit-turns-inbox-{self._key}"
            )

    async def close(self) -> None:
        """Stop reading; what is left in the inbox expires with it."""
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None
        if self._owns_client:
            await self._client.aclose()

    async def _read(self) -> None:
        """Read the inbox from its start, each entry handed on and deleted."""
        key = self._key
        last = "0-0"
        while True:
            answer = await self._entries_after(last)
            for _stream, entries in answer or ():
                ids = []
                for entry_id, fields in entries:
                    last = _text(entry_id)
                    ids.append(entry_id)
                    self._hand_on(fields)
                if ids:
                    await self._delete(key, ids)

    async def _entries_after(self, last: str) -> Any:
        """The entries after *last*, once read. A read its bound cut is read
        again at once, on a new connection, so it lands inside the silence
        bound (RFC §10.6.8); another failure, or a second cut in a row, is
        retried after a delay. The entries wait where they are meanwhile."""
        again = True
        while True:
            try:
                return await self._client.xread(
                    {self._key: last}, count=_READ_COUNT, block=self._block_ms
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Reading the turn inbox %s failed; retrying", self._key, exc_info=True
                )
                if again and isinstance(exc, self._timeout_errors):
                    again = False
                    continue
                again = True
                await asyncio.sleep(_RETRY_DELAY)

    def _hand_on(self, fields: dict[Any, Any]) -> None:
        try:
            envelope = Envelope.parse(fields)
        except (KeyError, UnicodeDecodeError):
            logger.warning("A malformed entry in the turn inbox was dropped")
            return
        try:
            self._handle(envelope)
        except Exception:
            logger.exception("A %s message of the turn inbox could not be handled", envelope.kind)

    async def _delete(self, key: str, ids: list[Any]) -> None:
        try:
            await self._client.xdel(key, *ids)
        except Exception:
            logger.debug("Deleting read entries of %s failed", key, exc_info=True)
