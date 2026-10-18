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

_READ_BLOCK_MS = 1000
_READ_COUNT = 256
_RETRY_DELAY = 1.0


@dataclass(frozen=True, slots=True)
class Envelope:
    """One inbox entry: a protocol message and its transport fields."""

    kind: str
    """``request``, ``frame``, ``control``, ``released``, ``channel``,
    ``channel_result``, ``owner`` or ``host_result``."""
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
    """

    def __init__(
        self, client: Any, key: str, handle: Callable[[Envelope], None], *, owns_client: bool
    ) -> None:
        self._client = client
        self._key = key
        self._handle = handle
        self._owns_client = owns_client
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
        """Read the inbox from its start, each entry handed on and deleted;
        a failed read is retried, the entries waiting where they are."""
        key = self._key
        last = "0-0"
        while True:
            try:
                answer = await self._client.xread(
                    {key: last}, count=_READ_COUNT, block=_READ_BLOCK_MS
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Reading the turn inbox %s failed; retrying", key, exc_info=True)
                await asyncio.sleep(_RETRY_DELAY)
                continue
            for _stream, entries in answer or ():
                ids = []
                for entry_id, fields in entries:
                    last = _text(entry_id)
                    ids.append(entry_id)
                    self._hand_on(fields)
                if ids:
                    await self._delete(key, ids)

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
