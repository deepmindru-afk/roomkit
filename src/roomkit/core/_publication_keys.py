"""The keys of proactive publications, apart from every other key (RFC §22.1).

A keyed ``deliver()`` names one publication in its room, whatever transport
carries it, and is looked up within the room, whatever channel committed it.
An inbound message's key is its sender's: on a channel whose client chooses
it (a WebSocket, a webhook), a sender could key a message like a coming
publication, which a lookup within the room would then take for that
publication and never publish. So a publication is stored under its key in
a namespace of the framework's own (``roomkit:publication:<key>``), which
only the delivery path writes: the pipeline drops a key in that namespace
from any other message, and ``send_event()`` refuses one. The key is part of
the committed event, so the lookup under the room lock and the commit that
writes it stay one section, and the store's per-channel uniqueness never
sets a publication against a message.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.models.event import RoomEvent

PUBLICATION_KEY_PREFIX = "roomkit:publication:"
"""The namespace of the keys the framework stores its publications under."""

_PUBLISHING: ContextVar[str | None] = ContextVar("roomkit_publishing", default=None)


def publication_key(key: str) -> str:
    """The key a publication keyed *key* by the application is stored under."""
    return f"{PUBLICATION_KEY_PREFIX}{key}"


def application_key(stored: str | None) -> str | None:
    """The application's key of a publication stored under *stored*; any
    other key as it is."""
    if stored is not None and stored.startswith(PUBLICATION_KEY_PREFIX):
        return stored[len(PUBLICATION_KEY_PREFIX) :]
    return stored


def is_reserved(key: str | None) -> bool:
    """Whether *key* is in the framework's publication namespace."""
    return key is not None and key.startswith(PUBLICATION_KEY_PREFIX)


@contextmanager
def publishing(stored: str | None) -> Generator[None]:
    """Let the message the delivery path processes inside carry *stored*,
    its publication key, which the pipeline drops from any other."""
    token = _PUBLISHING.set(stored)
    try:
        yield
    finally:
        _PUBLISHING.reset(token)


def refuse_reserved(key: str | None) -> None:
    """Raise ``ValueError`` for *key* in the publication namespace: a door
    that stores a caller's event as given (``send_event()``,
    ``commit_event()``) never writes one."""
    if is_reserved(key):
        raise ValueError(
            f"The idempotency_key namespace {PUBLICATION_KEY_PREFIX} is the framework's: "
            "it keys deliver()'s publications (RFC §22.1)"
        )


def without_reserved_key(event: RoomEvent) -> RoomEvent:
    """*event* without a key in the publication namespace: a row the
    framework writes as a channel made it (a delegated turn's trace) never
    carries one."""
    if not is_reserved(event.idempotency_key):
        return event
    return event.model_copy(update={"idempotency_key": None})


def admitted_key(key: str | None) -> str | None:
    """*key* as the pipeline admits it on a message: a key in the
    publication namespace only from the delivery path publishing it, no key
    otherwise."""
    if is_reserved(key) and key != _PUBLISHING.get():
        return None
    return key
