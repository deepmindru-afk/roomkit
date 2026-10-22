"""The deliveries a caller named by a key (RFC §10.1, *Explicit delivery
cancellation*).

A caller that sets ``InboundMessage.delivery_key`` can cancel the message's
delivery from anywhere in the process, without holding its handle: an agent's
tool passes the key along as text. A key names every delivery in flight that
carries it, from the call's start until its delivery is over: a delivery
cancelled before its event is committed is committed and delivered all the
same, and none of its turns starts. A cancellation scoped to an organization
reaches only the deliveries of its rooms (RFC §17.2); a delivery whose room is
not known yet (a call that did not name its organization, before its room is
read) takes it once its room proves the organization's, uncounted.

The delivery of a message held for verification (RFC §11.7.3) is over only
once its release is answered (RFC §11.7.6): the key names it past its cascade,
until the release or the policy's last moment to release it, and then names
the cascade of its answer. A held delivery cancelled at any point is answered
by no agent; the cancellation is also recorded on the held event, for a
release another process runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade


class NamedDelivery:
    """A delivery a key names, from its call's start: the organization of
    its room once read, its cascade once the call commits, and past that
    cascade the event it holds for verification, until its release."""

    __slots__ = (
        "_cancelled",
        "_located",
        "_organization_id",
        "_owner",
        "_requests",
        "cascade",
        "held",
        "key",
        "until",
    )

    def __init__(self, key: str | None, organization_id: str | None, owner: DeliveryKeys) -> None:
        self.key = key
        self.cascade: DeliveryCascade | None = None
        # The event the delivery holds for verification, (room_id, event_id),
        # and the last moment (``time.monotonic()``) it may be released.
        self.held: tuple[str, str] | None = None
        self.until = 0.0
        # A call that names its organization lands in one of its rooms or is
        # refused (RFC §17.2), so its delivery is located from the start.
        self._organization_id = organization_id
        self._located = organization_id is not None
        self._owner = owner
        # A cancellation received while no cascade runs the delivery.
        self._cancelled: str | None = None
        # Scoped cancellations waiting for the room's organization.
        self._requests: list[tuple[str, str]] = []

    @property
    def cancelled(self) -> str | None:
        """Why the delivery was cancelled, or ``None``."""
        if self.cascade is not None:
            return self.cascade.cancelled
        return self._cancelled

    def cancel(self, reason: str, organization_id: str | None) -> bool:
        """Cancel the delivery for a caller acting for *organization_id* (any
        caller when ``None``): whether this call cancelled it. A scoped
        request on a delivery whose room is not known yet waits for it, and
        is not counted: until then the delivery is no organization's."""
        if organization_id is not None and not self._located:
            self._requests.append((organization_id, reason))
            return False
        if organization_id is not None and organization_id != self._organization_id:
            return False
        if self.cancelled is not None:
            return False
        if self.cascade is not None:
            self.cascade.cancel(reason)
        else:
            self._cancelled = reason
            if self.held is not None:
                self._owner.held_cancelled(self.held, reason)
        return True

    def locate(self, organization_id: str | None) -> None:
        """The delivery's room is read, of *organization_id*: a scoped
        cancellation that waited for it cancels the delivery when it is that
        organization's."""
        self._organization_id = organization_id
        self._located = True
        if self._cancelled is None:
            self._cancelled = next(
                (asked for owner, asked in self._requests if owner == organization_id), None
            )
        self._requests.clear()

    def attach(self, cascade: DeliveryCascade) -> None:
        """Make *cascade* the delivery named: cancelled at once when the
        delivery already was, named until it is over."""
        reason = self.cancelled
        self.cascade = cascade
        if reason is not None:
            cascade.cancel(reason)
        cascade.when_over(partial(self._over, cascade))

    def release(self) -> None:
        """Stop naming the delivery, unless a cascade still runs it or it
        holds its event: their end does then."""
        if self.cascade is None and self.held is None:
            self._owner.release(self)

    def _over(self, cascade: DeliveryCascade) -> None:
        """*cascade* ended: the delivery is over, unless it holds its event,
        whose release it waits for, its cancellation recorded on the event."""
        if cascade is not self.cascade:
            return  # a later cascade runs the delivery now
        self.cascade = None
        self._cancelled = cascade.cancelled
        if self.held is None:
            self.release()
        elif self._cancelled is not None:
            self._owner.held_cancelled(self.held, self._cancelled)


class DeliveryKeys:
    """Which deliveries each key names: in flight, or holding their event
    for verification."""

    def __init__(self, record: Callable[[tuple[str, str], str], None]) -> None:
        self._named: dict[str, list[NamedDelivery]] = {}
        # The deliveries holding their event for verification, by event id.
        self._held: dict[str, NamedDelivery] = {}
        self._record = record

    @contextmanager
    def naming(
        self, key: str | None, organization_id: str | None
    ) -> Generator[NamedDelivery, None, None]:
        """Name the delivery of a call acting for *organization_id* by *key*
        (nothing for ``None``) while the call runs; a cascade the call
        attaches keeps the name until its delivery is over."""
        named = NamedDelivery(key, organization_id, self)
        if key is not None:
            self._named.setdefault(key, []).append(named)
        try:
            yield named
        finally:
            named.release()

    def hold(self, named: NamedDelivery, room_id: str, event_id: str, lifetime: float) -> None:
        """*named* holds its event for verification (RFC §11.7.3): its
        delivery is over once the event is released (:meth:`unhold`), or
        once *lifetime* seconds have passed and nothing may release it."""
        self._expire()
        named.held = (room_id, event_id)
        named.until = time.monotonic() + lifetime
        self._held[event_id] = named

    def unhold(self, event_id: str) -> NamedDelivery | None:
        """The delivery that held *event_id*, now released (RFC §11.7.6), to
        name the cascade of its answer or to release; ``None`` when this
        process holds no delivery for the event."""
        named = self._held.pop(event_id, None)
        if named is not None:
            named.held = None
        return named

    def held_cancelled(self, held: tuple[str, str], reason: str) -> None:
        """A delivery holding *held* was cancelled for *reason*: record it."""
        self._record(held, reason)

    def release(self, named: NamedDelivery) -> None:
        """Stop naming *named* by its key: its delivery is over."""
        if named.key is None:
            return
        remaining = [entry for entry in self._named.get(named.key, ()) if entry is not named]
        if remaining:
            self._named[named.key] = remaining
        else:
            self._named.pop(named.key, None)

    def cancel(self, key: str, reason: str, *, organization_id: str | None = None) -> int:
        """Cancel every delivery *key* names, only those of
        *organization_id*'s rooms when given: how many this call cancelled,
        none for a key naming nothing (over, or never sent) or deliveries
        already cancelled."""
        self._expire()
        cancelled = 0
        for named in list(self._named.get(key, ())):
            if named.cancel(reason, organization_id):
                cancelled += 1
        return cancelled

    def _expire(self) -> None:
        """Release the held deliveries whose event nothing may release now."""
        now = time.monotonic()
        for event_id, named in list(self._held.items()):
            if named.until <= now:
                self.unhold(event_id)
                named.release()
