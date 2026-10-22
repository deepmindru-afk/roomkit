"""The deliveries in flight a caller named by a key (RFC §10.1, *Explicit
delivery cancellation*).

A caller that sets ``InboundMessage.delivery_key`` can cancel the message's
delivery from anywhere in the process, without holding its handle: an agent's
tool passes the key along as text. A key names every delivery in flight that
carries it, from the call's start until its delivery is over: a delivery
cancelled before its event is committed is committed and delivered all the
same, and none of its turns starts. A cancellation scoped to an organization
reaches only the deliveries of its rooms (RFC §17.2); a delivery whose room is
not known yet (a call that did not name its organization, before its commit)
takes it once its room proves the organization's, uncounted.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade


class NamedDelivery:
    """A delivery a key names, from its call's start: its cascade once the
    call commits, the organization of its room once known."""

    __slots__ = ("_cancelled", "_located", "_organization_id", "_release", "_requests", "cascade")

    def __init__(
        self, organization_id: str | None, release: Callable[[NamedDelivery], None]
    ) -> None:
        self.cascade: DeliveryCascade | None = None
        # A call that names its organization lands in one of its rooms or is
        # refused (RFC §17.2), so its delivery is located from the start.
        self._organization_id = organization_id
        self._located = organization_id is not None
        self._release = release
        # A cancellation received before the cascade exists.
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
        if self.cascade is None:
            self._cancelled = reason
        else:
            self.cascade.cancel(reason)
        return True

    def attach(self, cascade: DeliveryCascade, organization_id: str | None) -> None:
        """Make *cascade*, in a room of *organization_id*, the delivery named:
        cancelled at once when the delivery already was, named until it is
        over."""
        self.cascade = cascade
        self._organization_id = organization_id
        self._located = True
        reason = self._cancelled or next(
            (asked for owner, asked in self._requests if owner == organization_id), None
        )
        if reason is not None:
            cascade.cancel(reason)
        cascade.when_over(partial(self._release, self))

    def release(self) -> None:
        """Stop naming the delivery: its call ended before any cascade."""
        self._release(self)


class DeliveryKeys:
    """Which deliveries in flight each key names."""

    def __init__(self) -> None:
        self._named: dict[str, list[NamedDelivery]] = {}

    @contextmanager
    def naming(
        self, key: str | None, organization_id: str | None
    ) -> Generator[NamedDelivery, None, None]:
        """Name the delivery of a call acting for *organization_id* by *key*
        (nothing for ``None``) while the call runs; a cascade the call
        attaches keeps the name until its delivery is over."""
        named = NamedDelivery(organization_id, partial(self._release, key))
        if key is not None:
            self._named.setdefault(key, []).append(named)
        try:
            yield named
        finally:
            if named.cascade is None:
                named.release()

    def _release(self, key: str | None, named: NamedDelivery) -> None:
        """Stop naming *named* by *key*: its delivery is over."""
        if key is None:
            return
        remaining = [entry for entry in self._named.get(key, ()) if entry is not named]
        if remaining:
            self._named[key] = remaining
        else:
            self._named.pop(key, None)

    def cancel(self, key: str, reason: str, *, organization_id: str | None = None) -> int:
        """Cancel every delivery in flight *key* names, only those of
        *organization_id*'s rooms when given: how many this call cancelled,
        none for a key naming nothing in flight (over, or never sent) or
        deliveries already cancelled."""
        cancelled = 0
        for named in list(self._named.get(key, ())):
            if named.cancel(reason, organization_id):
                cancelled += 1
        return cancelled
