"""The deliveries in flight a caller named by a key (RFC §10.1, *Explicit
delivery cancellation*).

A caller that sets ``InboundMessage.delivery_key`` can cancel the message's
delivery from anywhere in the process, without holding its handle: an agent's
tool passes the key along as text. A key names every delivery in flight that
carries it, from the call's start until its delivery is over; a cancellation
scoped to an organization reaches only the deliveries of its rooms (RFC
§17.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade


class DeliveryKeys:
    """Which deliveries in flight each key names, with their room's
    organization."""

    def __init__(self) -> None:
        self._named: dict[str, list[tuple[DeliveryCascade, str | None]]] = {}

    def hold(self, key: str | None, cascade: DeliveryCascade, organization_id: str | None) -> None:
        """Name *cascade*, a delivery in a room of *organization_id*, by *key*
        (nothing for ``None``) until its delivery is over."""
        if key is None:
            return
        self._named.setdefault(key, []).append((cascade, organization_id))
        cascade.when_over(lambda: self._release(key, cascade))

    def _release(self, key: str, cascade: DeliveryCascade) -> None:
        """Stop naming *cascade* by *key*: its delivery is over."""
        named = [entry for entry in self._named.get(key, ()) if entry[0] is not cascade]
        if named:
            self._named[key] = named
        else:
            self._named.pop(key, None)

    def cancel(self, key: str, reason: str, *, organization_id: str | None = None) -> int:
        """Cancel every delivery in flight *key* names, only those of
        *organization_id*'s rooms when given: how many this call cancelled,
        none for a key naming nothing in flight (over, or never sent) or
        deliveries already cancelled."""
        cancelled = 0
        for cascade, owner in list(self._named.get(key, ())):
            if organization_id is not None and owner != organization_id:
                continue
            if cascade.cancelled is None:
                cascade.cancel(reason)
                cancelled += 1
        return cancelled
