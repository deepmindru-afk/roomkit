"""The deliveries in flight a caller named by a key (RFC §10.1, *Explicit
delivery cancellation*).

A caller that sets ``InboundMessage.delivery_key`` can cancel the message's
delivery from anywhere in the process, without holding its handle: an agent's
tool passes the key along as text. A key names every delivery in flight that
carries it, from the call's start until its delivery is over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade


class DeliveryKeys:
    """Which deliveries in flight each key names."""

    def __init__(self) -> None:
        self._named: dict[str, list[DeliveryCascade]] = {}

    def hold(self, key: str | None, cascade: DeliveryCascade) -> None:
        """Name *cascade* by *key* (nothing for ``None``) while it runs."""
        if key is not None:
            self._named.setdefault(key, []).append(cascade)

    def release(self, key: str | None, cascade: DeliveryCascade) -> None:
        """Stop naming *cascade* by *key*: its delivery is over."""
        if key is None:
            return
        named = self._named.get(key, [])
        if cascade in named:
            named.remove(cascade)
        if not named:
            self._named.pop(key, None)

    def cancel(self, key: str, reason: str) -> int:
        """Cancel every delivery in flight *key* names: how many this call
        cancelled, none for a key naming nothing in flight (over, or never
        sent) or deliveries already cancelled."""
        cancelled = 0
        for cascade in list(self._named.get(key, ())):
            if cascade.cancelled is None:
                cascade.cancel(reason)
                cancelled += 1
        return cancelled
