"""The injected events a locked section queues for their passes (RFC §9.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

from roomkit.models.event import RoomEvent
from roomkit.models.hook import InjectedEvent


@dataclass(slots=True)
class QueuedInjections:
    """The events the hook results of one locked section injected, each with
    the event whose hooks injected it, in the order they were queued.

    Their passes run once the section has released the room lock (RFC §9.4,
    §10.1 steps 10 to 14): queued under it, each one then commits in a pass
    and a section of its own, so another event may commit between a cause
    and what its hooks injected.
    """

    entries: list[tuple[list[InjectedEvent], RoomEvent]] = field(default_factory=list)

    def add(self, injected_events: list[InjectedEvent], cause: RoomEvent) -> None:
        """Queue *injected_events*, injected by the hooks of *cause*."""
        if injected_events:
            self.entries.append((list(injected_events), cause))

    def take(self) -> list[tuple[list[InjectedEvent], RoomEvent]]:
        """The entries queued so far, the queue left empty."""
        entries, self.entries = self.entries, []
        return entries
