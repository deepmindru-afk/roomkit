"""The ASYNC observers of a pass's ``BEFORE_BROADCAST`` chain (RFC §9.2,
§10.1).

A pass runs the chain under its room's lock, before its commit, and its
observers start once the pass is over: its event committed and its delivery
set placed in the room's lane, whatever the store's commit took. An observer
that closes the kit then finds its event's set in the lane, running when the
lane was idle (RFC §12.10.4).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent


@dataclass(slots=True)
class Observed:
    """What a pass's chain owes its observers: the event it allowed, as its
    SYNC hooks left it, in the context they were handed; nothing when it
    blocked."""

    owed: tuple[RoomEvent, RoomContext] | None = None

    def owe(self, event: RoomEvent, context: RoomContext) -> None:
        """The chain allowed *event*, read in *context*."""
        self.owed = (event, context)


@contextlib.asynccontextmanager
async def observed_once_over(hooks: HookEngine, room_id: str) -> AsyncGenerator[Observed]:
    """Around one pass of *room_id*: what its chain owes is started as it
    ends, on every path, a refusal, a failure or a cut included, in the
    pass's own context (the cascade's budget its section draws on)."""
    observed = Observed()
    try:
        yield observed
    finally:
        if observed.owed is not None:
            event, context = observed.owed
            await hooks.run_observers(room_id, HookTrigger.BEFORE_BROADCAST, event, context)
