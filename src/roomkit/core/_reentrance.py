"""What a call into a room is to the room's own work (RFC §9.5.1 Reentrance,
§8.3, §10.5, §10.1 step 18).

The execution context of a call tells it, for the room it names:

- under the room's lock, in the section a pass holds (a locked hook, code
  under the lock): it is *nested*, adopted by that pass, drawing on the
  reentry budget of the cascade whose section holds the lock, and what its
  own hooks inject joins that section's queue (§9.4);
- in the room's admission order (an off-lock hook's body, a message's parse):
  it is adopted *beside* the event that work is for, takes no ticket and
  draws on that event's budget;
- in the room's delivery work (a delivery set's deliveries and turns): it is a
  new *arrival*, with its ticket and its ``process_timeout``, drawing on the
  budget of the cascade that delivers, so a transport whose delivery calls
  into its room on every event stops at the cap;
- elsewhere: a new arrival with a budget of its own.

A call the room's work made is charged one pass as it starts. A call made
within a turn, or by work the turn began, is at least as deep as the turn's
response (§8.3, §10.6).

A frame stands for its work only while that work runs: a task the work
spawns inherits the frame and finds it inactive once the work is over, as a
room lock's lease is (:mod:`roomkit.core.locks`).
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from roomkit.core.exceptions import ReentrantCallError
from roomkit.core.locks import _has_room_lock
from roomkit.tools.context import _current_turn_chain_depth

if TYPE_CHECKING:
    from roomkit.core._queued_injections import QueuedInjections
    from roomkit.core.locks import RoomLockManager


REENTRY_LOOP_CAP = "reentry_loop_cap"
"""The ``blocked_by`` of what a cascade's budget no longer admits (RFC §8.3)."""


class ReentryBudget:
    """The passes a cascade may still run (RFC §8.3): its responses, chained
    streams and injected events, and the entries the room's work adopted or
    sent into it. Shared by reference between the cascades that draw on
    it."""

    __slots__ = ("_left",)

    def __init__(self, passes: int) -> None:
        self._left = passes

    def take(self) -> bool:
        """Take one pass; ``False`` once the budget is spent."""
        if self._left <= 0:
            return False
        self._left -= 1
        return True


class Placement(Enum):
    """Where a call into a room stands against the room's work."""

    NESTED = "nested"
    """Under the room's lock: in the section that holds it."""
    BESIDE = "beside"
    """In the room's admission order: beside the event its caller works for."""
    ARRIVAL = "arrival"
    """A new arrival: its own ticket and ``process_timeout``."""


@dataclass(frozen=True, slots=True)
class Entry:
    """What a call into a room is to the room's work."""

    placement: Placement
    budget: ReentryBudget
    charged: bool
    """Whether the room's work made the call: it draws one pass as it starts."""
    queued: QueuedInjections | None = None
    """Nested in a pass's section: that section's queue of injected events."""

    @property
    def adopted(self) -> bool:
        """Whether the call takes no ticket and runs every hook under the lock."""
        return self.placement is not Placement.ARRIVAL


class _Kind(Enum):
    SECTION = "section"
    ADMITTED = "admitted"
    DELIVERING = "delivering"


@dataclass(eq=False, slots=True)
class _Frame:
    kind: _Kind
    room_id: str
    budget: ReentryBudget
    queued: QueuedInjections | None = None
    active: bool = True


_frames: contextvars.ContextVar[tuple[_Frame, ...]] = contextvars.ContextVar(
    "_roomkit_room_work_frames", default=()
)
# The response depth of the deepest turn the context runs within (RFC §8.3).
# Unlike a frame it is never made inert: a task the turn spawns keeps it once
# the turn is over, since that task is the turn's work, which reaches the room
# after the turn at the turn's depth (RFC §10.6.5, as a delegated result does,
# §23.3). Made inert, an agent answering through background work at each turn
# would send at depth 0 and never meet the limit; its budget, a frame's, is
# its own then.
_turn_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_roomkit_turn_response_depth", default=0
)


@contextmanager
def _framed(frame: _Frame) -> Generator[None, None, None]:
    token = _frames.set((*_frames.get(), frame))
    try:
        yield
    finally:
        # Copies the work's tasks hold share this frame: over, it is inert.
        frame.active = False
        _frames.reset(token)


def section(
    room_id: str, budget: ReentryBudget, queued: QueuedInjections
) -> AbstractContextManager[None]:
    """The locked section of a pass of *room_id*, on the cascade's *budget*,
    the events its hooks inject queued on *queued*."""
    return _framed(_Frame(_Kind.SECTION, room_id, budget, queued))


def admission_order(room_id: str, budget: ReentryBudget) -> AbstractContextManager[None]:
    """Work standing in *room_id*'s admission order for an event drawing on
    *budget*: an off-lock hook's body, a message's parse (RFC §9.5.1)."""
    return _framed(_Frame(_Kind.ADMITTED, room_id, budget))


def delivering(room_id: str, budget: ReentryBudget) -> AbstractContextManager[None]:
    """The deliveries and turns of a delivery set of *room_id*, run for the
    cascade drawing on *budget*."""
    return _framed(_Frame(_Kind.DELIVERING, room_id, budget))


@contextmanager
def within_turn(response_depth: int) -> Generator[None, None, None]:
    """A turn whose response is *response_depth* deep: what the context calls
    into a room is at least that deep (RFC §8.3, §10.6), the work it spawns
    included, after the turn too (see ``_turn_depth``)."""
    token = _turn_depth.set(max(response_depth, _turn_depth.get()))
    try:
        yield
    finally:
        _turn_depth.reset(token)


def chain_floor() -> int:
    """The least chain depth of a call made in this context: the response
    depth of the turn it runs within, its tool calls' included; 0 outside a
    turn."""
    return max(_turn_depth.get(), _current_turn_chain_depth())


class RoomWork:
    """Tells a call into a room what it is to the room's work, from the
    frames of its execution context and the room locks it holds.

    *lock_manager* is the kit's room lock; *passes* the reentry budget of a
    cascade of its own (``10 × max_chain_depth``, RFC §8.3).
    """

    def __init__(self, lock_manager: RoomLockManager, passes: int) -> None:
        self._lock_manager = lock_manager
        self._passes = passes

    def budget(self) -> ReentryBudget:
        """A cascade's budget of its own."""
        return ReentryBudget(self._passes)

    def entry(self, room_id: str) -> Entry:
        """What a call into *room_id* made in this context is (RFC §9.5.1)."""
        locked = _has_room_lock(room_id, self._lock_manager)
        frame = _innermost(room_id, locked=locked)
        if locked:
            if frame is None:
                # Code under the lock outside any of the room's work (a host
                # serialising its changes): nested, on a budget of its own.
                return Entry(Placement.NESTED, self.budget(), False)
            section = _innermost(room_id, locked=True, kind=_Kind.SECTION)
            queued = section.queued if section is not None else None
            return Entry(Placement.NESTED, frame.budget, True, queued)
        if frame is not None and frame.kind is _Kind.ADMITTED:
            return Entry(Placement.BESIDE, frame.budget, True)
        if frame is not None and frame.kind is _Kind.DELIVERING:
            return Entry(Placement.ARRIVAL, frame.budget, True)
        return Entry(Placement.ARRIVAL, self.budget(), False)

    def refuse_routed(self, room_id: str) -> None:
        """Refuse a message routed into *room_id* from under its lock or from
        its admission order (RFC §9.5.1, §10.1 step 2): known only once
        routed, the room cannot adopt it, and committed it would come ahead
        of the event its caller works for, outside every order.

        Raises:
            ReentrantCallError: the context holds the room's lock or stands in
                its admission order.
        """
        locked = _has_room_lock(room_id, self._lock_manager)
        frame = _innermost(room_id, locked=locked)
        if locked or (frame is not None and frame.kind is _Kind.ADMITTED):
            raise ReentrantCallError(
                f"A message routed into room {room_id} from its own work (under its lock, "
                "in an off-lock hook or a parse) cannot be adopted: name the room to send "
                "into it"
            )


def _innermost(room_id: str, *, locked: bool, kind: _Kind | None = None) -> _Frame | None:
    """The innermost frame of *room_id* whose work still runs, of *kind* when
    given; a section counts only for a context that holds the room's lock
    (*locked*), which a task started in a context holding none does not."""
    for frame in reversed(_frames.get()):
        if not frame.active or frame.room_id != room_id:
            continue
        if frame.kind is _Kind.SECTION and not locked:
            continue
        if kind is None or frame.kind is kind:
            return frame
    return None
