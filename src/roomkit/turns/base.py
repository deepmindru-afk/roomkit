"""The turn backend: what carries a turn between the room owner and an
executor (RFC §10.6).

A backend hands the owner a :class:`TurnLink` per turn it submits, and an
executor that serves it an :class:`ExecutorLink` per turn it takes. Each link
carries numbered messages both ways; :class:`MessageOrder` is the rule every
backend applies on arrival: a duplicate is dropped, a gap fails the turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import TurnLinkError

if TYPE_CHECKING:
    from roomkit.realtime.base import EphemeralEvent
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import (
        ChannelControl,
        ChannelResult,
        Control,
        Frame,
        OwnerTool,
        ToolCall,
        TurnRequest,
    )


@dataclass(frozen=True, slots=True)
class TurnTimeouts:
    """The bounds both sides of a turn keep (RFC §10.6.8), in seconds."""

    accept: float = 10.0
    """How long the owner waits for an executor to take the turn."""
    heartbeat_interval: float = 2.0
    """How often each side tells the other it is still there."""
    heartbeat_timeout: float = 10.0
    """How long a silence lasts before its side is taken for lost."""
    cancel: float = 5.0
    """How long the owner waits for a cancelled turn's end."""
    input_hold_timeout: float = 60.0
    """How long the owner holds an answer for a human-input request it does
    not know yet (RFC §10.6.5)."""

    @property
    def silent_intervals(self) -> int:
        """How many intervals in a row with no message make a side lost:
        counted per interval, so a stall of the counting side's own loop
        counts once."""
        return max(1, round(self.heartbeat_timeout / self.heartbeat_interval))


async def heartbeat_tick(interval: float) -> None:
    """Wait one heartbeat interval on the loop's own timer, so code that
    patches ``asyncio.sleep`` (a test, a host's clock) cannot make the
    heartbeats spin."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.Event().wait(), interval)


class MessageOrder:
    """The receiving end of one direction of a link: messages numbered from 1
    arrive exactly once and in order."""

    __slots__ = ("_last",)

    def __init__(self) -> None:
        self._last = 0

    def admit(self, seq: int) -> bool:
        """Whether the message *seq* is the next one; ``False`` for a
        duplicate, :class:`TurnLinkError` for a gap."""
        if seq <= self._last:
            return False
        if seq != self._last + 1:
            raise TurnLinkError("frame_lost", f"Message {self._last + 1} of the turn was lost")
        self._last = seq
        return True


class TurnLink(ABC):
    """The owner's side of one turn."""

    request: TurnRequest

    @property
    @abstractmethod
    def taken(self) -> bool:
        """Whether an executor took the request."""

    @abstractmethod
    async def receive(self) -> tuple[int, Frame]:
        """The next frame and its sequence number."""

    @abstractmethod
    async def send(self, message: Control) -> None:
        """Send a control message to the executor."""

    @abstractmethod
    def release(self) -> None:
        """Forget the turn: later frames are dropped."""


class ExecutorLink(ABC):
    """The executor's side of one turn."""

    request: TurnRequest

    @abstractmethod
    async def receive(self) -> Control:
        """The next control message from the owner."""

    @abstractmethod
    async def send(self, frame: Frame) -> int:
        """Send a frame to the owner; its sequence number."""

    def turn_context(self) -> contextvars.Context:
        """The execution context the turn's work runs in: a fresh one, unless
        the backend shares the owner's process (RFC §10.6.5)."""
        return contextvars.Context()

    def stream_context(self) -> contextvars.Context | None:
        """The execution context the turn's stream advances in, once known;
        ``None`` means a fresh one, kept for the stream's life."""
        return None


class OwnerTools(ABC):
    """The tools a room owner serves its turns (RFC §10.6.5): what it lists in
    a turn's request, and how it runs a call's handler."""

    @abstractmethod
    def declared(self, channel_id: str, room_id: str) -> list[OwnerTool]:
        """The tools the owner serves *channel_id*'s turns in *room_id*."""

    @abstractmethod
    async def serve(self, request: TurnRequest, call: ToolCall) -> Any:
        """Run *call*'s handler, under the call's context already installed:
        its answer. Raises what the handler raises, and
        ``UnservedToolCallError`` when nothing serves the call."""


class OwnerHost(ABC):
    """What a room owner runs for the ``TurnHost`` of its turns (RFC §10.6.5)."""

    @abstractmethod
    async def call(self, request: TurnRequest, verb: str, arguments: dict[str, Any]) -> Any:
        """Run *verb* for the turn of *request*: its result, as data JSON
        carries. Raises what the verb raises; ``ValueError`` for a verb or
        arguments it does not know."""


class OwnerEvents(ABC):
    """Where a room owner publishes what its turns published on their
    executors (RFC §10.6.3): ephemeral events on its realtime backend,
    framework events on its own bus."""

    @abstractmethod
    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        """Publish *event* on the realtime channel *channel*."""

    @abstractmethod
    async def emit(
        self,
        event_type: str,
        *,
        room_id: str | None,
        channel_id: str | None,
        event_id: str | None,
        data: dict[str, Any],
    ) -> None:
        """Emit the framework event *event_type*."""


class TurnBackend(ABC):
    """Carries turns between room owners and executors (RFC §10.6)."""

    @abstractmethod
    async def submit(self, request: TurnRequest) -> TurnLink:
        """Start a turn: an executor that declares its channel takes it."""

    @abstractmethod
    async def serve(self, executor: TurnExecutor) -> None:
        """Have *executor* take the turns of the channels it declares, from
        now until the backend closes or the executor does."""

    @abstractmethod
    async def control(self, message: ChannelControl) -> ChannelResult | None:
        """Carry a channel control message (RFC §10.6.3): ``room_attached``
        and ``room_detached`` to every executor that declares the channel,
        answered by none; a ``channel_call`` to the executor that holds the
        room's state for the channel, else one that declares it, answered
        by its result.

        Raises:
            TurnLinkError: ``no_executor`` for a ``channel_call`` no executor
                declares.
        """

    @abstractmethod
    async def close(self) -> None:
        """Fail the turns in flight and release the transport."""
