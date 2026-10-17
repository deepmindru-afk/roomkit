"""Where a turn's realtime publications and framework events go (RFC §10.6.2):
during a turn, as ``ephemeral`` and ``framework_event`` frames its owner
publishes; outside one, to what the executor was given, else nowhere.

A channel the executor wires publishes through :class:`TurnRealtime` and its
hooks emit through :func:`emit_framework_event`; both find the turn they run
under in the context (:func:`current_turn_outlet`), as the turn's other
per-call state is found, so the code that publishes is the same on every
executor, the kit's own included.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.realtime.base import EphemeralCallback, EphemeralEvent, RealtimeBackend
from roomkit.turns.wire import Ephemeral, FrameworkEventFrame

if TYPE_CHECKING:
    from roomkit.turns.wire import Frame

logger = logging.getLogger("roomkit.turns")

FrameworkEmitter = Callable[..., Awaitable[None]]
"""Emits a framework event: ``emit(event_type, room_id=, channel_id=,
event_id=, data=)``, as ``RoomKit._emit_framework_event`` does."""

_TURN_OUTLET: contextvars.ContextVar[TurnOutlet | None] = contextvars.ContextVar(
    "roomkit_turn_outlet", default=None
)


def current_turn_outlet() -> TurnOutlet | None:
    """The outlet of the turn the current code runs under, ``None`` outside one."""
    return _TURN_OUTLET.get()


def set_turn_outlet(outlet: TurnOutlet | None) -> None:
    """Install *outlet* as the current turn's, in the current context."""
    _TURN_OUTLET.set(outlet)


class TurnOutlet:
    """Sends one turn's ephemeral events and framework events to its owner,
    until the turn ends."""

    def __init__(
        self,
        send: Callable[[Frame], Awaitable[None]],
        *,
        after: Callable[[Ephemeral | FrameworkEventFrame], Awaitable[None]] | None = None,
    ) -> None:
        self._send = send
        self._after = after
        self._open = True

    async def publish(self, channel: str, event: EphemeralEvent) -> bool:
        """Send *event* published on *channel*; whether it was carried, by the
        turn or, after it, to the owner outside a turn."""
        return await self._carry(Ephemeral(channel=channel, event=event.to_dict()))

    async def emit(
        self,
        event_type: str,
        *,
        room_id: str | None = None,
        channel_id: str | None = None,
        event_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send a framework event; whether it was carried, by the turn or,
        after it, to the owner outside a turn."""
        frame = FrameworkEventFrame(
            event_type=event_type,
            room_id=room_id,
            channel_id=channel_id,
            event_id=event_id,
            data=data or {},
        )
        return await self._carry(frame)

    async def _carry(self, frame: Ephemeral | FrameworkEventFrame) -> bool:
        if self._open:
            await self._send(frame)
            return True
        if self._after is None:
            return False
        await self._after(frame)
        return True

    def close(self) -> None:
        """The turn ended: what is published from now goes to the owner
        outside a turn (RFC §10.6.3), else elsewhere."""
        self._open = False


class TurnRealtime(RealtimeBackend):
    """The realtime backend an executor gives the channels it wires: a
    publication made during a turn crosses to the owner as an ``ephemeral``
    frame; one made outside goes to *fallback()*, the kit's own realtime for
    the kit's executor, and is dropped when there is none.

    Args:
        fallback: The backend to publish to outside a turn, read when used.
    """

    def __init__(self, fallback: Callable[[], RealtimeBackend | None]) -> None:
        self._fallback = fallback

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        outlet = current_turn_outlet()
        if outlet is not None and await outlet.publish(channel, event):
            return
        backend = self._fallback()
        if backend is None:
            logger.debug("Ephemeral %s on %s, outside a turn: dropped", event.type, channel)
            return
        await backend.publish(channel, event)

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        # The fallback's own room publish, which a backend may implement apart.
        outlet = current_turn_outlet()
        if outlet is not None and await outlet.publish(f"room:{room_id}", event):
            return
        backend = self._fallback()
        if backend is None:
            logger.debug("Ephemeral %s in room %s, outside a turn: dropped", event.type, room_id)
            return
        await backend.publish_to_room(room_id, event)

    async def subscribe(self, channel: str, callback: EphemeralCallback) -> str:
        return await self._backend().subscribe(channel, callback)

    async def unsubscribe(self, subscription_id: str) -> bool:
        return await self._backend().unsubscribe(subscription_id)

    def _backend(self) -> RealtimeBackend:
        backend = self._fallback()
        if backend is None:
            raise RuntimeError("This executor has no realtime backend to subscribe on")
        return backend


async def emit_framework_event(
    fallback: FrameworkEmitter | None,
    event_type: str,
    *,
    room_id: str | None = None,
    channel_id: str | None = None,
    event_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Emit a framework event: to the owner during a turn, else to *fallback*
    (the kit's own emitter for the kit's executor), else nowhere."""
    outlet = current_turn_outlet()
    fields: dict[str, Any] = {
        "room_id": room_id,
        "channel_id": channel_id,
        "event_id": event_id,
        "data": data,
    }
    if outlet is not None and await outlet.emit(event_type, **fields):
        return
    if fallback is None:
        logger.debug("Framework event %s emitted outside a turn: dropped", event_type)
        return
    await fallback(event_type, **fields)
