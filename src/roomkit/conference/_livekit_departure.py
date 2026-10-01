"""How one bot session ends: asked to leave, dropped by the SFU, or ended as unhealthy.

Three ways out, one owner. Whether the session still admits work — a chunk to
publish, an event to queue — is this object's to say, and nothing else writes
it. The order of a teardown lives here too: the media stops, the voice is
released, the bot is taken out of the SFU, the event bridge stops, and only
then is an end reported. Reporting before the bot is out would empty the
backend's registry and seat a replacement beside a live connection (RFC
12.10.3).

The bot is taken out through the SDK when the SDK can, and through the server
when it cannot: in livekit-rtc 1.1.20 a failed publish or unpublish leaves the
SDK's room listener stuck, and its ``disconnect()`` never returns.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.conference._livekit_bridge import EventBridge
from roomkit.conference._livekit_media import TrackPumps
from roomkit.conference._livekit_voice import BotVoiceTrack
from roomkit.conference.models import BotSession

logger = logging.getLogger("roomkit.conference.livekit")


def _consume_exception(task: asyncio.Task[Any]) -> None:
    """Take a shared task's parting error so no waiter-less run warns."""
    if not task.cancelled():
        task.exception()


# How long to wait before re-attempting the disconnect an unhealthy-session
# teardown needs. Unlike the SFU-drop path, an overflow ends a session whose
# connection is still live: the end must not be *reported* until the old bot
# is genuinely out, or the re-join would seat a replacement beside it. The
# first attempt is immediate; these pace the retries.
OVERFLOW_DISCONNECT_DELAYS_S: tuple[float, ...] = (1.0, 2.0, 4.0)

# How long the SDK's disconnect may take before the server is asked to confirm
# the bot is out. A disconnect normally returns in milliseconds; one that does
# not is waiting on the SDK's room listener, which a failed publish or unpublish
# leaves stuck for good (livekit-rtc 1.1.20).
DISCONNECT_TIMEOUT_S = 2.0

# How long that confirmation may take. The two bounds together stay under the
# channel's detach budget (``_conference_activity.DRAIN_TIMEOUT_S``, checked by
# a test), so the backend settles a departure before the channel gives up on it.
EVICTION_TIMEOUT_S = 2.0


class SessionDeparture:
    """End one bot session, whichever way it ends, exactly once."""

    def __init__(
        self,
        *,
        room: Any,
        session: BotSession,
        voice: BotVoiceTrack,
        pumps: TrackPumps,
        bridge: EventBridge,
        report_end: Callable[[BotSession, str], Awaitable[None]],
        evict: Callable[[], Awaitable[None]],
    ) -> None:
        self._room = room
        self.session = session
        self._voice = voice
        self._pumps = pumps
        self._bridge = bridge
        self._report_end = report_end
        self._evict = evict  # removes this bot through the server API
        self._left = False
        self._disconnected = False
        # The one disconnect in flight, whoever asked for it — see
        # `_disconnect_once` — and whether a *requested* leave() has asked:
        # an unhealthy end that loses that race reports nothing, because the
        # caller that requested the departure owns the books.
        self._disconnecting: asyncio.Task[None] | None = None
        self._sdk_disconnecting: asyncio.Task[None] | None = None
        self._leave_requested = False
        # The teardown task an SFU-side or unhealthy end runs on; the callbacks
        # that start one are synchronous. Kept referenced until it ends.
        self._ender: asyncio.Task[None] | None = None

    @property
    def room_id(self) -> str:
        return self.session.room_id

    @property
    def admitting(self) -> bool:
        """Whether the session still takes work: nothing has started ending it."""
        return not self._left

    async def leave(self) -> None:
        """Disconnect, and take the in-flight utterance with the session.

        RFC section 12.10.4: an utterance the channel abandons because it is
        leaving is not closed by a terminal chunk — the session that chunk would
        name is on its way out. So the session going away *is* the boundary, and
        what that means here is that the queued audio goes unplayed and the
        track goes with it. Nothing is published on the way out.

        A disconnect the SDK refuses *propagates*. Swallowing it here reported
        a success to a channel whose entire departure bookkeeping — the leaving
        ledger, ``info()``'s bot_present, the close's final raise — exists to
        never misstate whether the bot is out of the meeting (RFC 12.10.4:
        failing to remove a session is failing to close). The local teardown
        that already ran stays torn down; a retry reattempts the disconnect
        alone, and only a disconnect that returned makes later calls no-ops.

        ``_leave_requested`` is set before the first await: an unhealthy end
        may be mid-disconnect on its own task, and the flag is how it learns —
        after the one shared disconnect returns — that a *requested* leave now
        owns the books and nothing spontaneous is to be reported.
        """
        self._leave_requested = True
        if self._disconnected:
            return
        if not self._left:
            self._left = True
            if self._voice.abandon_utterance():
                logger.debug(
                    "Conference bot %s left room %s mid-utterance; the session ends it",
                    self.session.identity,
                    self.room_id,
                )
            await self._pumps.close()
            await self._voice.close()
        await self._disconnect_once()
        await self._bridge.stop()

    def dropped(self, reason: Any = None) -> None:
        """The SFU dropped the bot, which ends the session in fact.

        Not the same as :meth:`leave`: nothing here was asked for, and there
        is nowhere to publish a boundary — a dropped connection announces
        nothing, any more than a crashed process would (RFC section 12.10.4).
        What there *is* somewhere to send is the fact itself: the session's
        local state is torn down on a task of its own — this callback is
        synchronous — and the end is reported through ``bot_session_ended``,
        because a loss the framework never hears about is a bot it reports
        present forever.
        """
        if self._voice.abandon_utterance():
            logger.warning(
                "LiveKit disconnected the conference bot in room %s mid-utterance (%s)",
                self.room_id,
                reason,
            )
        else:
            logger.info(
                "LiveKit disconnected the conference bot in room %s (%s)", self.room_id, reason
            )
        if self._left:
            return
        # Admission closes here, in the synchronous callback: a leave() racing
        # this teardown must find the session already ended, not tear it down
        # a second time beside it.
        self._left = True
        self._disconnected = True
        self._ender = asyncio.create_task(self._ended_by_sfu(str(reason)))

    def end_unhealthy(self, reason: str) -> None:
        """Close admission and end the live session on a task of its own."""
        if self._left:
            return
        # Admission closes, but the session is NOT marked disconnected: unlike
        # the SFU-drop path, this connection is still live, and the end must
        # not be reported — nor the registry emptied, nor a replacement
        # seated — until the disconnect has actually happened.
        self._left = True
        self._ender = asyncio.create_task(self._end_unhealthy(reason))

    async def _disconnect_once(self) -> None:
        """One disconnect on the wire at a time, shared by every path.

        ``leave()`` and an unhealthy end can both need the disconnect, and
        each may be suspended in it when the other arrives — two concurrent
        calls into the SDK, and two owners for one outcome. Single-flight:
        the first caller starts the call, later callers await the same one
        (shielded, so a budget cancelling *a caller* does not cancel the call
        the other still awaits). A call that succeeded stays the answer: a
        caller cancelled while it ran finds it done on its retry, rather than a
        second disconnect into an SDK that has already let the room go. A call
        that failed propagates to every waiter and is not terminal — the next
        caller starts a fresh attempt.
        """
        if self._disconnected:
            return
        call = self._disconnecting
        if call is None or (call.done() and (call.cancelled() or call.exception() is not None)):
            call = self._disconnecting = asyncio.create_task(self._leave_the_sfu())
            call.add_done_callback(_consume_exception)
        await asyncio.shield(call)
        self._disconnected = True

    async def _leave_the_sfu(self) -> None:
        """Take the bot out: through the SDK, or through the server if the SDK hangs.

        A disconnect that does not return within :data:`DISCONNECT_TIMEOUT_S` is
        not a departure anyone can book, and waiting on it is a channel that
        never finishes leaving. The server is the authority on who is in the
        room, so it is asked to remove the bot; a bot it no longer knows is out.
        Only then is the stuck listener released. An eviction that fails or
        outlasts :data:`EVICTION_TIMEOUT_S` propagates, exactly as a refused
        disconnect does, and the next attempt waits on the same SDK call again
        before asking the server once more.
        """
        if await self._sdk_disconnect():
            return
        logger.warning(
            "The LiveKit SDK did not finish disconnecting the conference bot in room %s "
            "within %.1fs; asking the server to confirm the bot is out",
            self.room_id,
            DISCONNECT_TIMEOUT_S,
        )
        await asyncio.wait_for(self._evict(), EVICTION_TIMEOUT_S)
        self._release_sdk_listener()

    async def _sdk_disconnect(self) -> bool:
        """Whether the SDK's disconnect returned within :data:`DISCONNECT_TIMEOUT_S`.

        One SDK call for the session, however many attempts wait on it: a second
        ``disconnect()`` beside a pending one would be two calls into the SDK
        for one room. A call that failed is replaced by a fresh one, and its
        error propagates, so a refused disconnect stays a failed departure.
        """
        call = self._sdk_disconnecting
        if call is None or call.done():
            call = self._sdk_disconnecting = asyncio.create_task(self._room.disconnect())
            call.add_done_callback(_consume_exception)
        done, _ = await asyncio.wait({call}, timeout=DISCONNECT_TIMEOUT_S)
        if not done:
            return False
        call.result()
        return True

    def _release_sdk_listener(self) -> None:
        """Cancel the SDK's room listener a failed publish or unpublish left stuck.

        It is what the SDK's disconnect waits on, and it keeps the room's FFI
        subscription, which would otherwise queue every event of the process for
        a room nobody reads. Private to the SDK, hence the guard: an SDK without
        it has nothing to release.
        """
        listener = getattr(self._room, "_task", None)
        if isinstance(listener, asyncio.Task) and not listener.done():
            listener.cancel()

    async def _ended_by_sfu(self, reason: str) -> None:
        """Tear the session down after the SFU dropped it, and say so.

        The connection is already gone, so the disconnect attempt is a
        harmless best-effort no-op; what matters is the report. The overflow
        path is deliberately not this one — see :meth:`_end_unhealthy` — its
        connection is still live and the report has to wait for the
        disconnect.
        """
        await self._pumps.close()
        await self._voice.close()
        with contextlib.suppress(Exception):
            if not await self._sdk_disconnect():
                self._release_sdk_listener()
        await self._finish_end(reason)

    async def _end_unhealthy(self, reason: str) -> None:
        """End a live session whose view can no longer be trusted.

        The report comes *after* the disconnect, never before: reporting the
        session ended empties the backend's registry and seats a replacement,
        and doing that while the old connection is still up is two bots in
        one meeting. The disconnect is retried on a short backoff; a session
        whose disconnect will not go through is *kept* — on the backend's
        registry, on the channel's books, refusing a replacement — and said
        out loud. A later ``leave()`` (a detach, the close) retries the
        disconnect: failure is not terminal, exactly as in :meth:`leave`.
        """
        await self._pumps.close()
        await self._voice.close()
        for attempt, delay in enumerate((0.0, *OVERFLOW_DISCONNECT_DELAYS_S)):
            if self._leave_requested or self._disconnected:
                # A requested leave() owns the books from the moment it asks;
                # nothing spontaneous is reported over it.
                return
            if delay:
                await asyncio.sleep(delay)
            try:
                await self._disconnect_once()
            except Exception:
                logger.warning(
                    "Disconnecting the unhealthy conference bot in room %s failed "
                    "(attempt %d); the session is not reported ended while the bot may "
                    "still be connected",
                    self.room_id,
                    attempt + 1,
                    exc_info=True,
                )
                continue
            # Re-read the owner *after* the shared disconnect: a leave() that
            # arrived while the call was on the wire shared its outcome, and
            # the outcome is the leave's to book, not this path's to report.
            if self._leave_requested:
                return
            await self._finish_end(reason)
            return
        logger.error(
            "The unhealthy conference bot in room %s could not be disconnected after "
            "%d attempt(s). The session is being kept — on the backend's registry and "
            "the channel's books, holding the room's conference slot — rather than "
            "reported ended beside a live connection. A detach or the channel close "
            "retries the disconnect",
            self.room_id,
            len(OVERFLOW_DISCONNECT_DELAYS_S) + 1,
        )

    async def _finish_end(self, reason: str) -> None:
        """Stop the bridge and report the session's end, loss counted."""
        undelivered = await self._bridge.stop()
        if undelivered:
            reason = (
                f"{reason}; {undelivered} queued event(s) were discarded undelivered — "
                "what happened while the consumer was stalled is not recoverable"
            )
        await self._report_end(self.session, reason)
