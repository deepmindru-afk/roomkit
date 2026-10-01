"""The framework's single connection to one LiveKit room.

``LiveKitBotSession`` is the facade the backend holds, one per conference. It
owns the ``rtc.Room`` and composes the parts that do the work, each in a module
of its own:

- ``_livekit_room_view``: the room's books (who is there, what they publish,
  what the framework wants) and the emissions their changes become;
- ``_livekit_bridge``: the ordered, bounded fanout of those emissions;
- ``_livekit_media``: the frames of each subscribed track, one pump per track;
- ``_livekit_voice``: the one track the AI speaks on;
- ``_livekit_departure``: how the session ends, and whether it still admits work.

What stays here is the wiring: which SDK event goes to which part, the gate
every queued event and published chunk passes, and the session's answer to a
bridge that is full or a voice LiveKit refuses.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from roomkit.conference._livekit_bridge import MAX_QUEUED_EVENTS, EventBridge
from roomkit.conference._livekit_departure import SessionDeparture
from roomkit.conference._livekit_media import AudioSink, TrackPumps, VideoSink
from roomkit.conference._livekit_room_view import RoomView
from roomkit.conference._livekit_voice import BotVoiceTrack, VoicePublicationError
from roomkit.conference.models import (
    BotSession,
    ConferenceParticipant,
    ConferenceTrack,
)
from roomkit.voice.base import AudioChunk

logger = logging.getLogger("roomkit.conference.livekit")


@dataclass(frozen=True)
class ConferenceEmissions:
    """The backend's callback fanout, handed to a session as plain functions.

    A session emits without holding the backend, so what it needs from the
    backend is stated here rather than discovered by reaching into it.
    """

    participant_joined: Callable[[str, ConferenceParticipant], Awaitable[None]]
    participant_left: Callable[[str, ConferenceParticipant], Awaitable[None]]
    track_published: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_unpublished: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_muted: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_unmuted: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_audio: AudioSink
    track_video: VideoSink
    active_speaker_changed: Callable[[str, str], Awaitable[None]]
    connection_quality: Callable[[str, str, str], Awaitable[None]]
    bot_session_ended: Callable[[BotSession, str], Awaitable[None]]


class LiveKitBotSession:
    """One bot participant in one LiveKit room."""

    def __init__(
        self,
        *,
        rtc: Any,
        session: BotSession,
        config: Any,
        emissions: ConferenceEmissions,
        evict: Callable[[], Awaitable[None]],
    ) -> None:
        self._rtc = rtc
        self.session = session
        self._room: Any = rtc.Room()
        # A lifecycle event the bridge cannot hold ends the session rather
        # than being lost (see `_overflow`).
        self._bridge = EventBridge(session.room_id)
        self._pumps = TrackPumps(
            rtc=rtc,
            room_id=session.room_id,
            audio_sink=emissions.track_audio,
            video_sink=emissions.track_video,
            config=config,
        )
        self._voice = BotVoiceTrack(
            rtc=rtc,
            room=self._room,
            identity=session.identity,
            room_id=session.room_id,
            queue_ms=config.publish_queue_ms,
        )
        self._departure = SessionDeparture(
            room=self._room,
            session=session,
            voice=self._voice,
            pumps=self._pumps,
            bridge=self._bridge,
            report_end=emissions.bot_session_ended,
            evict=evict,
        )
        self._view = RoomView(
            rtc=rtc,
            room_id=session.room_id,
            emissions=emissions,
            pumps=self._pumps,
            enqueue=self._put,
            enqueue_state=self._put_state,
        )

    @property
    def room_id(self) -> str:
        return self.session.room_id

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self, url: str, token: str) -> None:
        """Join the room and report what is already there.

        Handlers and the consumer task are in place *before* the connect, so an
        event that lands during it is queued rather than dropped. The catch-up
        that follows is enqueued through the same queue for the same reason:
        ordering is what the queue is for, and a catch-up emitted directly would
        be the one thing that jumps it.
        """
        self._register_handlers()
        self._bridge.start()
        options = self._rtc.RoomOptions(auto_subscribe=False)
        await self._room.connect(url, token, options)
        local = self._room.local_participant
        self.session.id = local.sid or self.session.id
        if (joined_at := local.joined_at) is not None:
            self.session.joined_at = joined_at
        self._view.catch_up(self._room)

    async def leave(self) -> None:
        """Take the bot out of the room; see :meth:`SessionDeparture.leave`."""
        await self._departure.leave()

    def _register_handlers(self) -> None:
        """Route each SDK room event to the part that handles it."""
        view = self._view
        handlers: dict[str, Callable[..., None]] = {
            "participant_connected": view.on_participant_connected,
            "participant_disconnected": view.on_participant_disconnected,
            "track_published": view.on_track_published,
            "track_unpublished": view.on_track_unpublished,
            "track_subscribed": view.on_track_subscribed,
            "track_unsubscribed": view.on_track_unsubscribed,
            "track_muted": view.on_track_muted,
            "track_unmuted": view.on_track_unmuted,
            "active_speakers_changed": view.on_active_speakers_changed,
            "connection_quality_changed": view.on_connection_quality_changed,
            "disconnected": self._departure.dropped,
        }
        for event, handler in handlers.items():
            self._room.on(event, handler)

    # -------------------------------------------------------------------------
    # Subscription — the framework's set is the authoritative one
    # -------------------------------------------------------------------------

    async def subscribe(self, track_id: str) -> None:
        """Ask LiveKit to start forwarding a track to the bot; see :meth:`RoomView.subscribe`."""
        await self._view.subscribe(track_id)

    async def unsubscribe(self, track_id: str) -> None:
        await self._view.unsubscribe(track_id)

    def publisher_identity(self, track_id: str) -> str | None:
        """Who publishes a track; see :meth:`RoomView.publisher_identity`."""
        return self._view.publisher_identity(track_id)

    def tracks(self) -> Mapping[str, ConferenceTrack]:
        return self._view.tracks()

    # -------------------------------------------------------------------------
    # Publishing the AI's voice
    # -------------------------------------------------------------------------

    async def publish(self, chunk: AudioChunk) -> None:
        """Put one chunk of the AI's speech on the bot's track.

        The track keeps the publishing contract; what the session adds is that
        there has to be a session at all — a chunk arriving after the bot left
        has no track to go on, and the SDK's own error for that would say
        nothing about why.
        """
        if not self._departure.admitting:
            raise RuntimeError(
                f"the bot has left room {self.room_id!r}, so there is no track to publish on"
            )
        try:
            await self._voice.publish(chunk)
        except VoicePublicationError as exc:
            self._voice_refused(exc)
            raise

    def _voice_refused(self, failure: VoicePublicationError) -> None:
        """End the session whose voice LiveKit did not publish.

        The SDK leaves its room listener waiting on the refused publication, so
        the session stops receiving arrivals and publications: a view that can
        no longer be trusted, which RFC 12.10.3 ends rather than keeps silently.
        The re-join that follows brings a session with a working listener. A
        session already leaving has nothing left to end.
        """
        if not self._departure.admitting:
            return
        logger.error(
            "%s. The bot session is being ended and re-joined, because the SDK stops "
            "delivering the room's events after a refused publication. If the SFU refused "
            "the source, check that the bot's grants include publish_audio: explicit "
            "bot_grants are taken as given",
            failure,
        )
        self._departure.end_unhealthy(f"voice publication failed: {failure}")

    def stop_playback(self) -> None:
        """Drop the queued, unplayed audio of the bot's track.

        A session that has left is a no-op rather than the error
        :meth:`publish` raises: a chunk for a departed session is a write with
        no track to land on, while the silence a stop asks for is already true
        of one (RFC section 12.10.3).
        """
        if not self._departure.admitting:
            return
        self._voice.discard_queued()

    # -------------------------------------------------------------------------
    # The gate every queued event passes
    # -------------------------------------------------------------------------

    def _put(self, emit: Callable[..., Awaitable[None]], *args: Any) -> None:
        """Queue a lifecycle event while the session admits, ending it if the bridge is full."""
        if not self._departure.admitting:
            return
        if not self._bridge.put(emit, *args):
            self._overflow()

    def _put_state(self, key: Any, emit: Callable[..., Awaitable[None]], *args: Any) -> None:
        """Queue a state event while the session admits, ending it if the bridge is full."""
        if not self._departure.admitting:
            return
        if not self._bridge.put_state(key, emit, *args):
            self._overflow()

    def _overflow(self) -> None:
        """The consumer has fallen unrecoverably behind: end the session.

        Evicting a lifecycle event would be worse than the memory it saves —
        an arrival or a publication lost here is a roster that lies and a
        track never transcribed, with nothing anywhere to say so, against the
        lifecycle MUSTs of RFC 12.10.4. A session whose view of the
        conference can no longer be trusted has one honest exit, and the
        contract already names it: the session ends, the loss is reported
        through ``bot_session_ended``, and the channel's re-join builds a
        fresh session whose catch-up announces the *current* truth.
        """
        logger.error(
            "The LiveKit event bridge for room %s overflowed at %d queued event(s): the "
            "framework's fanout is not keeping up with the conference, and going on would "
            "mean losing lifecycle facts silently. The bot session is being ended and "
            "re-joined for a consistent view",
            self.room_id,
            MAX_QUEUED_EVENTS,
        )
        self._departure.end_unhealthy(
            f"event queue overflow at {MAX_QUEUED_EVENTS} events; the session's view "
            "of the conference can no longer be trusted"
        )
