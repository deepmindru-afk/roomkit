"""The framework's single connection to one LiveKit room.

The session owns an ``rtc.Room`` and the state that hangs off it: which tracks
the framework asked for, which pumps are running, and the audio source the AI's
voice goes out on. The frame delivery itself is in ``_livekit_media``, which is
the one piece that reads none of this, and the ordered fanout of the room's
events is ``_livekit_bridge``. The backend keeps the control plane and one
session per conference.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from roomkit.conference._livekit_bridge import MAX_QUEUED_EVENTS, EventBridge
from roomkit.conference._livekit_departure import SessionDeparture
from roomkit.conference._livekit_mapping import (
    participant_record,
    quality_label,
    track_record,
)
from roomkit.conference._livekit_media import AudioSink, TrackPumps, VideoSink
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
        self._config = config
        self._emissions = emissions
        self._room: Any = rtc.Room()
        # A lifecycle event the bridge cannot hold ends the session rather
        # than being lost (see `_overflow`).
        self._bridge = EventBridge(session.room_id)
        self._announced: set[str] = set()
        self._tracks: dict[str, ConferenceTrack] = {}
        self._publications: dict[str, Any] = {}
        self._wanted: set[str] = set()
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
        self._dominant_speaker: str | None = None
        self._departure = SessionDeparture(
            room=self._room,
            session=session,
            voice=self._voice,
            pumps=self._pumps,
            bridge=self._bridge,
            report_end=emissions.bot_session_ended,
            evict=evict,
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
        self._catch_up()

    def _catch_up(self) -> None:
        """Announce the participants that were here before the bot was.

        ``participant_connected`` fires for arrivals, and everyone already in
        the room is not an arrival — so without this a bot joining a meeting in
        progress sees an empty conference and subscribes to nothing. This is the
        first half of the race the mock cannot stage: the second half is a
        publisher that leaves between its track being announced and the
        subscription reaching the server, which :meth:`subscribe` handles.
        """
        for participant in self._room.remote_participants.values():
            self._enqueue_participant_joined(participant)
            for publication in participant.track_publications.values():
                self._enqueue_track_published(publication, participant)

    # -------------------------------------------------------------------------
    # Subscription — the framework's set is the authoritative one
    # -------------------------------------------------------------------------

    async def leave(self) -> None:
        """Take the bot out of the room; see :meth:`SessionDeparture.leave`."""
        await self._departure.leave()

    async def subscribe(self, track_id: str) -> None:
        """Ask LiveKit to start forwarding a track to the bot.

        A track that is no longer published is recorded as wanted and nothing
        else: the publisher left between the announcement and this call, which
        is ordinary in a conference and not a failure the channel should have to
        handle. Raising would turn every such race into an error, and the
        publication is gone for good — a republish arrives under a new sid.
        """
        self._wanted.add(track_id)
        publication = self._publications.get(track_id)
        if publication is None:
            logger.debug(
                "Track %s in room %s is not published, so there is nothing to subscribe to yet",
                track_id,
                self.room_id,
            )
            return
        publication.set_subscribed(True)

    async def unsubscribe(self, track_id: str) -> None:
        self._wanted.discard(track_id)
        if (publication := self._publications.get(track_id)) is not None:
            publication.set_subscribed(False)
        await self._pumps.stop(track_id)

    def publisher_identity(self, track_id: str) -> str | None:
        """Who publishes a track, for the moderation calls that need it.

        LiveKit's mute API is keyed on the participant *and* the track, while the
        interface passes only the track — so the answer has to come from
        somewhere, and this session watched it be published.
        """
        track = self._tracks.get(track_id)
        return None if track is None else track.participant_id

    def tracks(self) -> Mapping[str, ConferenceTrack]:
        return self._tracks

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
    # Event bridge — sync handlers in, ordered emissions out
    # -------------------------------------------------------------------------

    def _register_handlers(self) -> None:
        room = self._room
        room.on("participant_connected", self._enqueue_participant_joined)
        room.on("participant_disconnected", self._on_participant_disconnected)
        room.on("track_published", self._enqueue_track_published)
        room.on("track_unpublished", self._on_track_unpublished)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("track_muted", self._on_track_muted)
        room.on("track_unmuted", self._on_track_unmuted)
        room.on("active_speakers_changed", self._on_active_speakers_changed)
        room.on("connection_quality_changed", self._on_connection_quality_changed)
        room.on("disconnected", self._departure.dropped)

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

    def _enqueue_participant_joined(self, participant: Any) -> None:
        """Announce an arrival, once.

        The catch-up and the arrival event overlap: a participant that connects
        while :meth:`connect` is still returning is both reported as an arrival
        and already in the room the catch-up walks. Announcing it twice would
        have the roster resolve one person's identity twice and open their lane
        against a second announcement of the same track.
        """
        if participant.identity in self._announced:
            return
        self._announced.add(participant.identity)
        self._put(self._emissions.participant_joined, self.room_id, self._participant(participant))

    def _on_participant_disconnected(self, participant: Any) -> None:
        self._announced.discard(participant.identity)
        self._forget_tracks_of(participant.identity)
        self._put(self._emissions.participant_left, self.room_id, self._participant(participant))

    def _forget_tracks_of(self, identity: str) -> None:
        """Drop what a departing participant published.

        Nothing is emitted for it: LiveKit reports the tracks unpublished on its
        own where it does, and inventing the event where it does not would close
        a lane the channel already closed on the departure. What this is for is
        the bookkeeping — a stale entry here would answer
        :meth:`publisher_identity` for a track that no longer exists, and send a
        moderation call after someone who has left.
        """
        for track_id, record in list(self._tracks.items()):
            if record.participant_id != identity:
                continue
            self._tracks.pop(track_id, None)
            self._publications.pop(track_id, None)
            self._wanted.discard(track_id)
            self._pumps.cancel(track_id)

    def _participant(self, participant: Any) -> ConferenceParticipant:
        return participant_record(
            identity=participant.identity,
            sid=participant.sid,
            kind_name=self._rtc.ParticipantKind.Name(participant.kind),
            name=participant.name or "",
            metadata=participant.metadata or "",
            attributes=participant.attributes or {},
            connected_at=participant.joined_at,
        )

    def _enqueue_track_published(self, publication: Any, participant: Any) -> None:
        """Announce a published track, once — same overlap as an arrival."""
        if publication.sid in self._publications:
            return
        record = self._record_track(publication, participant)
        if record is None:
            return
        self._publications[record.id] = publication
        self._put(self._emissions.track_published, self.room_id, record)

    def _record_track(self, publication: Any, participant: Any) -> ConferenceTrack | None:
        try:
            record = track_record(
                sid=publication.sid,
                room_id=self.room_id,
                participant_id=participant.identity,
                kind_name=self._rtc.TrackKind.Name(publication.kind),
                source_name=self._rtc.TrackSource.Name(publication.source),
                muted=publication.muted,
                name=publication.name or "",
                mime_type=publication.mime_type or "",
            )
        except ValueError:
            logger.warning(
                "Ignoring LiveKit track %s in room %s: RoomKit has no kind for it",
                publication.sid,
                self.room_id,
                exc_info=True,
            )
            return None
        self._tracks[record.id] = record
        return record

    def _on_track_unpublished(self, publication: Any, participant: Any) -> None:
        track_id = publication.sid
        self._publications.pop(track_id, None)
        self._wanted.discard(track_id)
        record = self._tracks.pop(track_id, None)
        if record is None:
            return
        self._put(self._emissions.track_unpublished, self.room_id, record)

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        """Start a pump, but only for a track the framework asked for.

        The bot joins with ``auto_subscribe`` off, so this should only ever fire
        behind a :meth:`subscribe`. Should is not must — a subscription the SDK
        arranged on its own would deliver frames the framework never requested,
        which RFC section 12.10.3 forbids — so it is undone here rather than
        trusted.
        """
        track_id = publication.sid
        if track_id not in self._wanted:
            logger.warning(
                "LiveKit subscribed the bot to track %s in room %s without being asked; "
                "undoing it",
                track_id,
                self.room_id,
            )
            publication.set_subscribed(False)
            return
        record = self._tracks.get(track_id) or self._record_track(publication, participant)
        if record is None:
            return
        self._pumps.start(record, track)

    def _on_track_unsubscribed(self, track: Any, publication: Any, participant: Any) -> None:
        self._pumps.cancel(publication.sid)

    def _on_track_muted(self, participant: Any, publication: Any) -> None:
        self._set_muted(publication.sid, True)

    def _on_track_unmuted(self, participant: Any, publication: Any) -> None:
        self._set_muted(publication.sid, False)

    def _set_muted(self, track_id: str, muted: bool) -> None:
        """Keep the record's mute flag true to the publisher's own state, and say so.

        The record is updated before the report goes out, which is the order
        the contract promises (RFC 12.10.3): a callback that re-reads
        ``ConferenceTrack.muted`` reads the state it was told about. A mute is
        a state, not a fact — only the current value matters, so a consumer
        that fell behind hears the newest transition per track rather than a
        replay of the toggling.
        """
        if (record := self._tracks.get(track_id)) is None:
            return
        record.muted = muted
        emit = self._emissions.track_muted if muted else self._emissions.track_unmuted
        self._put_state(("mute", track_id), emit, self.room_id, record)

    def _on_active_speakers_changed(self, speakers: list[Any]) -> None:
        """Report the dominant speaker, when it is a different one.

        LiveKit sends the whole active set, loudest first, and the interface
        carries one identity — so the loudest is the dominant one. An empty set
        means nobody is speaking, which the interface has no way to say, so it
        says nothing rather than naming a speaker who has stopped.
        """
        dominant = speakers[0].identity if speakers else None
        if dominant is None or dominant == self._dominant_speaker:
            self._dominant_speaker = dominant
            return
        self._dominant_speaker = dominant
        self._put_state("speaker", self._emissions.active_speaker_changed, self.room_id, dominant)

    def _on_connection_quality_changed(self, participant: Any, quality: Any) -> None:
        label = quality_label(getattr(quality, "name", str(quality)))
        if label is None:
            return
        self._put_state(
            ("quality", participant.identity),
            self._emissions.connection_quality,
            self.room_id,
            participant.identity,
            label,
        )
