"""What the bot knows of its room: who is there, what they publish, what it wants.

LiveKit reports the room through synchronous callbacks; this keeps the books
they feed — the participants already announced, the tracks and their
publications, the tracks the framework asked to receive, the dominant speaker
— and turns each change into the emission the framework expects, queued
through the session. It decides nothing about admission or overflow: the
enqueue callables it is given are the session's, and a refusal there is the
session's to act on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from roomkit.conference._livekit_emissions import ConferenceEmissions
from roomkit.conference._livekit_mapping import (
    participant_record,
    quality_label,
    track_record,
)
from roomkit.conference._livekit_media import TrackPumps
from roomkit.conference.models import ConferenceParticipant, ConferenceTrack

logger = logging.getLogger("roomkit.conference.livekit")

Enqueue = Callable[..., None]
"""``(emit, *args)``: queue a lifecycle event through the session."""

EnqueueState = Callable[..., None]
"""``(key, emit, *args)``: queue a state event through the session."""


class RoomView:
    """Keep the books of one LiveKit room and announce what changes in it."""

    def __init__(
        self,
        *,
        rtc: Any,
        room_id: str,
        emissions: ConferenceEmissions,
        pumps: TrackPumps,
        enqueue: Enqueue,
        enqueue_state: EnqueueState,
    ) -> None:
        self._rtc = rtc
        self._room_id = room_id
        self._emissions = emissions
        self._pumps = pumps
        self._enqueue = enqueue
        self._enqueue_state = enqueue_state
        self._announced: set[str] = set()
        self._tracks: dict[str, ConferenceTrack] = {}
        self._publications: dict[str, Any] = {}
        self._wanted: set[str] = set()
        self._dominant_speaker: str | None = None

    @property
    def room_id(self) -> str:
        return self._room_id

    def catch_up(self, room: Any) -> None:
        """Announce the participants that were here before the bot was.

        ``participant_connected`` fires for arrivals, and everyone already in
        the room is not an arrival — so without this a bot joining a meeting in
        progress sees an empty conference and subscribes to nothing. This is the
        first half of the race the mock cannot stage: the second half is a
        publisher that leaves between its track being announced and the
        subscription reaching the server, which :meth:`subscribe` handles.
        """
        for participant in room.remote_participants.values():
            self.on_participant_connected(participant)
            for publication in participant.track_publications.values():
                self.on_track_published(publication, participant)

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
        somewhere, and this view watched it be published.
        """
        track = self._tracks.get(track_id)
        return None if track is None else track.participant_id

    def tracks(self) -> Mapping[str, ConferenceTrack]:
        return self._tracks

    def on_participant_connected(self, participant: Any) -> None:
        """Announce an arrival, once.

        The catch-up and the arrival event overlap: a participant that connects
        while the session's ``connect()`` is still returning is both reported as an arrival
        and already in the room the catch-up walks. Announcing it twice would
        have the roster resolve one person's identity twice and open their lane
        against a second announcement of the same track.
        """
        if participant.identity in self._announced:
            return
        self._announced.add(participant.identity)
        self._enqueue(
            self._emissions.participant_joined, self.room_id, self._participant(participant)
        )

    def on_participant_disconnected(self, participant: Any) -> None:
        self._announced.discard(participant.identity)
        self._forget_tracks_of(participant.identity)
        self._enqueue(
            self._emissions.participant_left, self.room_id, self._participant(participant)
        )

    def on_track_published(self, publication: Any, participant: Any) -> None:
        """Announce a published track, once — same overlap as an arrival."""
        if publication.sid in self._publications:
            return
        record = self._record_track(publication, participant)
        if record is None:
            return
        self._publications[record.id] = publication
        self._enqueue(self._emissions.track_published, self.room_id, record)

    def on_track_unpublished(self, publication: Any, participant: Any) -> None:
        track_id = publication.sid
        self._publications.pop(track_id, None)
        self._wanted.discard(track_id)
        record = self._tracks.pop(track_id, None)
        if record is None:
            return
        self._enqueue(self._emissions.track_unpublished, self.room_id, record)

    def on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
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

    def on_track_unsubscribed(self, track: Any, publication: Any, participant: Any) -> None:
        self._pumps.cancel(publication.sid)

    def on_track_muted(self, participant: Any, publication: Any) -> None:
        self._set_muted(publication.sid, True)

    def on_track_unmuted(self, participant: Any, publication: Any) -> None:
        self._set_muted(publication.sid, False)

    def on_active_speakers_changed(self, speakers: list[Any]) -> None:
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
        self._enqueue_state(
            "speaker", self._emissions.active_speaker_changed, self.room_id, dominant
        )

    def on_connection_quality_changed(self, participant: Any, quality: Any) -> None:
        label = quality_label(getattr(quality, "name", str(quality)))
        if label is None:
            return
        self._enqueue_state(
            ("quality", participant.identity),
            self._emissions.connection_quality,
            self.room_id,
            participant.identity,
            label,
        )

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
        self._enqueue_state(("mute", track_id), emit, self.room_id, record)
