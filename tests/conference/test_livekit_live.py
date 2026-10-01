"""LiveKit backend against a real SFU — the things a mock cannot prove.

Skipped unless a server is named. See the module docstring of
``roomkit.conference.livekit`` for the ``livekit.yaml`` this needs and why::

    docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\
        -e LIVEKIT_CONFIG="$(cat livekit.yaml)" \\
        livekit/livekit-server --dev --bind 0.0.0.0

    ROOMKIT_LIVEKIT_URL=ws://127.0.0.1:7880 \\
    ROOMKIT_LIVEKIT_API_KEY=devkey ROOMKIT_LIVEKIT_API_SECRET=secret \\
        uv run pytest tests/conference/test_livekit_live.py -v

Every test here exists because the mock backend cannot stage it. The mock hands
the lane frames the test itself built, at a rate it chose, through a
subscription that took effect instantly, against grants nothing enforced. Here a
second real connection stands in for the human — publishing 48 kHz **stereo**,
which is what a browser actually sends and what RoomKit's own resampling path
had never seen — and the SFU decides what reaches whom.

What is still not proven here is that a human hears the bot: test 9 shows an
``AudioChunk`` reaching another participant's decoder, which is as far as an
automated test can carry it. A person in a room with a microphone is the rest.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
from array import array
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from roomkit.conference.livekit import (
    LiveKitConferenceBackend,
    LiveKitConfig,
    VoicePublicationError,
)
from roomkit.conference.models import (
    ConferenceGrants,
    ConferenceParticipant,
    ConferenceTrack,
    TrackKind,
)
from roomkit.core.exceptions import ConferenceCapabilityError
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

rtc = pytest.importorskip("livekit.rtc")

URL = os.getenv("ROOMKIT_LIVEKIT_URL")
API_KEY = os.getenv("ROOMKIT_LIVEKIT_API_KEY")
API_SECRET = os.getenv("ROOMKIT_LIVEKIT_API_SECRET")

pytestmark = pytest.mark.skipif(
    not (URL and API_KEY and API_SECRET),
    reason=(
        "needs a LiveKit server: set ROOMKIT_LIVEKIT_URL, ROOMKIT_LIVEKIT_API_KEY "
        "and ROOMKIT_LIVEKIT_API_SECRET"
    ),
)

PUBLISH_RATE = 48_000
PUBLISH_CHANNELS = 2
"""Stereo on purpose: it is what a browser sends, and it is the shape RoomKit's
lane had only ever been handed by a mock that made it up."""

FRAME_MS = 10
TIMEOUT_S = 10.0
QUIET_S = 1.5
"""Long enough that an unsubscribed track would have delivered something."""


async def wait_for(predicate: Callable[[], bool], *, timeout: float = TIMEOUT_S) -> None:
    """Wait for a condition the SFU will bring about, or fail saying it did not."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"condition still false after {timeout}s")


async def poll[T](
    read: Callable[[], Awaitable[T]],
    done: Callable[[T], bool],
    missing: str,
    *,
    timeout: float = TIMEOUT_S,
) -> T:
    """Re-read what the server reports until ``done`` holds, or fail saying
    what is still ``missing``.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = await read()
        if done(value):
            return value
        await asyncio.sleep(0.05)
    raise AssertionError(f"{missing} after {timeout}s")


async def joined_participants(
    backend: LiveKitConferenceBackend, room_id: str, *, timeout: float = TIMEOUT_S
) -> list[ConferenceParticipant]:
    """Poll the server until it lists someone, and hand back what it listed.

    Reads the control plane rather than the bot's callbacks, so a test can ask
    what the server knows about a participant without a bot in the room to hear
    it announced.
    """
    return await poll(
        lambda: backend.list_participants(room_id),
        bool,
        f"nobody listed in room {room_id}",
        timeout=timeout,
    )


async def listed_tracks(
    backend: LiveKitConferenceBackend, room_id: str, wanted: set[str]
) -> dict[str, ConferenceTrack]:
    """Poll the server until it lists every track in ``wanted``.

    A confirmed publication is not listed at once: the server reports the
    track only after it negotiated the media, a moment later.
    """

    async def tracks() -> dict[str, ConferenceTrack]:
        participants = await backend.list_participants(room_id)
        return {track.id: track for p in participants for track in p.tracks}

    return await poll(
        tracks,
        lambda listed: wanted <= listed.keys(),
        f"tracks {sorted(wanted)} not listed in room {room_id}",
    )


def tone_frame(step: int) -> Any:
    """One 10 ms stereo frame of a 440 Hz tone, loud enough for a VAD to notice."""
    samples = array("h")
    per_channel = PUBLISH_RATE * FRAME_MS // 1000
    for index in range(per_channel):
        position = step * per_channel + index
        value = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * position / PUBLISH_RATE))
        samples.extend([value] * PUBLISH_CHANNELS)
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=PUBLISH_RATE,
        num_channels=PUBLISH_CHANNELS,
        samples_per_channel=per_channel,
    )


class Participant:
    """A second real connection, standing in for a person in the room.

    Its own ``rtc.Room``, its own token minted through the backend under test, so
    what it publishes travels the same path a browser's audio would.
    """

    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.room: Any = rtc.Room()
        self.received: list[Any] = []
        self._tone: asyncio.Task[None] | None = None
        self._sink: asyncio.Task[None] | None = None
        self._source: Any | None = None

    async def join(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        *,
        attributes: dict[str, str] | None = None,
        grants: ConferenceGrants | None = None,
    ) -> None:
        access = await backend.mint_access(
            room_id, self.identity, grants or ConferenceGrants(), attributes=attributes
        )
        self.room.on("track_subscribed", self._on_track_subscribed)
        await self.room.connect(access.url, access.token, rtc.RoomOptions(auto_subscribe=True))

    async def publish_tone(self) -> str:
        """Start publishing, and report the track sid the SFU gave it."""
        self._source = rtc.AudioSource(PUBLISH_RATE, PUBLISH_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track(f"{self.identity}-mic", self._source)
        publication = await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        self._tone = asyncio.create_task(self._speak())
        return publication.sid

    async def publish_screen_share(self, *, with_sound: bool) -> list[str]:
        """Publish what a browser's screen share publishes: the picture, and
        the sound on a track of its own when the shared tab has some.

        Nothing is captured into either source — the SFU decides on the
        publication, not on the media — so a refusal surfaces here, as the
        ``PublishTrackError`` the SDK raises once the server never confirms.
        """
        published = await publish_track(
            self.room,
            rtc.LocalVideoTrack.create_video_track("screen", rtc.VideoSource(64, 64)),
            rtc.TrackSource.SOURCE_SCREENSHARE,
        )
        sids = [published]
        if with_sound:
            sids.append(await publish_screen_share_audio(self.room))
        return sids

    async def _speak(self) -> None:
        step = 0
        while self._source is not None:
            await self._source.capture_frame(tone_frame(step))
            step += 1

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        self._sink = asyncio.create_task(self._listen(track))

    async def _listen(self, track: Any) -> None:
        stream = rtc.AudioStream.from_track(track=track, sample_rate=48_000, num_channels=1)
        try:
            async for event in stream:
                self.received.append(event.frame)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def leave(self) -> None:
        for task in (self._tone, self._sink):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tone = self._sink = None
        source, self._source = self._source, None
        if source is not None:
            with contextlib.suppress(Exception):
                await source.aclose()
        # Bounded: after a publication the server refused, the SDK's
        # disconnect waits on a room listener the refusal left stuck
        # (livekit-rtc 1.1.20), so the listener is released past the bound.
        with contextlib.suppress(Exception):
            disconnecting = asyncio.ensure_future(self.room.disconnect())
            done, _ = await asyncio.wait({disconnecting}, timeout=2.0)
            if not done:
                self.room._task.cancel()
                await asyncio.wait({disconnecting}, timeout=2.0)


async def publish_track(room: Any, track: Any, source: Any) -> str:
    publication = await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=source)
    )
    return publication.sid


async def publish_screen_share_audio(room: Any) -> str:
    """Publish a silent track as the sound of a screen share."""
    track = rtc.LocalAudioTrack.create_audio_track(
        "screen-audio", rtc.AudioSource(PUBLISH_RATE, PUBLISH_CHANNELS)
    )
    return await publish_track(room, track, rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO)


class Observed:
    """What the backend told the framework, in the order it said it."""

    def __init__(self, backend: LiveKitConferenceBackend) -> None:
        self.joined: list[ConferenceParticipant] = []
        self.left: list[ConferenceParticipant] = []
        self.published: list[ConferenceTrack] = []
        self.unpublished: list[ConferenceTrack] = []
        self.audio: list[tuple[ConferenceTrack, AudioFrame]] = []
        self.speakers: list[str] = []
        self.order: list[str] = []
        backend.on_participant_joined(self._joined)
        backend.on_participant_left(self._left)
        backend.on_track_published(self._published)
        backend.on_track_unpublished(self._unpublished)
        backend.on_track_audio(self._audio)
        backend.on_active_speaker_changed(self._speaker)

    def _joined(self, room_id: str, participant: ConferenceParticipant) -> None:
        self.joined.append(participant)
        self.order.append(f"joined:{participant.participant_id}")

    def _left(self, room_id: str, participant: ConferenceParticipant) -> None:
        self.left.append(participant)

    def _published(self, room_id: str, track: ConferenceTrack) -> None:
        self.published.append(track)
        self.order.append(f"published:{track.participant_id}")

    def _unpublished(self, room_id: str, track: ConferenceTrack) -> None:
        self.unpublished.append(track)

    def _audio(self, track: ConferenceTrack, frame: AudioFrame) -> None:
        self.audio.append((track, frame))

    def _speaker(self, room_id: str, participant_id: str) -> None:
        self.speakers.append(participant_id)

    def audio_for(self, track_id: str) -> list[AudioFrame]:
        return [frame for track, frame in self.audio if track.id == track_id]


def make_backend(**overrides: Any) -> LiveKitConferenceBackend:
    settings: dict[str, Any] = {
        "url": URL,
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "audio_channels": PUBLISH_CHANNELS,
    }
    settings.update(overrides)
    return LiveKitConferenceBackend(LiveKitConfig(**settings))


@pytest.fixture
async def backend() -> AsyncIterator[LiveKitConferenceBackend]:
    instance = make_backend()
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
async def room_id(backend: LiveKitConferenceBackend) -> AsyncIterator[str]:
    identifier = f"rmk-live-{uuid4().hex[:10]}"
    await backend.ensure_room(identifier)
    try:
        yield identifier
    finally:
        with contextlib.suppress(Exception):
            await backend.close_room(identifier)


@pytest.fixture
def observed(backend: LiveKitConferenceBackend) -> Observed:
    return Observed(backend)


@pytest.fixture
async def alice() -> AsyncIterator[Participant]:
    person = Participant("p-alice")
    try:
        yield person
    finally:
        await person.leave()


class TestControlPlane:
    async def test_a_created_room_is_empty_and_can_be_closed(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        assert await backend.list_participants(room_id) == []

        await backend.close_room(room_id)

        assert await backend.list_participants(room_id) == []

    async def test_creating_a_room_twice_is_idempotent(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        await backend.ensure_room(room_id)
        await backend.ensure_room(room_id, {"tenant": "acme"})

        assert await backend.list_participants(room_id) == []


class TestBotSession:
    async def test_the_bot_joins_and_the_server_sees_it(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        listed = await backend.list_participants(room_id)

        assert [p.participant_id for p in listed] == ["roomkit"]
        assert bot.identity == "roomkit"

    async def test_the_session_reports_when_the_sfu_says_it_joined(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """RFC section 12.10.2 asks a backend holding a better figure than
        construction time to use it, and the value must be aware. Checked against
        the wall clock because the unit LiveKit reports it in is exactly the sort
        of thing that only shows up against a real server — a millisecond field
        read as seconds lands in 1970, and the error would surface in a teardown
        as a missing announcement.
        """
        before = datetime.now(UTC)

        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        assert bot.joined_at.tzinfo is not None
        assert abs((bot.joined_at - before).total_seconds()) < 60

    async def test_leaving_takes_the_bot_out_of_the_room(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """Teardown observable from outside: the bot is gone from the list a
        human's client reads, not merely marked gone in our own state.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await wait_for(lambda: True)

        await backend.leave(bot)

        await wait_for(lambda: True, timeout=1.0)
        remaining = await backend.list_participants(room_id)
        assert "roomkit" not in [p.participant_id for p in remaining]

    async def test_a_join_the_server_refuses_leaves_nothing_running(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """A failed join is never registered, so the channel gets no handle to
        close it with — whatever it started has to be torn down on the way out
        or it runs for the life of the process.
        """
        before = len(asyncio.all_tasks())
        broken = make_backend(url="ws://127.0.0.1:1")

        with pytest.raises(Exception, match=r".*"):
            await broken.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        await asyncio.sleep(0.3)
        assert len(asyncio.all_tasks()) <= before
        await broken.close()

    async def test_an_observer_bot_is_hidden_from_the_room(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """``hidden`` is a grant the SFU enforces, and the point of asking it to."""
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.observer())
        await alice.join(backend, room_id)

        await wait_for(lambda: len(alice.room.remote_participants) >= 0, timeout=2.0)
        await asyncio.sleep(QUIET_S)

        assert "roomkit" not in alice.room.remote_participants

    async def test_a_revealed_bot_appears_to_connected_clients(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """The in-place reveal RFC §12.10.4 leans on: removing ``hidden``
        through ``update_bot_grants()`` makes the SFU announce the session to
        the clients already connected — no re-join needed. (The reverse is
        not true, which is why concealment replaces the session instead.)
        """
        appeared: list[str] = []
        alice.room.on(
            "participant_connected",
            lambda participant: appeared.append(participant.identity),
        )
        await alice.join(backend, room_id)
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.observer())
        await asyncio.sleep(QUIET_S)
        assert "roomkit" not in alice.room.remote_participants
        assert "roomkit" not in appeared

        await backend.update_bot_grants(bot, ConferenceGrants.for_bot(listens=True))

        await wait_for(lambda: "roomkit" in alice.room.remote_participants)
        assert "roomkit" in appeared


class TestMintedAttributes:
    """A credential carries more than an identity, and the server proves it.

    The mock can be told to surface what a mint remembered; only a real SFU
    shows that the claim in the token *becomes* the attribute map the server
    reports back, which is the whole reason the field is worth having
    (RFC §12.10.3).
    """

    async def test_the_server_reports_what_the_credential_carried(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        await alice.join(backend, room_id, attributes={"app.user": "user-42"})

        await wait_for(lambda: bool(observed.joined))
        assert observed.joined[0].metadata["app.user"] == "user-42"

    async def test_a_mint_that_carried_nothing_adds_nothing(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        alice: Participant,
    ) -> None:
        await alice.join(backend, room_id)

        listed = await joined_participants(backend, room_id)

        assert not [key for key in listed[0].metadata if not key.startswith("livekit.")]

    async def test_the_server_vouches_for_none_of_it(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        alice: Participant,
    ) -> None:
        """It rode a token, and a token is not something the SFU established.

        LiveKit reports it beside a dial-in's ``sip.*`` in one flat map with
        nothing in the shape to tell them apart, so provenance is decided on
        the participant kind — and a browser's kind asserts nothing.
        """
        await alice.join(backend, room_id, attributes={"app.user": "user-42"})

        listed = await joined_participants(backend, room_id)

        assert listed[0].metadata["app.user"] == "user-42"
        assert listed[0].asserted_metadata is not None
        assert "app.user" not in listed[0].asserted_metadata


class TestPresenceAndTracks:
    async def test_a_participant_and_its_track_are_announced_in_order(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """A track arriving before its publisher would hand the roster a lane to
        open for someone it has never heard of. The serialized bridge is what
        keeps the order, and only a real SFU sends the two close enough together
        to test it.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        await alice.publish_tone()

        await wait_for(lambda: bool(observed.published))

        assert observed.order.index("joined:p-alice") < observed.order.index("published:p-alice")
        assert observed.published[0].kind is TrackKind.AUDIO
        assert observed.published[0].participant_id == "p-alice"

    async def test_a_bot_that_arrives_late_still_sees_who_is_there(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """LiveKit announces arrivals, and everyone already in the room is not an
        arrival — so without the catch-up a bot joining a meeting in progress
        subscribes to nothing at all.
        """
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()

        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        await wait_for(lambda: bool(observed.published))
        assert [p.participant_id for p in observed.joined] == ["p-alice"]
        assert [t.id for t in observed.published] == [track_id]

    async def test_a_participant_leaving_is_announced(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        await wait_for(lambda: bool(observed.joined))

        await alice.leave()

        await wait_for(lambda: bool(observed.left))
        assert observed.left[0].participant_id == "p-alice"

    async def test_a_participant_present_at_the_join_is_announced_once(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The catch-up and the arrival event overlap, and only a real server
        puts a participant in both at once. Announced twice, the roster would
        resolve one person's identity twice and open their lane against a second
        announcement of the same track.
        """
        await alice.join(backend, room_id)
        await alice.publish_tone()

        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await wait_for(lambda: bool(observed.published))
        await asyncio.sleep(QUIET_S)

        assert [p.participant_id for p in observed.joined] == ["p-alice"]
        assert len(observed.published) == 1

    async def test_a_departure_takes_its_tracks_off_the_books(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """A stale entry would answer for a track that no longer exists, and
        send a moderation call after somebody who has left.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await alice.leave()
        await wait_for(lambda: bool(observed.left))

        with pytest.raises(ValueError, match="nobody to moderate"):
            await backend.mute_track(room_id, track_id)


class TestSubscription:
    async def test_no_frames_arrive_before_the_framework_asks(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The framework's subscription set is the authoritative one, and the bot
        joins with auto-subscription off. Against a mock this is bookkeeping;
        here it is the SFU declining to forward.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await asyncio.sleep(QUIET_S)

        assert observed.audio == []

    async def test_frames_arrive_once_it_does(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await backend.subscribe_track(bot, track_id)

        await wait_for(lambda: len(observed.audio_for(track_id)) > 5)

    async def test_frames_declare_the_format_they_arrive_in(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The reason this card exists. 48 kHz stereo, declared on every frame
        and not normalised by the transport — the lane's resampler finally meets
        audio it did not manufacture.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))
        await backend.subscribe_track(bot, track_id)

        await wait_for(lambda: len(observed.audio_for(track_id)) > 10)

        frames = observed.audio_for(track_id)
        assert {f.sample_rate for f in frames} == {PUBLISH_RATE}
        assert {f.channels for f in frames} == {PUBLISH_CHANNELS}
        assert {f.sample_width for f in frames} == {2}
        assert all(f.data for f in frames)
        assert all(len(f.data) % (f.sample_width * f.channels) == 0 for f in frames), (
            "a frame that is not whole samples would shift every one after it"
        )

    async def test_frame_timestamps_advance_with_the_audio(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))
        await backend.subscribe_track(bot, track_id)
        await wait_for(lambda: len(observed.audio_for(track_id)) > 10)

        stamps = [f.timestamp_ms for f in observed.audio_for(track_id)]

        assert stamps[0] == 0
        assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))

    async def test_unsubscribing_stops_the_frames(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))
        await backend.subscribe_track(bot, track_id)
        await wait_for(lambda: len(observed.audio_for(track_id)) > 5)

        await backend.unsubscribe_track(bot, track_id)
        await asyncio.sleep(0.3)
        settled = len(observed.audio_for(track_id))
        await asyncio.sleep(QUIET_S)

        assert len(observed.audio_for(track_id)) == settled

    async def test_subscribing_a_track_whose_publisher_left_does_not_raise(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The race the card names: ``subscribe_track`` is genuinely asynchronous,
        and the publisher can already be gone. Ordinary in a conference, so it
        must not be an error the channel has to handle.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await alice.leave()
        await wait_for(lambda: bool(observed.unpublished) or bool(observed.left))
        await backend.subscribe_track(bot, track_id)

        await asyncio.sleep(QUIET_S)
        assert observed.audio_for(track_id) == []

    async def test_a_bot_denied_subscription_receives_nothing(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """Grant semantics enforced by the SFU rather than by us: a speak-only
        bot asks for no subscription right, and asking anyway gets nothing.
        """
        observed = Observed(backend)
        bot = await backend.join_as_bot(
            room_id, "roomkit", ConferenceGrants.for_bot(speaks=True, listens=False)
        )
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await backend.subscribe_track(bot, track_id)
        await asyncio.sleep(QUIET_S)

        assert observed.audio_for(track_id) == []


class TestPublishing:
    async def test_the_bots_voice_reaches_another_participants_decoder(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """As close to "audible" as an automated test reaches: PCM handed to
        ``publish_audio`` comes out of a *different* connection's Opus decoder.
        A person with a microphone is what is left, and it is not this test's.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot(speaks=True))
        await alice.join(backend, room_id)
        await wait_for(lambda: "roomkit" in alice.room.remote_participants)

        for step in range(60):
            await backend.publish_audio(
                bot,
                AudioChunk(
                    data=_pcm_mono(step),
                    sample_rate=PUBLISH_RATE,
                    channels=1,
                    is_final=step == 59,
                ),
            )

        await wait_for(lambda: len(alice.received) > 5)
        assert sum(len(bytes(frame.data)) for frame in alice.received) > 0

    async def test_a_chunk_in_another_format_is_refused_mid_stream(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """An ``rtc.AudioSource`` is fixed once published; republishing the track
        to follow a format change would drop the bot's voice out of the
        conference for as long as renegotiation takes.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot(speaks=True))
        await backend.publish_audio(
            bot, AudioChunk(data=_pcm_mono(0), sample_rate=PUBLISH_RATE, channels=1)
        )

        with pytest.raises(ValueError, match="format is fixed"):
            await backend.publish_audio(
                bot, AudioChunk(data=_pcm_mono(1), sample_rate=16_000, channels=1)
            )

    async def test_publishing_after_leaving_is_refused(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot(speaks=True))
        session = backend._sessions[bot.id]
        await backend.leave(bot)

        with pytest.raises(RuntimeError, match="has left"):
            await session.publish(AudioChunk(data=_pcm_mono(0), sample_rate=PUBLISH_RATE))


class TestScreenShareAudio:
    """The sound of a screen share is a publish right of its own (RFC
    12.10.2), and the SFU is what enforces it: the grant either reaches the
    server on the carrier in question or the publication is refused.
    """

    async def test_a_participant_granted_it_shares_a_screen_with_its_sound(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        await alice.join(
            backend, room_id, grants=ConferenceGrants(publish_screen_share_audio=True)
        )

        picture, sound = await alice.publish_screen_share(with_sound=True)

        tracks = await listed_tracks(backend, room_id, {picture, sound})
        assert tracks[picture].kind is TrackKind.SCREEN_SHARE
        # Audio like any other once published — RoomKit's track model has no
        # screen-share audio kind; the source says where it came from.
        assert tracks[sound].kind is TrackKind.AUDIO
        assert tracks[sound].metadata["source"] == "SOURCE_SCREENSHARE_AUDIO"

    async def test_a_participant_minted_with_the_defaults_is_refused_it(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """A mint that does not name the right carries none, so the SFU never
        confirms the publication: it logs the refusal and the client gives up
        waiting.
        """
        await alice.join(backend, room_id)

        with pytest.raises(rtc.participant.PublishTrackError):
            await publish_screen_share_audio(alice.room)

    async def test_a_share_without_sound_needs_no_new_grant(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        await alice.join(backend, room_id)

        [picture] = await alice.publish_screen_share(with_sound=False)

        assert picture

    async def test_an_in_place_update_carries_it_to_a_connected_session(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """The other carrier: ``UpdateParticipant`` on a session already
        connected. The bot is the session this backend can re-permission, so
        it stands in — the grant, not who holds it, is under test.

        No refused control in this session: a refused publication leaves the
        SDK's room listener stuck, and the update's own publication would then
        never be confirmed. ``TestAStuckSdkDoesNotHoldTheBot`` refuses the
        same publication on a session that was not updated.
        """
        held = ConferenceGrants.for_bot(speaks=True)
        bot = await backend.join_as_bot(room_id, "roomkit", held)

        await backend.update_bot_grants(bot, replace(held, publish_screen_share_audio=True))

        session_room = backend._sessions[bot.id]._room
        assert await publish_screen_share_audio(session_room)


class TestAStuckSdkDoesNotHoldTheBot:
    """After a failed publication the SDK's room listener is stuck and its
    disconnect never returns (livekit-rtc 1.1.20). The bot must still get out,
    and a session that stopped hearing the room must end (RMK-350).
    """

    async def test_a_bot_whose_publication_was_refused_still_leaves(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot(speaks=True))
        with pytest.raises(rtc.participant.PublishTrackError):
            await publish_screen_share_audio(backend._sessions[bot.id]._room)

        room = backend._sessions[bot.id]._room
        await asyncio.wait_for(backend.leave(bot), timeout=TIMEOUT_S)

        assert bot.id not in backend._sessions
        assert await backend.list_participants(room_id) == []
        # The SDK's stuck listener is released, which is what frees the room's
        # FFI subscription; this reaches into the SDK, and that is the point.
        assert room._task.done()

    async def test_a_refused_voice_ends_the_session_and_says_why(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """A bot granted no microphone whose voice is published anyway: the
        explicit-grants case the channel accepts as given.
        """
        ended: list[str] = []

        async def _ended(session: Any, reason: str) -> None:
            ended.append(reason)

        backend.on_bot_session_ended(_ended)
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        with pytest.raises(VoicePublicationError):
            await backend.publish_audio(
                bot, AudioChunk(data=_pcm_mono(0), sample_rate=PUBLISH_RATE, channels=1)
            )

        await wait_for(lambda: bool(ended))
        assert ended[0].startswith("voice publication failed")
        assert bot.id not in backend._sessions
        assert await backend.list_participants(room_id) == []


class TestModeration:
    async def test_muting_a_participants_track_takes_effect(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await backend.mute_track(room_id, track_id)

        async def muted() -> bool:
            for participant in await backend.list_participants(room_id):
                for track in participant.tracks:
                    if track.id == track_id:
                        return track.muted
            return False

        deadline = asyncio.get_running_loop().time() + TIMEOUT_S
        while asyncio.get_running_loop().time() < deadline:
            if await muted():
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("the track was never reported muted")

    async def test_unmuting_is_refused_unless_the_server_allows_it(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """LiveKit needs ``room.enable_remote_unmute`` server-side, which is the
        asymmetry ``REMOTE_UNMUTE`` exists to surface.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        with pytest.raises(ConferenceCapabilityError):
            await backend.unmute_track(room_id, track_id)


def _pcm_mono(step: int) -> bytes:
    """10 ms of a 440 Hz mono tone, as the framework's TTS would hand it over."""
    samples = array("h")
    per_frame = PUBLISH_RATE * FRAME_MS // 1000
    for index in range(per_frame):
        samples.append(
            int(
                0.3
                * 32767
                * math.sin(2 * math.pi * 440 * (step * per_frame + index) / PUBLISH_RATE)
            )
        )
    return samples.tobytes()
