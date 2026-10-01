"""Testing a conference against a backend that misbehaves.

A conference that only ever meets a working SFU is a conference whose failure
paths are untested. MockConferenceBackend can therefore be made to refuse calls,
to take time answering them, to publish tracks in formats that disagree with
each other, and to say which chunks of the bot's voice belonged to which
utterance.

Four levers, one per thing a real conference does that a happy-path mock does
not:

1. ``fail(method, error, times=)``  — the SFU refuses, or times out
2. ``backend.deliveries``           — how long each frame took to reach the
                                      subscribers, measured on one track while
                                      another track's recognizer is slow
3. ``MockTrackFormat``              — participants negotiate their own formats
4. ``backend.utterances``           — what the bot published, per utterance

(``delay(operation, seconds)`` slows a backend call or a callback emission the
same way ``fail()`` makes one raise; the scenarios below do not need it.)

The fifth scenario needs no lever at all: the storage is what is slow there, and
a recorder that blocks is a recorder written the only way the interface allows.

Run with:
    uv run python examples/conference_fault_injection.py
"""

from __future__ import annotations

import asyncio
import threading
import time

from roomkit import (
    ConferenceGrants,
    ConferenceRecordingConfig,
    ConferenceRecordingStarted,
    ConferenceTranscription,
    HookExecution,
    HookResult,
    HookTrigger,
    MockConferenceBackend,
    MockTrackFormat,
    RoomKit,
)
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.models import ConferenceTrack
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import AudioContent, EventSource, RoomEvent, TextContent
from roomkit.recorder.base import MediaRecordingHandle, RecordingTrack
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

ROOM = "board-meeting"

SPEECH_FRAMES = 15
"""300 ms — past the energy VAD's minimum for an utterance."""

SILENCE_FRAMES = 30
"""600 ms — past its end-of-speech threshold, so the utterance closes."""


class AISource(Channel):
    """Stands in for an AIChannel, so the bot has something to say."""

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.AI

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class SlowRecognizer(MockSTTProvider):
    """A recognizer that takes ``seconds`` over the first utterance it hears.

    Recognition is the realistic slow stage of a lane — a round trip to a
    remote STT. Only the first call is slow, so the track that speaks first is
    stuck in it while every other track carries on.
    """

    def __init__(self, *, seconds: float) -> None:
        super().__init__(transcripts=["(alice's words)", "(bob's words)"])
        self.seconds = seconds
        self.stuck = asyncio.Event()
        """Set once the slow recognition has begun."""

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        first = not self.calls
        result = await super().transcribe(audio, language=language)
        if first:
            self.stuck.set()
            await asyncio.sleep(self.seconds)
        return result


class SlowRecorder(MockMediaRecorder):
    """A recorder whose writes take time, the only way a recorder can.

    ``MediaRecorder`` is synchronous throughout, so "slow storage" means a call
    that blocks — which is why the channel makes it somewhere other than the
    thread its event loop runs on.
    """

    def __init__(self, *, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self.threads: set[int] = set()

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        self.threads.add(threading.get_ident())
        time.sleep(self.seconds)
        super().on_data(handle, track, data, timestamp_ms)


async def say(backend: MockConferenceBackend, track: ConferenceTrack) -> None:
    """One utterance's worth of frames, in the track's own format."""
    for _ in range(SPEECH_FRAMES):
        await backend.simulate_audio(track, backend.frame_for(track))
    for _ in range(SILENCE_FRAMES):
        await backend.simulate_audio(track, backend.frame_for(track, amplitude=0.0))


async def failures() -> None:
    """1. The SFU refuses."""
    print("1. A backend that refuses\n")
    backend = MockConferenceBackend()
    kit = RoomKit()
    channel = ConferenceChannel("conf", backend=backend)
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.ensure_participant(ROOM, "conf", "p-alice", display_name="Alice")

    # Minting admission is the one call whose failure reaches the integrator:
    # a credential nobody received is better than one handed out blind.
    backend.fail("mint_access", TimeoutError("SFU unreachable"), times=1)
    try:
        await channel.mint_access(ROOM, "p-alice", grants=ConferenceGrants())
    except TimeoutError as exc:
        print(f"   mint_access → {type(exc).__name__}: {exc}")

    # times=1 retired the fault, so the retry finds a working SFU.
    access = await channel.mint_access(ROOM, "p-alice", grants=ConferenceGrants())
    print(f"   retry      → {access.token}")

    # The attempt is recorded even though it failed: the request did go out.
    attempts = [call for call in backend.calls if call.method == "mint_access"]
    print(f"   backend saw {len(attempts)} mint attempts, not {len(attempts) - 1}\n")

    await kit.detach_channel(ROOM, "conf")


async def latency() -> None:
    """2. Recognition is slow on one track, and the other track does not wait.

    RFC §12.10.4 makes lane isolation checkable from outside: delay recognition
    on one track and measure frame delivery — and transcription — on another.
    """
    print("2. A slow recognizer on one track must not slow the others (RFC §12.10.4)\n")
    backend = MockConferenceBackend()
    stt = SlowRecognizer(seconds=1.0)
    kit = RoomKit()
    channel = ConferenceChannel("conf", backend=backend, stt=stt)
    kit.register_channel(channel)
    loop = asyncio.get_running_loop()
    heard: dict[str, float] = {}
    both_heard = asyncio.Event()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(payload: ConferenceTranscription, ctx: object) -> HookResult:
        heard[payload.participant_id] = loop.time()
        if len(heard) == 2:
            both_heard.set()
        return HookResult.allow()

    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    await backend.simulate_participant_joined(ROOM, "p-alice")
    await backend.simulate_participant_joined(ROOM, "p-bob")
    alice = await backend.simulate_track_published(ROOM, "p-alice")
    bob = await backend.simulate_track_published(ROOM, "p-bob")

    started = loop.time()
    await say(backend, alice)
    # Alice's lane is now inside its slow recognition. Bob speaks meanwhile.
    await asyncio.wait_for(stt.stuck.wait(), timeout=5)
    await say(backend, bob)
    await asyncio.wait_for(both_heard.wait(), timeout=5)

    bob_frames = [d.elapsed for d in backend.deliveries if d.track_id == bob.id]
    print(f"   p-alice speaks first; her recognition takes {stt.seconds * 1000:.0f} ms")
    print(
        f"   p-bob's {len(bob_frames)} frames, delivered meanwhile: "
        f"slowest {max(bob_frames) * 1000:.1f} ms"
    )
    for participant in sorted(heard, key=heard.__getitem__):
        print(f"   {participant} transcribed at {(heard[participant] - started) * 1000:.0f} ms")
    print("   (each track has a lane of its own: Bob's text did not queue behind Alice's)\n")

    await kit.detach_channel(ROOM, "conf")


async def formats() -> None:
    """3. Participants negotiate their own formats, and nothing makes them agree."""
    print("3. Three publishers, three formats\n")
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(
        ConferenceChannel("conf", backend=backend, stt=MockSTTProvider(transcripts=["bonjour"]))
    )
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    published = {
        "p-dial-in": MockTrackFormat(sample_rate=8_000, channels=1, sample_width=1),
        "p-laptop": MockTrackFormat(sample_rate=16_000, channels=1, sample_width=2),
        "p-studio": MockTrackFormat(sample_rate=48_000, channels=2, sample_width=4),
    }

    for participant, audio_format in published.items():
        await backend.simulate_participant_joined(ROOM, participant)
        track = await backend.simulate_track_published(
            ROOM, participant, audio_format=audio_format
        )
        print(f"   {participant:>10}: {audio_format.describe()}")
        await say(backend, track)

    await asyncio.sleep(0.2)  # let the lanes drain
    spoke = {
        event.source.participant_id
        for event in await kit.store.list_events(ROOM)
        if getattr(event.content, "body", None) == "bonjour" and event.source.participant_id
    }
    print(f"\n   transcribed: {', '.join(sorted(spoke)) or 'nobody'}")
    print("   (format normalisation runs first in the lane, so the stages see one format)\n")

    await kit.detach_channel(ROOM, "conf")


async def utterances() -> None:
    """4. What the bot published, grouped by utterance."""
    print("4. Two answers on the bot's track\n")
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(ConferenceChannel("conf", backend=backend, tts=MockTTSProvider()))
    kit.register_channel(AISource("ai"))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "ai")

    await kit.send_event(ROOM, "ai", TextContent(body="bonjour tout le monde"))
    await kit.send_event(ROOM, "ai", TextContent(body="je vous écoute"))

    for index, utterance in enumerate(backend.utterances, start=1):
        state = "complete" if utterance.complete else "unfinished"
        print(f"   utterance {index}: {len(utterance.chunks)} chunks, {state}")
    print(
        "   (one record per utterance: two answers published concurrently would "
        "share one,\n    which is how a test sees them run together)\n"
    )

    await kit.detach_channel(ROOM, "conf")


async def slow_disk() -> None:
    """5. The storage is slow, and the conference does not wait for it."""
    print("5. A recorder that blocks blocks nothing but itself (RFC §12.10.8)\n")
    backend = MockConferenceBackend()
    recorder = SlowRecorder(seconds=0.05)
    kit = RoomKit()
    channel = ConferenceChannel(
        "conf",
        backend=backend,
        recorder=recorder,
        recording=ConferenceRecordingConfig(),
        # A deliberately small backlog, so a dozen frames are enough to show
        # what overload does. The default is 100 — about two seconds of audio.
        max_queued_frames=4,
    )
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    # A recording opens on its track's first frame and nothing is written until
    # ON_RECORDING_STARTED has been heard: it is the consent point (RFC 17.6),
    # where a detach refuses the recording and drops what was buffered.
    started_recording = asyncio.Event()

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
    async def on_recording_started(event: ConferenceRecordingStarted, ctx: object) -> None:
        started_recording.set()

    await backend.simulate_participant_joined(ROOM, "p-alice")
    alice = await backend.simulate_track_published(ROOM, "p-alice")

    loop = asyncio.get_running_loop()
    started = loop.time()
    for _ in range(20):
        await backend.simulate_audio(alice, backend.frame_for(alice))
    delivered_in = (loop.time() - started) * 1000

    # Detaching now would land inside that announcement and refuse the
    # recording. Wait until it has been heard and the first write has begun.
    await asyncio.wait_for(started_recording.wait(), timeout=5)
    while not recorder.threads:
        await asyncio.sleep(0.005)

    dropped = channel.info()["rooms"][ROOM]["recording_dropped_frames"]
    print(f"   20 frames delivered in {delivered_in:.1f} ms")
    print(f"   (each one takes the recorder {recorder.seconds * 1000:.0f} ms to write)")
    print(f"   dropped so far, oldest first: {dropped}")

    # Detaching finalizes the recording, and what is still queued is written
    # first: a container closed over frames in flight would end early and say
    # nothing about it.
    await kit.detach_channel(ROOM, "conf")
    print(f"   frames the recorder ended up with: {len(recorder.chunks)}")
    print(f"   the loop runs on thread {threading.get_ident()}")
    print(f"   the writes ran on {sorted(recorder.threads)} — never on the loop's\n")


async def main() -> None:
    await failures()
    await latency()
    await formats()
    await utterances()
    await slow_disk()


if __name__ == "__main__":
    asyncio.run(main())
