"""Learning where a conference recording was written.

Framework-mode conference recording opens one recording per subscribed track,
attributed to the participant publishing it (RFC §12.10.8). This example shows
how an integrator finds out where those files went, which is the whole point of
recording something: ON_RECORDING_STARTED when a track's recording opens, and
ON_RECORDING_STOPPED when it closes, carrying the location, the duration and the
attribution.

One report per track, not one per conference — the tracks of a meeting do not
end together, and a participant who leaves halfway through has a finished file
while the meeting runs on. An integrator wanting the meeting's full list
accumulates it by room, as this example does.

Runs against MockConferenceBackend and MockMediaRecorder, so it needs no SFU and
no codec: swap in a real backend and PyAVMediaRecorder and the same handlers
report real paths on disk.

Run with:
    uv run python examples/conference_recording_result.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections import defaultdict
from pathlib import Path

from roomkit import (
    ConferenceRecordingConfig,
    ConferenceRecordingStarted,
    ConferenceRecordingStopped,
    HookExecution,
    HookTrigger,
    MockConferenceBackend,
    RoomContext,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.audio_frame import AudioFrame

ROOM = "board-meeting"
STORAGE = Path(tempfile.gettempdir()) / "roomkit-conference-recordings"


def speech() -> AudioFrame:
    """20 ms of 16 kHz PCM. Content does not matter here — arrival does."""
    return AudioFrame(data=b"\x01\x00" * 320, sample_rate=16000)


async def main() -> None:
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(
        ConferenceChannel(
            "conf",
            backend=backend,
            recorder=MockMediaRecorder(),
            # MockMediaRecorder writes nothing; PyAVMediaRecorder would write
            # real files under STORAGE, and refuses to unless the config also
            # sets encryption= or storage_encrypted_at_rest=True (RFC 17.6).
            recording=ConferenceRecordingConfig(storage=str(STORAGE), format="wav"),
        )
    )
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    # What a compliance archive would keep: one entry per file, by meeting.
    archive: dict[str, list[ConferenceRecordingStopped]] = defaultdict(list)
    # The hooks run asynchronously, after the frame or the unpublish that caused
    # them; the example waits on these so each report prints under its step.
    opened: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
    closed: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
    async def on_started(event: ConferenceRecordingStarted, ctx: RoomContext) -> None:
        print(f"  ▶ recording {event.id} opened for {event.participant_id} ({event.kind})")
        opened[event.participant_id].set()

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
    async def on_stopped(event: ConferenceRecordingStopped, ctx: RoomContext) -> None:
        archive[event.room_id].append(event)
        print(
            f"  ■ recording {event.id} closed for {event.participant_id} → "
            f"{event.url} ({event.duration_seconds:.1f}s, {event.size_bytes} bytes)"
        )
        closed[event.participant_id].set()

    async def heard(events: defaultdict[str, asyncio.Event], *participants: str) -> None:
        waits = (events[participant].wait() for participant in participants)
        await asyncio.wait_for(asyncio.gather(*waits), timeout=5)

    print("Alice and Bob join and speak:")
    await backend.simulate_participant_joined(ROOM, "p-alice")
    await backend.simulate_participant_joined(ROOM, "p-bob")
    alice = await backend.simulate_track_published(ROOM, "p-alice")
    bob = await backend.simulate_track_published(ROOM, "p-bob")
    for _ in range(5):
        await backend.simulate_audio(alice, speech())
        await backend.simulate_audio(bob, speech())
    await heard(opened, "p-alice", "p-bob")

    print("\nA participant who publishes but never speaks leaves no file:")
    await backend.simulate_participant_joined(ROOM, "p-carol")
    carol = await backend.simulate_track_published(ROOM, "p-carol")
    await backend.simulate_track_unpublished(carol.id)
    print("  (nothing reported for p-carol — the recording opens on the first frame)")

    print("\nAlice leaves early — her recording closes while the meeting runs on:")
    await backend.simulate_track_unpublished(alice.id)
    await heard(closed, "p-alice")

    print("\nThe meeting ends:")
    await kit.detach_channel(ROOM, "conf")
    await heard(closed, "p-bob")

    print(f"\nArchived for {ROOM}:")
    for entry in archive[ROOM]:
        print(f"  {entry.participant_id:>10}  {entry.url}")


if __name__ == "__main__":
    asyncio.run(main())
