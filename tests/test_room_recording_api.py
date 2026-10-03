"""A room's recordings, started on an existing room, fed, listed and stopped (RFC §12.11).

Every start is announced (ON_RECORDING_STARTED, the consent point of §17.6)
before any media, and every end with its result (ON_RECORDING_STOPPED),
whichever path stops it: an explicit stop, a close, an archive, a shutdown.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.core.exceptions import RoomClosedError, RoomNotFoundError
from roomkit.recorder.base import (
    MediaRecordingConfig,
    MediaRecordingHandle,
    RecordingTrack,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder

_TRACK = RecordingTrack(
    id="audio:s1", kind="audio", channel_id="capture", codec="pcm_s16le", sample_rate=48000
)


class _Refusing(MockMediaRecorder):
    """A recorder that refuses to start: storage it cannot write, say."""

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        raise ValueError("storage refused")


def _binding(recorder: MockMediaRecorder) -> RoomRecorderBinding:
    return RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())


async def _kit() -> tuple[RoomKit, list[tuple[str, Any]]]:
    """A kit with room r1 of tenant-a, and the recording hooks it heard, in order."""
    kit = RoomKit()
    await kit.create_room(room_id="r1", organization_id="tenant-a")
    heard: list[tuple[str, Any]] = []

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC, name="started")
    async def started(event: Any, ctx: Any) -> None:
        heard.append(("started", event))

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC, name="stopped")
    async def stopped(event: Any, ctx: Any) -> None:
        heard.append(("stopped", event))

    return kit, heard


async def test_a_recording_started_on_an_existing_room_is_announced_before_any_media() -> None:
    kit, heard = await _kit()
    recorder = MockMediaRecorder()

    handles = await kit.start_room_recording(
        "r1", [_binding(recorder)], organization_id="tenant-a"
    )

    assert [(kind, event.id, event.room_id) for kind, event in heard] == [
        ("started", handles[0].id, "r1")
    ]
    assert recorder.chunks == []
    assert kit.room_recordings("r1") == handles
    await kit.close()


async def test_a_track_is_fed_through_the_framework() -> None:
    kit, _heard = await _kit()
    recorder = MockMediaRecorder()
    await kit.start_room_recording("r1", [_binding(recorder)])

    feed = kit.add_room_recording_track("r1", _TRACK)
    assert feed is not None
    feed.feed(b"\x01\x02" * 480, 20.0)
    feed.close()

    assert [chunk.data for chunk in recorder.chunks] == [b"\x01\x02" * 480]
    assert recorder.tracks == []  # the closed track is flushed and removed
    await kit.close()


async def test_a_room_that_records_nothing_gives_no_feed() -> None:
    kit, _heard = await _kit()

    assert kit.add_room_recording_track("r1", _TRACK) is None
    assert kit.room_recordings("r1") == []
    await kit.close()


async def test_the_start_is_all_or_nothing() -> None:
    kit, heard = await _kit()
    started = MockMediaRecorder()

    with pytest.raises(ValueError, match="storage refused"):
        await kit.start_room_recording("r1", [_binding(started), _binding(_Refusing())])

    assert len(started.results) == 1  # stopped, not left running
    assert kit.room_recordings("r1") == []
    assert heard == []
    await kit.close()


@pytest.mark.parametrize("operation", ["close_room", "archive_room"])
async def test_a_room_that_refuses_events_refuses_a_recording(operation: str) -> None:
    kit, heard = await _kit()
    await getattr(kit, operation)("r1")
    recorder = MockMediaRecorder()

    with pytest.raises(RoomClosedError):
        await kit.start_room_recording("r1", [_binding(recorder)])

    assert recorder.handles == []
    assert heard == []
    await kit.close()


async def test_another_tenants_room_is_not_found_and_nothing_starts() -> None:
    kit, _heard = await _kit()
    recorder = MockMediaRecorder()

    with pytest.raises(RoomNotFoundError):
        await kit.start_room_recording("r1", [_binding(recorder)], organization_id="tenant-b")
    with pytest.raises(RoomNotFoundError):
        await kit.stop_room_recording("r1", organization_id="tenant-b")

    assert recorder.handles == []
    await kit.close()


async def test_an_explicit_stop_returns_and_announces_each_result() -> None:
    kit, heard = await _kit()
    recorder = MockMediaRecorder()
    [handle] = await kit.start_room_recording("r1", [_binding(recorder)])

    results = await kit.stop_room_recording("r1", organization_id="tenant-a")

    assert [result.id for result in results] == [handle.id]
    kind, event = heard[-1]
    assert (kind, event.id, event.room_id, event.session) == ("stopped", handle.id, "r1", None)
    assert event.urls == (results[0].url,)
    assert kit.room_recordings("r1") == []
    await kit.close()


@pytest.mark.parametrize("operation", ["close_room", "archive_room", "close"])
async def test_every_other_path_that_stops_a_recording_announces_its_end(operation: str) -> None:
    kit, heard = await _kit()
    [handle] = await kit.start_room_recording("r1", [_binding(MockMediaRecorder())])

    if operation == "close":
        await kit.close()
    else:
        await getattr(kit, operation)("r1")

    assert [(kind, event.id) for kind, event in heard] == [
        ("started", handle.id),
        ("stopped", handle.id),
    ]
    if operation != "close":
        await kit.close()


async def test_a_recording_resumed_after_a_stop_is_announced_again() -> None:
    """The consent point is announced on every start, a resume included."""
    kit, heard = await _kit()
    await kit.start_room_recording("r1", [_binding(MockMediaRecorder())])
    await kit.stop_room_recording("r1")

    [resumed] = await kit.start_room_recording("r1", [_binding(MockMediaRecorder())])

    assert heard[-1][0] == "started" and heard[-1][1].id == resumed.id
    await kit.close()
