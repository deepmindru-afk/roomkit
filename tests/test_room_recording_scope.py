"""Closing or archiving a room stops its recordings only once the room is found (RFC §12.11).

Both are scoped to a tenant (§17.2): a call naming another organization finds
no room. It stopped the room's recordings first and raised after, so a caller
who could not even see the room ended its recording.
"""

from __future__ import annotations

import pytest

from roomkit import RoomKit
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.recorder.base import MediaRecordingConfig, RoomRecorderBinding
from roomkit.recorder.mock import MockMediaRecorder


async def _recorded_room() -> tuple[RoomKit, MockMediaRecorder]:
    kit = RoomKit()
    recorder = MockMediaRecorder()
    await kit.create_room(
        room_id="r1",
        organization_id="tenant-a",
        recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())],
    )
    return kit, recorder


@pytest.mark.parametrize("operation", ["close_room", "archive_room"])
async def test_another_tenants_call_leaves_the_recording_running(operation: str) -> None:
    kit, recorder = await _recorded_room()

    with pytest.raises(RoomNotFoundError):
        await getattr(kit, operation)("r1", organization_id="tenant-b")

    assert recorder.results == []
    await getattr(kit, operation)("r1", organization_id="tenant-a")
    assert len(recorder.results) == 1
    await kit.close()
