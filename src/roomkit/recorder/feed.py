"""The feed a host hands a room recording's media through (RFC §12.11)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.recorder._room_recorder_manager import RoomRecorderManager
    from roomkit.recorder.base import RecordingTrack


class RoomRecordingFeed:
    """Hands one track's media to a room's recordings.

    What :meth:`RoomKit.add_room_recording_track` returns to a caller that
    feeds a room recording from a source the framework does not wire itself:
    the track was declared to every recording of the room (and is to any that
    joins later), and the media goes through the framework, never through a
    recorder the caller holds.
    """

    def __init__(self, manager: RoomRecorderManager, room_id: str, track: RecordingTrack) -> None:
        self._manager = manager
        self._room_id = room_id
        self.track = track

    def feed(self, data: bytes, timestamp_ms: float | None = None) -> None:
        """Hand *data*, in the format the track declares, to the room's recordings."""
        self._manager.on_data(self._room_id, self.track, data, timestamp_ms)

    def close(self) -> None:
        """End the track: each recording flushes what it holds of it."""
        self._manager.on_track_removed(self._room_id, self.track)
