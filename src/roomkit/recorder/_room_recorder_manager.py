"""Internal orchestration for room-level media recording."""

from __future__ import annotations

import logging

from roomkit.recorder.base import (
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
    RoomRecorderBinding,
)

logger = logging.getLogger("roomkit.recorder")

# Type alias for clarity: each active binding pairs with its recording handle.
_ActiveBinding = tuple[RoomRecorderBinding, MediaRecordingHandle]


class RoomRecorderManager:
    """Manages room-level media recorders across all rooms.

    Isolates recording orchestration from the framework so that
    ``framework.py`` stays focused on routing and lifecycle.
    """

    def __init__(self) -> None:
        self._registry: dict[str, list[_ActiveBinding]] = {}
        # The tracks each room declares, recorded or not, until they end: a
        # recording that joins the room later is told them before any media.
        self._tracks: dict[str, dict[str, RecordingTrack]] = {}

    def register(
        self, room_id: str, bindings: list[RoomRecorderBinding]
    ) -> list[MediaRecordingHandle]:
        """Start recordings for all bindings in a room, all or nothing, and file them.

        For a room that already exists. Returns the handles started, so the
        caller can announce them (ON_RECORDING_STARTED, RFC §17.6); a room
        recorder captures nothing until a track is added, so the announcement
        still precedes any audio. ``create_room`` calls :meth:`start` and
        :meth:`adopt` itself, to write the room in between.
        """
        return self.adopt(room_id, self.start(room_id, bindings))

    def start(self, room_id: str, bindings: list[RoomRecorderBinding]) -> list[_ActiveBinding]:
        """Start every enabled binding for *room_id*, all or nothing, without filing them.

        A binding that refuses stops the ones already started, and its error is
        raised (RFC §12.11): nothing keeps running for a room that will not
        record. The registry is left alone, so a caller that then fails to
        create the room gives the recordings up with :meth:`discard`.
        """
        active: list[_ActiveBinding] = []
        try:
            for binding in bindings:
                if binding.enabled:
                    active.append((binding, self._start_one(room_id, binding)))
        except BaseException:
            self.discard(active)
            raise
        return active

    @staticmethod
    def _start_one(room_id: str, binding: RoomRecorderBinding) -> MediaRecordingHandle:
        handle = binding.recorder.on_recording_start(binding.config)
        handle.room_id = room_id
        logger.info(
            "Room recording started: %s (recorder=%s, room=%s)",
            handle.id,
            binding.recorder.name,
            room_id,
        )
        return handle

    def adopt(self, room_id: str, active: list[_ActiveBinding]) -> list[MediaRecordingHandle]:
        """File recordings :meth:`start` opened under *room_id*; returns their handles.

        Each is told the room's tracks first, so media already flowing in the
        room reaches it only once it knows the track's format (RFC §12.11).
        They join the room's recordings rather than replace them: a store that
        rewrites a room created again under its id (in-memory, SQLite) must not
        orphan the recordings that room already runs, which only a registry
        entry lets anything stop.
        """
        tracks = list(self._tracks.get(room_id, {}).values())
        for binding, handle in active:
            for track in tracks:
                binding.recorder.on_track_added(handle, track)
        if active:
            self._registry.setdefault(room_id, []).extend(active)
        return [handle for _binding, handle in active]

    @staticmethod
    def discard(active: list[_ActiveBinding]) -> None:
        """Stop recordings that were started but never filed under a room."""
        for binding, handle in active:
            if _stop_quietly(binding, handle) is not None:
                logger.info("Room recording discarded: %s (room=%s)", handle.id, handle.room_id)

    def on_track_added(self, room_id: str, track: RecordingTrack) -> None:
        """Declare a track of *room_id*: to its recordings, and to any that joins later."""
        self._tracks.setdefault(room_id, {})[track.id] = track
        for binding, handle in self._registry.get(room_id, []):
            binding.recorder.on_track_added(handle, track)

    def on_track_removed(self, room_id: str, track: RecordingTrack) -> None:
        """End a track of *room_id*: each recording flushes it, and a recording
        joining later is not told of it."""
        tracks = self._tracks.get(room_id, {})
        tracks.pop(track.id, None)
        if not tracks:
            self._tracks.pop(room_id, None)
        for binding, handle in self._registry.get(room_id, []):
            binding.recorder.on_track_removed(handle, track)

    def on_data(
        self,
        room_id: str,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        """Fan out media data to all recorders in a room."""
        for binding, handle in self._registry.get(room_id, []):
            binding.recorder.on_data(handle, track, data, timestamp_ms)

    def stop_room(self, room_id: str) -> list[MediaRecordingResult]:
        """Stop all recordings in a room and return the results of those that stopped.

        A recorder that fails to stop is logged and its result left out: the
        room's other recordings still stop, and the room still closes.
        """
        results: list[MediaRecordingResult] = []
        for binding, handle in self._registry.pop(room_id, []):
            result = _stop_quietly(binding, handle)
            if result is None:
                continue
            results.append(result)
            logger.info(
                "Room recording stopped: %s (%.1fs, %d bytes)",
                result.id,
                result.duration_seconds,
                result.size_bytes,
            )
        return results

    def has_recorders(self, room_id: str) -> bool:
        """Check if a room has active recorders."""
        return room_id in self._registry

    def handles(self, room_id: str) -> list[MediaRecordingHandle]:
        """The handles of the recordings *room_id* runs, in the order they started."""
        return [handle for _binding, handle in self._registry.get(room_id, [])]

    def close(self) -> dict[str, list[MediaRecordingResult]]:
        """Stop all rooms and close all recorders, each failure logged on its own.

        Returns the results of the recordings that stopped, by room, for the
        caller to announce.
        """
        stopped: dict[str, list[MediaRecordingResult]] = {}
        seen_recorders: set[int] = set()
        self._tracks.clear()
        for room_id in list(self._registry):
            for binding, handle in self._registry.pop(room_id, []):
                result = _stop_quietly(binding, handle)
                if result is not None:
                    stopped.setdefault(room_id, []).append(result)
                recorder_id = id(binding.recorder)
                if recorder_id not in seen_recorders:
                    seen_recorders.add(recorder_id)
                    _close_quietly(binding)
        return stopped


def _stop_quietly(
    binding: RoomRecorderBinding, handle: MediaRecordingHandle
) -> MediaRecordingResult | None:
    """Stop one recording; a recorder that fails is logged and ``None`` returned.

    Every path that stops room recordings goes through here, so one recorder
    that raises never leaves the others of its room running.
    """
    try:
        return binding.recorder.on_recording_stop(handle)
    except Exception:
        logger.exception(
            "Failed to stop room recording %s (recorder=%s)", handle.id, binding.recorder.name
        )
        return None


def _close_quietly(binding: RoomRecorderBinding) -> None:
    """Close one recorder; a recorder that fails is logged."""
    try:
        binding.recorder.close()
    except Exception:
        logger.exception("Failed to close room recorder %s", binding.recorder.name)
