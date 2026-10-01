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

    def register(
        self, room_id: str, bindings: list[RoomRecorderBinding]
    ) -> list[MediaRecordingHandle]:
        """Start recordings for all bindings in a room, all or nothing, and file them.

        Returns the handles started, so the caller can announce them
        (ON_RECORDING_STARTED, RFC §17.6). A room recorder captures nothing
        until a track is added, so the announcement still precedes any audio.
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
        """File recordings :meth:`start` opened under *room_id*; returns their handles."""
        if active:
            self._registry[room_id] = active
        return [handle for _binding, handle in active]

    @staticmethod
    def discard(active: list[_ActiveBinding]) -> None:
        """Stop recordings that were started but never filed under a room."""
        for binding, handle in active:
            try:
                binding.recorder.on_recording_stop(handle)
            except Exception:
                logger.exception("Failed to stop discarded room recording %s", handle.id)

    def on_track_added(self, room_id: str, track: RecordingTrack) -> None:
        """Notify all recorders in a room about a new track."""
        for binding, handle in self._registry.get(room_id, []):
            binding.recorder.on_track_added(handle, track)

    def on_track_removed(self, room_id: str, track: RecordingTrack) -> None:
        """Notify all recorders in a room about a removed track."""
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
        """Stop all recordings in a room and return results."""
        results: list[MediaRecordingResult] = []
        for binding, handle in self._registry.pop(room_id, []):
            result = binding.recorder.on_recording_stop(handle)
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

    def close(self) -> None:
        """Stop all rooms and close all recorders."""
        seen_recorders: set[int] = set()
        for room_id in list(self._registry):
            for binding, handle in self._registry.pop(room_id, []):
                binding.recorder.on_recording_stop(handle)
                recorder_id = id(binding.recorder)
                if recorder_id not in seen_recorders:
                    seen_recorders.add(recorder_id)
                    binding.recorder.close()
