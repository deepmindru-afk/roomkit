"""RoomRecordingMixin — a room's media recordings, started, fed and stopped (RFC §12.11).

Recorders bound at ``create_room(recorders=...)`` start with the room; these
verbs start them on a room that already exists (a recording resumed after a
restart), list them, feed them a track the framework does not wire itself, and
stop them. Every start is announced (ON_RECORDING_STARTED, the consent point of
RFC §17.6) before any media, and every stop with its result
(ON_RECORDING_STOPPED), whichever path stops it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import RoomClosedError
from roomkit.core.mixins.helpers import HelpersMixin, _refuses_writes
from roomkit.models.enums import HookTrigger
from roomkit.recorder._room_recorder_manager import RoomRecordingFeed
from roomkit.voice.events import RecordingStartedEvent, RecordingStoppedEvent

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.recorder._room_recorder_manager import RoomRecorderManager
    from roomkit.recorder.base import (
        MediaRecordingHandle,
        MediaRecordingResult,
        RecordingTrack,
        RoomRecorderBinding,
    )

logger = logging.getLogger("roomkit.framework")


@runtime_checkable
class RoomRecordingHost(Protocol):
    """Contract: capabilities a host class must provide for RoomRecordingMixin.

    Attributes provided by the host's ``__init__``:
        _room_recorder_mgr: Manager of the rooms' recordings.
        _lock_manager: Per-room lock the start and the stop are taken under.
        _hook_engine: Hook engine the announcements run on.

    Methods:
        get_room: From :class:`RoomLifecycleMixin`.
    """

    _room_recorder_mgr: RoomRecorderManager
    _lock_manager: RoomLockManager
    _hook_engine: HookEngine


class RoomRecordingMixin(HelpersMixin):
    """Start, list, feed and stop a room's recordings, each start and stop announced.

    Host contract: :class:`RoomRecordingHost`.
    """

    _room_recorder_mgr: RoomRecorderManager
    _lock_manager: RoomLockManager
    _hook_engine: HookEngine
    get_room: Any  # see RoomRecordingHost / RoomLifecycleMixin

    async def start_room_recording(
        self,
        room_id: str,
        recorders: list[RoomRecorderBinding],
        *,
        organization_id: str | None = None,
    ) -> list[MediaRecordingHandle]:
        """Start *recorders* on an existing room, all or nothing, and announce each.

        The room is read under its lock, scoped to *organization_id* (RFC
        §17.2). The recordings join the ones the room already runs, and
        ON_RECORDING_STARTED fires for each before this returns: a room
        recording captures nothing until a track is added, so the announcement
        precedes any media (RFC §12.11, §17.6). A recording resumed after a
        restart is started here.

        Raises:
            RoomNotFoundError: the room is missing, or another organization's.
            RoomClosedError: the room's status refuses new events (RFC §5.1).
            Exception: a recorder refused to start; the ones already started
                for this call are stopped.
        """
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id, organization_id=organization_id)
            if _refuses_writes(room):
                raise RoomClosedError(f"Room {room_id} does not accept new recordings")
            handles = self._room_recorder_mgr.register(room_id, recorders)
        for handle in handles:
            await self._fire_recording_started(room_id, handle.id)
        return handles

    def room_recordings(self, room_id: str) -> list[MediaRecordingHandle]:
        """The recordings *room_id* runs, in the order they started; empty when none."""
        return self._room_recorder_mgr.handles(room_id)

    def add_room_recording_track(
        self, room_id: str, track: RecordingTrack
    ) -> RoomRecordingFeed | None:
        """Declare *track* to the room's recordings; the feed its media goes through.

        For a source the framework does not wire itself (a channel joined to
        the room wires its own). The track describes the media it will carry
        (RFC §12.11): the feed hands it as declared. ``None`` when the room
        records nothing.
        """
        if not self._room_recorder_mgr.has_recorders(room_id):
            return None
        self._room_recorder_mgr.on_track_added(room_id, track)
        return RoomRecordingFeed(self._room_recorder_mgr, room_id, track)

    async def stop_room_recording(
        self, room_id: str, *, organization_id: str | None = None
    ) -> list[MediaRecordingResult]:
        """Stop the room's recordings, each announced with its result (RFC §12.11).

        The room is read under its lock, scoped to *organization_id*: a call
        refused for another organization stops nothing. Returns the results of
        the recordings that stopped; empty when the room recorded nothing.

        Raises:
            RoomNotFoundError: the room is missing, or another organization's.
        """
        async with self._lock_manager.locked(room_id):
            await self.get_room(room_id, organization_id=organization_id)
            return await self._stop_room_recordings(room_id)

    async def _stop_room_recordings(self, room_id: str) -> list[MediaRecordingResult]:
        """Stop *room_id*'s recordings and announce each that stopped: the one
        stop an explicit stop, a close, an archive and a shutdown share."""
        results = self._room_recorder_mgr.stop_room(room_id)
        await self._announce_stopped(room_id, results)
        return results

    async def _close_room_recorders(self) -> None:
        """Stop every room's recordings and close the recorders, each end
        announced while the hooks can still hear it (a framework shutdown)."""
        for room_id, results in self._room_recorder_mgr.close().items():
            await self._announce_stopped(room_id, results)

    async def _announce_stopped(self, room_id: str, results: list[MediaRecordingResult]) -> None:
        for result in results:
            await self._fire_recording_stopped(room_id, result)

    async def _fire_recording_started(self, room_id: str, recording_id: str) -> None:
        """Announce a room-level recording (ON_RECORDING_STARTED, RFC §17.6)."""
        try:
            context = await self._build_context(room_id)
            await self._hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_RECORDING_STARTED,
                RecordingStartedEvent(id=recording_id, room_id=room_id),
                context,
                skip_event_filter=True,
            )
            await self._emit_framework_event(
                "recording_started",
                room_id=room_id,
                data={"id": recording_id, "scope": "room"},
            )
        except Exception:
            logger.exception("Error announcing room recording %s", recording_id)

    async def _fire_recording_stopped(self, room_id: str, result: MediaRecordingResult) -> None:
        """Announce a room-level recording's end with its result (ON_RECORDING_STOPPED).

        A lookup or a hook that fails is logged: the file is written by then,
        and the close or shutdown that stopped it goes on.
        """
        event = RecordingStoppedEvent(
            id=result.id,
            urls=(result.url,) if result.url else (),
            duration_seconds=result.duration_seconds,
            room_id=room_id,
        )
        try:
            context = await self._build_context(room_id)
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.ON_RECORDING_STOPPED, event, context, skip_event_filter=True
            )
            await self._emit_framework_event(
                "recording_stopped",
                room_id=room_id,
                data={"id": result.id, "scope": "room", "url": result.url},
            )
        except Exception:
            logger.exception("Error announcing the end of room recording %s", result.id)
