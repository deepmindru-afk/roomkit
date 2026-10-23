"""In-memory implementation of ConversationStore."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import VerificationStatus
from roomkit.models.event import DeleteContent, EditContent, RoomEvent, ThreadSummary
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter, includes_blocked, received_events
from roomkit.models.task import Observation, Task
from roomkit.models.verification import Verification, VerificationRequest
from roomkit.models.voice_delivery import VoiceDeliveryRecord
from roomkit.store.base import (
    ConversationStore,
    address_linked,
    committed_state,
    deletes_edits,
    edits_target,
    is_marked_deleted,
    link_target,
    marked_deleted,
    room_taken,
    target_gone,
    without_address,
)


class InMemoryStore(ConversationStore):
    """Dict-based in-memory store for development and testing."""

    is_process_local = True

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}
        self._events: dict[str, RoomEvent] = {}
        self._room_events: dict[str, list[str]] = {}
        # Next authoritative event index per room. This is a sequence, not the
        # current event count: hard deletion never makes an old index reusable.
        self._next_event_index: dict[str, int] = {}
        self._bindings: dict[str, dict[str, ChannelBinding]] = {}
        self._participants: dict[str, dict[str, Participant]] = {}
        # room -> (source channel, idempotency key) -> the event id it
        # committed: a key is unique per room and source channel (RFC §13.4)
        self._idempotency: dict[str, dict[tuple[str, str], str]] = {}
        # room -> channel -> the highest read event index (a high-water mark)
        self._read_markers: dict[str, dict[str, int]] = {}
        self._identities: dict[str, Identity] = {}
        # (channel_type, address, organization_id) — the empty string is
        # the unscoped tenant, so every key has the same shape (RFC §17.2).
        self._address_index: dict[tuple[str, str, str], str] = {}
        # (identity_id, organization_id), the same empty-string convention.
        self._verifications: dict[tuple[str, str], Verification] = {}
        self._verification_requests: dict[str, VerificationRequest] = {}
        self._tasks: dict[str, Task] = {}
        self._room_tasks: dict[str, list[str]] = {}
        self._observations: dict[str, Observation] = {}
        self._room_observations: dict[str, list[str]] = {}
        # Per-room locks for atomic index assignment
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._voice_deliveries: dict[tuple[str, str], VoiceDeliveryRecord] = {}
        # find_latest_room candidate index: participant id -> rooms where that
        # participant appears (as a Participant row or on a binding). Turns
        # the per-inbound-message room scan into a lookup over the handful of
        # rooms the sender actually touches. Over-inclusion is harmless — the
        # read re-checks the full predicate per candidate — so writes only
        # ever add, except the two spots that can name what they invalidate
        # (remove_binding, delete_room).
        self._participant_room_index: dict[str, set[str]] = {}

    def _index_participant(self, pid: str | None, room_id: str) -> None:
        if pid:
            self._participant_room_index.setdefault(pid, set()).add(room_id)

    def _unindex_participant(self, pid: str | None, room_id: str) -> None:
        """Drop pid→room when nothing in the room references pid anymore."""
        if not pid:
            return
        if pid in self._participants.get(room_id, {}):
            return
        for b in self._bindings.get(room_id, {}).values():
            if b.participant_id == pid:
                return
        rooms = self._participant_room_index.get(pid)
        if rooms is not None:
            rooms.discard(room_id)
            if not rooms:
                del self._participant_room_index[pid]

    def _unindex_room(self, room_id: str) -> None:
        """Remove a deleted room from every participant's candidate set."""
        pids = set(self._participants.get(room_id, {}).keys())
        pids.update(
            b.participant_id for b in self._bindings.get(room_id, {}).values() if b.participant_id
        )
        for pid in pids:
            rooms = self._participant_room_index.get(pid)
            if rooms is not None:
                rooms.discard(room_id)
                if not rooms:
                    del self._participant_room_index[pid]

    def _lock_for(self, room_id: str) -> asyncio.Lock:
        """Return (or create) the asyncio.Lock for a room."""
        if room_id not in self._room_locks:
            self._room_locks[room_id] = asyncio.Lock()
        return self._room_locks[room_id]

    async def get_voice_delivery(self, room_id: str, key_hash: str) -> VoiceDeliveryRecord | None:
        record = self._voice_deliveries.get((room_id, key_hash))
        return record.model_copy(deep=True) if record is not None else None

    async def claim_voice_delivery(self, record: VoiceDeliveryRecord) -> VoiceDeliveryRecord:
        if record.outcome is not None:
            raise ValueError("A new claim must have no outcome")
        if record.room_id not in self._rooms:
            raise ValueError("Room does not exist")
        key = (record.room_id, record.key_hash)
        current = self._voice_deliveries.get(key)
        if current is None or (current.retryable and current.content_hash == record.content_hash):
            current = record.model_copy(deep=True)
            self._voice_deliveries[key] = current
        return current.model_copy(deep=True)

    async def complete_voice_delivery(self, record: VoiceDeliveryRecord) -> bool:
        if record.outcome is None:
            raise ValueError("Completion requires an outcome")
        key = (record.room_id, record.key_hash)
        current = self._voice_deliveries.get(key)
        if current is None or current.attempt_id != record.attempt_id:
            return False
        if current.outcome is not None:
            return current.outcome == record.outcome
        self._voice_deliveries[key] = current.model_copy(
            update={"outcome": record.outcome.model_copy(deep=True)}
        )
        return True

    def _reserve_event_index(self, room_id: str) -> int:
        """Return and advance the room sequence while its lock is held."""
        index = self._next_event_index.get(room_id, 0)
        self._next_event_index[room_id] = index + 1
        return index

    # Room operations

    async def create_room(self, room: Room) -> Room:
        if room.id in self._rooms:
            raise room_taken(room.id)
        self._rooms[room.id] = room
        self._room_events.setdefault(room.id, [])
        self._next_event_index.setdefault(room.id, 0)
        self._bindings.setdefault(room.id, {})
        self._participants.setdefault(room.id, {})
        self._idempotency.setdefault(room.id, {})
        self._read_markers.setdefault(room.id, {})
        return room

    async def get_room(self, room_id: str) -> Room | None:
        room = self._rooms.get(room_id)
        return room.model_copy(deep=True) if room is not None else None

    async def update_room(self, room: Room) -> Room:
        if room.id not in self._rooms:
            from roomkit.core.framework import RoomNotFoundError

            raise RoomNotFoundError(room.id)
        # delivered_index, event_count and latest_index are store-managed
        # (advance_delivered_index, add_event): a caller's stale copy must
        # never rewind them.
        current = self._rooms[room.id]
        room = room.model_copy(
            update={
                "delivered_index": current.delivered_index,
                "event_count": current.event_count,
                "latest_index": current.latest_index,
            }
        )
        self._rooms[room.id] = room
        return room

    async def room_exists(self, room_id: str) -> bool:
        return room_id in self._rooms

    async def get_delivered_index(self, room_id: str) -> int:
        room = self._rooms.get(room_id)
        return -1 if room is None else room.delivered_index

    async def advance_delivered_index(
        self, room_id: str, index: int, *, force: bool = False
    ) -> bool:
        room = self._rooms.get(room_id)
        if room is None:
            return False
        if room.delivered_index >= index:
            return False
        if not force and room.delivered_index != index - 1:
            return False
        self._rooms[room_id] = room.model_copy(update={"delivered_index": index})
        return True

    async def delete_room(self, room_id: str) -> bool:
        if room_id not in self._rooms:
            return False
        self._unindex_room(room_id)
        del self._rooms[room_id]
        # Clean up events
        event_ids = self._room_events.pop(room_id, [])
        for eid in event_ids:
            self._events.pop(eid, None)
        # Clean up tasks
        task_ids = self._room_tasks.pop(room_id, [])
        for tid in task_ids:
            self._tasks.pop(tid, None)
        # Clean up observations
        obs_ids = self._room_observations.pop(room_id, [])
        for oid in obs_ids:
            self._observations.pop(oid, None)
        self._bindings.pop(room_id, None)
        self._participants.pop(room_id, None)
        self._idempotency.pop(room_id, None)
        self._read_markers.pop(room_id, None)
        self._next_event_index.pop(room_id, None)
        self._room_locks.pop(room_id, None)
        self._voice_deliveries = {
            key: value for key, value in self._voice_deliveries.items() if key[0] != room_id
        }
        return True

    def _rooms_in_order(self) -> list[Room]:
        """Rooms in the order every store pages them: created_at, then id."""
        return sorted(self._rooms.values(), key=_recency)

    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        rooms = self._rooms_in_order()
        return [r.model_copy(deep=True) for r in rooms[offset : offset + limit]]

    async def find_rooms(
        self,
        organization_id: str | None = None,
        status: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Room]:
        results: list[Room] = []
        for room in self._rooms_in_order():
            if organization_id is not None and room.organization_id != organization_id:
                continue
            if status is not None and room.status.value != status:
                continue
            if metadata_filter and not all(
                room.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            results.append(room)
        return [room.model_copy(deep=True) for room in results[offset : offset + limit]]

    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        candidates = self._participant_room_index.get(participant_id)
        if not candidates:
            return None
        best: Room | None = None
        for room_id in candidates:
            room = self._rooms.get(room_id)
            if room is None:
                continue
            if status is not None and room.status.value != status:
                continue
            # Check if participant is in this room
            if participant_id not in self._participants.get(room.id, {}):
                # Also check bindings for participant_id
                found = False
                for binding in self._bindings.get(room.id, {}).values():
                    if binding.participant_id == participant_id and (
                        channel_type is None or binding.channel_type.value == channel_type
                    ):
                        found = True
                        break
                if not found:
                    continue
            if best is None or _recency(room) > _recency(best):
                best = room
        return best.model_copy(deep=True) if best is not None else None

    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        matches = await self.find_room_ids_by_channel(channel_id, status=status, limit=1)
        return matches[0] if matches else None

    async def find_room_id_by_binding(
        self,
        channel_id: str,
        participant_id: str,
        status: str | None = None,
        *,
        organization_id: str | None = None,
    ) -> str | None:
        # The participant index holds every room a binding names the
        # participant in, so this reads a handful of rooms, not all of them.
        best: tuple[float, str] | None = None
        for room_id in self._participant_room_index.get(participant_id, ()):
            room = self._rooms.get(room_id)
            binding = self._bindings.get(room_id, {}).get(channel_id)
            if room is None or binding is None or binding.participant_id != participant_id:
                continue
            if not _matches(room, status, organization_id):
                continue
            if best is None or _recency(room) > best:
                best = _recency(room)
        return best[1] if best is not None else None

    async def find_room_id_by_participant(
        self,
        participant_id: str,
        status: str | None = None,
        *,
        organization_id: str | None = None,
    ) -> str | None:
        best: tuple[float, str] | None = None
        for room_id in self._participant_room_index.get(participant_id, ()):
            room = self._rooms.get(room_id)
            if room is None or participant_id not in self._participants.get(room_id, {}):
                continue
            if not _matches(room, status, organization_id):
                continue
            if best is None or _recency(room) > best:
                best = _recency(room)
        return best[1] if best is not None else None

    async def find_room_ids_by_channel(
        self,
        channel_id: str,
        status: str | None = None,
        limit: int = 2,
        *,
        organization_id: str | None = None,
    ) -> list[str]:
        # Ordered by the room's creation time, oldest first, with the id as
        # tie-breaker: dict insertion order would be an accident of how the
        # process happened to load, and the answer has to be the same one the
        # SQL store gives (RFC §10.4).
        matched: list[tuple[float, str]] = []
        for room_id, bindings in self._bindings.items():
            if channel_id not in bindings:
                continue
            room = self._rooms.get(room_id)
            if room is None or not _matches(room, status, organization_id):
                continue
            matched.append(_recency(room))
        matched.sort()
        return [room_id for _created_at, room_id in matched[:limit]]

    # Event operations

    async def add_event(self, event: RoomEvent) -> RoomEvent:
        # Copy-in: the store owns its stored representation from the moment
        # the write returns (RFC 14.4) — a caller's later mutation of the
        # object it passed must not reach the log. One deep copy per write
        # is what lets every read share the stored object instead of paying
        # a deep copy per event per read.
        event = event.model_copy(deep=True)
        self._events[event.id] = event
        self._room_events.setdefault(event.room_id, []).append(event.id)
        self._next_event_index[event.room_id] = max(
            self._next_event_index.get(event.room_id, 0), event.index + 1
        )
        if event.idempotency_key:
            self._idempotency.setdefault(event.room_id, {})[
                (event.source.channel_id, event.idempotency_key)
            ] = event.id
        return event

    async def get_event(self, event_id: str) -> RoomEvent | None:
        # Committed events are immutable (RFC 4, 14.4): reads share the
        # stored snapshot instead of deep-copying it — the caller must
        # treat it as frozen.
        return self._events.get(event_id)

    async def update_event(self, event: RoomEvent) -> RoomEvent:
        event = event.model_copy(deep=True)
        self._events[event.id] = event
        return event

    async def write_change_target(self, room_id: str, target: RoomEvent) -> None:
        async with self._lock_for(room_id):
            stored = self._events.get(target.id)
            if stored is None or stored.room_id != room_id:
                raise target_gone(target)
            self._events[target.id] = target.model_copy(deep=True)
            if is_marked_deleted(target):
                self._mark_edits_deleted_now(room_id, target.id)

    def _mark_edits_deleted_now(self, room_id: str, target_id: str) -> None:
        """Mark every edit of *target_id* deleted (RFC §10.3), under the
        caller's critical section."""
        for eid in self._room_events.get(room_id, []):
            edit = self._events.get(eid)
            if edit is not None and edits_target(edit, target_id):
                self._events[eid] = marked_deleted(edit)

    async def record_delivery_results(
        self, room_id: str, event_id: str, results: dict[str, Any]
    ) -> None:
        stored = self._events.get(event_id)
        if stored is not None and stored.room_id == room_id:
            self._events[event_id] = stored.model_copy(
                update={"delivery_results": copy.deepcopy(results)}
            )

    async def delete_event(
        self, room_id: str, event_id: str, *, cascade_replies: bool = True
    ) -> list[RoomEvent]:
        root = self._events.get(event_id)
        if root is None or root.room_id != room_id:
            return []
        in_room = sorted(
            (e for eid in self._room_events.get(room_id, []) if (e := self._events.get(eid))),
            key=lambda e: e.index,
        )
        removed = [root]
        if cascade_replies:
            removed.extend(
                e for e in in_room if e.id != event_id and e.parent_event_id == event_id
            )
        gone = {e.id for e in removed}
        removed.extend(e for e in in_room if e.id not in gone and _changes_one_of(e, gone))
        gone = {e.id for e in removed}
        for event in removed:
            del self._events[event.id]
            if event.idempotency_key:
                self._idempotency.get(room_id, {}).pop(
                    (event.source.channel_id, event.idempotency_key), None
                )
        self._room_events[room_id] = [
            eid for eid in self._room_events.get(room_id, []) if eid not in gone
        ]
        return removed

    async def list_events(
        self,
        room_id: str,
        offset: int = 0,
        limit: int = 50,
        visibility_filter: str | None = None,
        *,
        after_index: int | None = None,
        before_index: int | None = None,
        event_filter: EventFilter | None = None,
        newest_first: bool = False,
    ) -> list[RoomEvent]:
        if after_index is not None and before_index is not None:
            raise ValueError("after_index and before_index are mutually exclusive")

        event_ids = self._room_events.get(room_id, [])
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        # Rendered by ``index`` whatever the commit order (RFC §14.1); the sort
        # is stable, so events stored without an index keep commit order.
        events.sort(key=lambda e: e.index)

        if after_index is not None:
            events = [e for e in events if e.index > after_index]
        elif before_index is not None:
            events = [e for e in events if e.index < before_index]

        # Visibility: event_filter.visibility takes precedence. The criteria
        # and the received-rows default run ahead of the page cut, so a page of
        # ``limit`` events is full whatever was refused around them.
        effective_visibility = (
            event_filter.visibility
            if event_filter is not None and event_filter.visibility is not None
            else visibility_filter
        )
        events = self._apply_event_filter(events, event_filter, visibility=effective_visibility)

        if before_index is not None:
            return events[-limit:] if limit < len(events) else events
        if after_index is not None:
            return events[:limit]
        if newest_first:
            # Newest ``limit`` events (offset counted from the newest end),
            # returned oldest-first so the snapshot reads chronologically.
            # Mirrors the ``before_index`` tail slice.
            end = max(0, len(events) - offset)
            start = max(0, end - limit)
            return events[start:end]
        return events[offset : offset + limit]

    @staticmethod
    def _apply_event_filter(
        events: list[RoomEvent], ef: EventFilter | None, *, visibility: str | None = None
    ) -> list[RoomEvent]:
        """The rows of a page, or of the count that stands for it, under ``ef``.

        The one constructor behind :meth:`list_events` and
        :meth:`get_event_count`, so the two never diverge on a criterion: the
        visibility the caller resolved, then the filter's own fields, then the
        received-rows default (RFC §14.1) unless ``include_blocked`` lifts it.
        """
        if visibility is not None:
            events = [e for e in events if e.visibility == visibility]
        if ef is None:
            return received_events(events)
        if ef.event_types is not None:
            allowed = set(ef.event_types)
            events = [e for e in events if e.type in allowed]
        if ef.exclude_types is not None:
            excluded = set(ef.exclude_types)
            events = [e for e in events if e.type not in excluded]
        if ef.source_channel_id is not None:
            events = [e for e in events if e.source.channel_id == ef.source_channel_id]
        if ef.source_channel_type is not None:
            events = [e for e in events if e.source.channel_type == ef.source_channel_type]
        if ef.correlation_id is not None:
            events = [e for e in events if e.correlation_id == ef.correlation_id]
        if ef.responds_to is not None:
            events = [e for e in events if e.responds_to == ef.responds_to]
        if ef.participant_id is not None:
            events = [e for e in events if e.source.participant_id == ef.participant_id]
        if ef.parent_event_id is not None:
            events = [e for e in events if e.parent_event_id == ef.parent_event_id]
        if ef.top_level_only:
            events = [e for e in events if e.parent_event_id is None]
        # Compared as timestamps, as SQLite stores them: a naive bound against
        # an aware created_at would make the datetimes themselves refuse.
        if ef.after_time is not None:
            after = ef.after_time.timestamp()
            events = [e for e in events if e.created_at.timestamp() > after]
        if ef.before_time is not None:
            before = ef.before_time.timestamp()
            events = [e for e in events if e.created_at.timestamp() < before]
        if not includes_blocked(ef):
            events = received_events(events)
        return events

    async def get_thread_summaries(
        self, room_id: str, root_event_ids: list[str]
    ) -> dict[str, ThreadSummary]:
        roots = set(root_event_ids)
        summaries: dict[str, ThreadSummary] = {}
        stored = (
            self._events[eid] for eid in self._room_events.get(room_id, []) if eid in self._events
        )
        # A refused reply is not counted: the list a reader opens from the
        # affordance is a default timeline read, which does not serve it.
        for event in received_events(stored):
            if event.parent_event_id not in roots:
                continue
            summary = summaries.get(event.parent_event_id)
            if summary is None:
                summary = ThreadSummary(root_event_id=event.parent_event_id)
                summaries[event.parent_event_id] = summary
            summary.reply_count += 1
            last = summary.last_reply_at
            if last is None or event.created_at.timestamp() > last.timestamp():
                summary.last_reply_at = event.created_at
        return summaries

    async def check_idempotency(self, room_id: str, key: str, *, channel_id: str) -> bool:
        return (channel_id, key) in self._idempotency.get(room_id, {})

    async def get_event_by_idempotency_key(
        self, room_id: str, key: str, *, channel_id: str
    ) -> RoomEvent | None:
        event_id = self._idempotency.get(room_id, {}).get((channel_id, key))
        return self._events.get(event_id) if event_id is not None else None

    async def get_event_count(self, room_id: str, event_filter: EventFilter | None = None) -> int:
        event_ids = self._room_events.get(room_id, [])
        if event_filter is None:
            return len(event_ids)
        # The rows a ``list_events`` page would serve under this filter, and
        # no page (RFC §14.1).
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        return len(
            self._apply_event_filter(events, event_filter, visibility=event_filter.visibility)
        )

    async def add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Atomically reserve the room's next monotonic index and append."""
        async with self._lock_for(room_id):
            events = self._room_events.setdefault(room_id, [])
            indexed = event.model_copy(
                update={"index": self._reserve_event_index(room_id)}, deep=True
            )
            self._events[indexed.id] = indexed
            events.append(indexed.id)
            if indexed.idempotency_key:
                self._idempotency.setdefault(room_id, {})[
                    (indexed.source.channel_id, indexed.idempotency_key)
                ] = indexed.id
            return indexed

    async def commit_event(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        target: RoomEvent | None = None,
        activity: bool = True,
    ) -> RoomEvent:
        """Atomic commit (RFC §10.1 step 12 / §14.3): index, store, and bump the
        room counters under the per-room lock as one critical section, so the
        timeline and the counters can never diverge; a change's *target*
        is written in it, or the whole commit fails when the room no longer
        holds it. Only an *activity* commit moves ``last_activity_at`` and
        resumes a PAUSED room (RFC §5.1)."""
        async with self._lock_for(room_id):
            if target is not None:
                stored = self._events.get(target.id)
                if stored is None or stored.room_id != room_id:
                    raise target_gone(target)
                self._events[target.id] = target.model_copy(deep=True)
            if target is not None and deletes_edits(event, target):
                self._mark_edits_deleted_now(room_id, target.id)
            events = self._room_events.setdefault(room_id, [])
            index = self._reserve_event_index(room_id)
            indexed = event.model_copy(update={"index": index}, deep=True)
            self._events[indexed.id] = indexed
            events.append(indexed.id)
            if indexed.idempotency_key:
                self._idempotency.setdefault(room_id, {})[
                    (indexed.source.channel_id, indexed.idempotency_key)
                ] = indexed.id
            room = self._rooms.get(room_id)
            if room is not None:
                self._rooms[room_id] = room.model_copy(
                    update={
                        # "event_count += 1" (RFC §10.1 step 12), not a recount.
                        # len(events) would be free here, but it would make this
                        # store disagree with a persistent one after a deletion:
                        # the counter is best-effort and the two must drift the
                        # same way, or a test passing on one fails on the other.
                        "event_count": room.event_count + 1,
                        "latest_index": index,
                        **committed_state(room, activity),
                    }
                )
            return indexed

    # Binding operations

    async def add_binding(self, binding: ChannelBinding) -> ChannelBinding:
        self._bindings.setdefault(binding.room_id, {})[binding.channel_id] = binding
        self._index_participant(binding.participant_id, binding.room_id)
        return binding

    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        binding = self._bindings.get(room_id, {}).get(channel_id)
        return binding.model_copy(deep=True) if binding is not None else None

    async def binding_exists(self, room_id: str, channel_id: str) -> bool:
        # Inbound routing asks on every message: a key lookup, not the deep
        # copy get_binding pays for a binding nobody reads.
        return channel_id in self._bindings.get(room_id, {})

    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        self._bindings.setdefault(binding.room_id, {})[binding.channel_id] = binding
        self._index_participant(binding.participant_id, binding.room_id)
        return binding

    async def remove_binding(self, room_id: str, channel_id: str) -> bool:
        room_bindings = self._bindings.get(room_id, {})
        if channel_id not in room_bindings:
            return False
        removed = room_bindings.pop(channel_id)
        self._unindex_participant(removed.participant_id, room_id)
        return True

    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        return [b.model_copy(deep=True) for b in self._bindings.get(room_id, {}).values()]

    # Participant operations

    async def add_participant(self, participant: Participant) -> Participant:
        self._participants.setdefault(participant.room_id, {})[participant.id] = participant
        self._index_participant(participant.id, participant.room_id)
        return participant

    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        participant = self._participants.get(room_id, {}).get(participant_id)
        return participant.model_copy(deep=True) if participant is not None else None

    async def update_participant(self, participant: Participant) -> Participant:
        self._participants.setdefault(participant.room_id, {})[participant.id] = participant
        self._index_participant(participant.id, participant.room_id)
        return participant

    async def list_participants(self, room_id: str) -> list[Participant]:
        return [p.model_copy(deep=True) for p in self._participants.get(room_id, {}).values()]

    # Read tracking

    def _room_indexes(self, room_id: str) -> list[int]:
        return [
            self._events[eid].index
            for eid in self._room_events.get(room_id, [])
            if eid in self._events
        ]

    def _raise_read_marker(self, room_id: str, channel_id: str, index: int) -> None:
        markers = self._read_markers.setdefault(room_id, {})
        markers[channel_id] = max(index, markers.get(channel_id, index))

    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        event = self._events.get(event_id)
        if event is not None and event.room_id == room_id:
            self._raise_read_marker(room_id, channel_id, event.index)

    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        indexes = self._room_indexes(room_id)
        if indexes:
            self._raise_read_marker(room_id, channel_id, max(indexes))

    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        indexes = self._room_indexes(room_id)
        last_read = self._read_markers.get(room_id, {}).get(channel_id)
        if last_read is None:
            return len(indexes)
        return sum(1 for index in indexes if index > last_read)

    async def list_read_markers(self, room_id: str) -> dict[str, int]:
        return dict(self._read_markers.get(room_id, {}))

    # Identity operations

    async def create_identity(self, identity: Identity) -> Identity:
        scope = identity.organization_id or ""
        keys = [
            (ch_type, addr, scope)
            for ch_type, addresses in identity.channel_addresses.items()
            for addr in addresses
        ]
        for key in keys:
            if self._address_index.get(key, identity.id) != identity.id:
                raise address_linked(key[0])
        self._identities[identity.id] = identity
        for key in keys:
            self._address_index[key] = identity.id
        return identity

    async def get_identity(self, identity_id: str) -> Identity | None:
        identity = self._identities.get(identity_id)
        return identity.model_copy(deep=True) if identity is not None else None

    async def resolve_identity(
        self, channel_type: str, address: str, organization_id: str | None = None
    ) -> Identity | None:
        identity_id = self._address_index.get((channel_type, address, organization_id or ""))
        if identity_id is None:
            return None
        identity = self._identities.get(identity_id)
        return identity.model_copy(deep=True) if identity is not None else None

    async def link_address(
        self,
        identity_id: str,
        channel_type: str,
        address: str,
        organization_id: str | None = None,
    ) -> None:
        identity = link_target(self._identities.get(identity_id), identity_id, organization_id)
        # The empty string stands for "no organization" so the key shape is the
        # same either way — and so an unscoped registration cannot be found by
        # a scoped lookup, which is the isolation this key exists for.
        key = (channel_type, address, organization_id or "")
        if self._address_index.get(key, identity_id) != identity_id:
            raise address_linked(channel_type)
        current = identity.channel_addresses.get(channel_type, [])
        if address not in current:
            new_addresses = {**identity.channel_addresses, channel_type: [*current, address]}
            self._identities[identity_id] = identity.model_copy(
                update={"channel_addresses": new_addresses}
            )
        self._address_index[key] = identity_id

    async def unlink_address(
        self,
        identity_id: str,
        channel_type: str,
        address: str,
        organization_id: str | None = None,
    ) -> None:
        identity = link_target(self._identities.get(identity_id), identity_id, organization_id)
        key = (channel_type, address, organization_id or "")
        if self._address_index.get(key) == identity_id:
            del self._address_index[key]
        addresses = without_address(identity.channel_addresses, channel_type, address)
        if addresses != identity.channel_addresses:
            self._identities[identity_id] = identity.model_copy(
                update={"channel_addresses": addresses}
            )

    # Verification operations (RFC §11.7)

    async def get_verification(
        self, identity_id: str, organization_id: str | None = None
    ) -> Verification | None:
        verification = self._verifications.get((identity_id, organization_id or ""))
        return verification.model_copy(deep=True) if verification is not None else None

    async def put_verification(self, verification: Verification) -> Verification:
        key = (verification.identity_id, verification.organization_id or "")
        self._verifications[key] = verification.model_copy(deep=True)
        return verification

    async def list_verifications_in_force(self) -> list[Verification]:
        return [
            v.model_copy(deep=True)
            for v in self._verifications.values()
            if v.status == VerificationStatus.VERIFIED and v.expires_at is not None
        ]

    async def create_verification_request(
        self, request: VerificationRequest
    ) -> VerificationRequest:
        self._verification_requests[request.id] = request.model_copy(deep=True)
        return request

    async def get_verification_request(self, request_id: str) -> VerificationRequest | None:
        request = self._verification_requests.get(request_id)
        return request.model_copy(deep=True) if request is not None else None

    async def update_verification_request(
        self, request: VerificationRequest
    ) -> VerificationRequest:
        self._verification_requests[request.id] = request.model_copy(deep=True)
        return request

    # Task operations

    async def add_task(self, task: Task) -> Task:
        self._tasks[task.id] = task
        self._room_tasks.setdefault(task.room_id, []).append(task.id)
        return task

    async def get_task(self, task_id: str) -> Task | None:
        task = self._tasks.get(task_id)
        return task.model_copy(deep=True) if task is not None else None

    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        task_ids = self._room_tasks.get(room_id, [])
        tasks = [self._tasks[tid] for tid in task_ids if tid in self._tasks]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return [t.model_copy(deep=True) for t in tasks]

    async def update_task(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    # Observation operations

    async def add_observation(self, observation: Observation) -> Observation:
        self._observations[observation.id] = observation
        self._room_observations.setdefault(observation.room_id, []).append(observation.id)
        return observation

    async def list_observations(self, room_id: str) -> list[Observation]:
        obs_ids = self._room_observations.get(room_id, [])
        return [
            self._observations[oid].model_copy(deep=True)
            for oid in obs_ids
            if oid in self._observations
        ]


def _changes_one_of(event: RoomEvent, event_ids: set[str]) -> bool:
    """Whether *event* is an edit or a delete of one of *event_ids*."""
    content = event.content
    return isinstance(content, EditContent | DeleteContent) and (
        content.target_event_id in event_ids
    )


def _matches(room: Room, status: str | None, organization_id: str | None) -> bool:
    """Whether a routing lookup keeps *room*: of *status* and of
    *organization_id* when either is named (RFC §10.4, §17.2)."""
    if status is not None and room.status.value != status:
        return False
    return organization_id is None or room.organization_id == organization_id


def _recency(room: Room) -> tuple[float, str]:
    """A room's place in creation order: created_at, then id.

    Compared as timestamps, as SQLite stores them: a naive created_at beside
    an aware one would make the datetimes themselves refuse.
    """
    return room.created_at.timestamp(), room.id
