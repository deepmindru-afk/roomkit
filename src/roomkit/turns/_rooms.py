"""What each side of the turn backend records per room and channel (RFC
§10.6.2, §10.6.7): the owner its previous turn, the backend where the room's
state is, the executor the turn that left it.

A record falls out once a side keeps :data:`ROOMS_KEPT` newer ones. Every
side reads a missing record as a state it cannot vouch for, so a room that
was quiet that long starts over (its state dropped and rebuilt, RFC
§10.6.2) and never resumes a stale one.
"""

from __future__ import annotations

from collections import OrderedDict

ROOMS_KEPT = 10_000
"""How many (channel, room) pairs a side keeps a record for."""

ROOM_STATE_KEY = "roomkit:channel_state:"
"""The prefix of the room-metadata key under which the owner keeps what the
room holds for a channel (RFC §6.4): an ACP session's config values."""


def room_state_key(channel_id: str) -> str:
    """The room-metadata key of what the room holds for *channel_id*."""
    return f"{ROOM_STATE_KEY}{channel_id}"


def stored_room_state(metadata: dict[str, object], channel_id: str) -> dict[str, object]:
    """What *metadata* holds for *channel_id*; empty when it holds nothing
    readable."""
    state = metadata.get(room_state_key(channel_id))
    return dict(state) if isinstance(state, dict) else {}


class RoomRecords[T]:
    """One record per (channel, room), the least recently written falling out
    past *kept*."""

    def __init__(self, kept: int = ROOMS_KEPT) -> None:
        self._kept = kept
        self._records: OrderedDict[tuple[str, str], T] = OrderedDict()

    def get(self, channel_id: str, room_id: str) -> T | None:
        return self._records.get((channel_id, room_id))

    def put(self, channel_id: str, room_id: str, record: T) -> None:
        key = (channel_id, room_id)
        self._records[key] = record
        self._records.move_to_end(key)
        while len(self._records) > self._kept:
            self._records.popitem(last=False)

    def pop(self, channel_id: str, room_id: str) -> T | None:
        return self._records.pop((channel_id, room_id), None)
