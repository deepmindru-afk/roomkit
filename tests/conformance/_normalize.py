"""What a run produced, normalized as the suite compares it.

roomkit-specs ``conformance/README.md`` ("Normalization") defines it: a room
the run created is named ``$r1``, ``$r2``... in the order of its
``room_created``; an event of a timeline is named ``<room>#<rank>``, its rank
among the room's events once the lifecycle records (§5.1, which an
implementation MAY write) are left out, and its ``index`` is that rank; any
other event id is ``$e1``, ``$e2``... by first appearance; a timestamp is
``@1``, ``@2``... by its order among the run's instants.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_INSTANT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
_ID_KEYS = frozenset({"id", "event_id", "responds_to", "parent_event_id", "target_event_id"})


def is_lifecycle_record(event: dict[str, Any]) -> bool:
    """A record the framework writes of a lifecycle change (§5.1): its source
    is the framework's, its visibility ``internal``, its content a system one."""
    source = event.get("source") or {}
    content = event.get("content") or {}
    return (
        source.get("channel_id") == "system"
        and event.get("visibility") == "internal"
        and content.get("type") == "system"
    )


@dataclass
class Observed:
    """What a run produced, in the RFC's serialized form, before normalization."""

    literal_rooms: set[str]
    created_rooms: list[str] = field(default_factory=list)
    timelines: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    rooms: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    framework_events: list[dict[str, Any]] = field(default_factory=list)
    hooks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    hook_order: list[str] = field(default_factory=list)
    deliveries: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    turns: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    observations: dict[str, list[Any]] = field(default_factory=dict)
    tasks: dict[str, list[Any]] = field(default_factory=dict)
    steps: list[Any] = field(default_factory=list)
    actions: dict[str, list[Any]] = field(default_factory=dict)


class Normalizer:
    """Rewrites the ids, indexes and instants of one run's records."""

    def __init__(self, observed: Observed) -> None:
        self._rooms: dict[str, str] = {room: room for room in observed.literal_rooms}
        for room_id in observed.created_rooms:
            if room_id not in self._rooms:
                self._rooms[room_id] = (
                    f"$r{sum(a.startswith('$r') for a in self._rooms.values()) + 1}"
                )
        self._events: dict[str, str] = {}
        self._kept: dict[str, list[int]] = {}
        for room_id, events in observed.timelines.items():
            kept = [event for event in events if not is_lifecycle_record(event)]
            self._kept[room_id] = [event["index"] for event in kept]
            for rank, event in enumerate(kept):
                self._events[event["id"]] = f"{self.room(room_id)}#{rank}"
        self._others: dict[str, str] = {}
        self._instants = sorted(set(_instants(observed)))
        # The ids of no timeline are named in the order the step outcomes,
        # then the framework events, show them first.
        self.value(observed.steps)
        self.value(observed.framework_events)

    def room(self, room_id: str) -> str:
        """The alias of a room."""
        return self._rooms.get(room_id, room_id)

    def rank(self, room_id: str, index: int) -> int:
        """How many of the room's kept events come before *index*."""
        kept = self._kept.get(room_id)
        return index if kept is None else bisect.bisect_left(kept, index)

    def value(self, value: Any, key: str | None = None) -> Any:
        """*value* with its ids, indexes and instants normalized."""
        if isinstance(value, dict):
            return self._object(value)
        if isinstance(value, list):
            return [self.value(item, key) for item in value]
        if isinstance(value, str):
            return self._string(value, key)
        return value

    def _object(self, value: dict[str, Any]) -> dict[str, Any]:
        room_id = value.get("room_id")
        is_event = isinstance(room_id, str) and "source" in value and "index" in value
        result: dict[str, Any] = {}
        for key, item in value.items():
            if is_event and key == "index" and isinstance(item, int):
                result[key] = self.rank(room_id, item)
            else:
                result[key] = self.value(item, key)
        return result

    def _string(self, value: str, key: str | None) -> str:
        if value in self._rooms:
            return self._rooms[value]
        if value in self._events:
            return self._events[value]
        if value in self._others:
            return self._others[value]
        if key in _ID_KEYS:
            if value not in self._others:
                self._others[value] = f"$e{len(self._others) + 1}"
            return self._others[value]
        if _INSTANT.match(value):
            return f"@{bisect.bisect_left(self._instants, _parse(value)) + 1}"
        return value


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _instants(observed: Observed) -> list[datetime]:
    found: list[datetime] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and _INSTANT.match(value):
            found.append(_parse(value))

    walk(
        [
            observed.steps,
            observed.actions,
            observed.framework_events,
            observed.hooks,
            observed.turns,
            observed.deliveries,
            observed.timelines,
            observed.rooms,
            observed.bindings,
            observed.observations,
            observed.tasks,
        ]
    )
    return found
