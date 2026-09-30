"""What a channel serves and what is installed on it, per room (RFC §19.7, §21.1).

One channel object serves every room it is attached to. The tools it serves
itself and the ones orchestration sets up on it are kept here, one entry per
tool: the definition the model reads, the function that serves a call, who set
it up, and the traits the channel's rules read (exempt from the policy, never
deferred by Tool Search, left out of the usage digest, pure within a turn).
An entry is the channel's for every room, or one room's: a strategy sets up a
room's tools, with that room's configuration, beside the other rooms' and
without touching them. A name is served by one tool in a room, so a second one
is refused when it is given.

The host's own tools stay the host's: its definitions and the handler it gave
serve every name no entry serves. The registry reads their names only, to
refuse an entry that would shadow one.

Two more things orchestration installs live here rather than on the channel
object: the runner that takes a room's turns in place of the agent's own (a
loop, a supervisor's delegation passes), and the source a realtime channel
reads a new session's configuration from (a pipeline's active agent). Nothing
is ever written onto the shared channel for one room: a write made for one room
would be read by every other room the channel serves.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from roomkit.channels._skill_constants import (
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._tool_eviction import REREAD_TOOL
from roomkit.channels._tool_search_constants import (
    TOOL_CALL_TOOL,
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
)
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

ToolServe = Callable[[dict[str, Any]], Any]
"""Serves one call from its arguments; may return an awaitable. The room is the
call's (``current_tool_room_id``, RFC §21.4)."""

TurnRunner = Callable[["RoomEvent", "ChannelBinding", "RoomContext"], Awaitable["ChannelOutput"]]
"""Takes one of a room's turns in place of the channel's own answer."""

SessionSource = Callable[[str], Awaitable[Any]]
"""A realtime session's configuration for a room id, or ``None`` for the channel's."""


class ToolSource(StrEnum):
    """Who set a tool up on the channel."""

    CHANNEL = "channel"
    """One of the channel's own features: skills, Tool Search, the re-read, the planner."""

    ORCHESTRATION = "orchestration"
    """A strategy, a handoff, a delegation or a delegation's result capture."""


@dataclass(frozen=True, slots=True)
class ToolTraits:
    """What the channel's rules read of a tool."""

    exempt: bool = False
    """Only reads or unlocks: escapes the tool policy and skill gating (RFC §21.1)."""

    deferrable: bool = True
    """Tool Search may hide it behind its discovery tools."""

    always_declared: bool = False
    """Declared at every round and in every session of its scope, outside the
    catalogue whose size decides Tool Search, reported ``always`` (RFC §21.1)."""

    in_digest: bool = True
    """Recorded in the room's usage digest and re-revealed as a tool in use."""

    pure: bool = False
    """Reads what cannot change within a turn: an identical repeat says nothing new."""


CHANNEL_TOOL_TRAITS: dict[str, ToolTraits] = {
    TOOL_ACTIVATE_SKILL: ToolTraits(exempt=True, deferrable=False, in_digest=False),
    TOOL_READ_REFERENCE: ToolTraits(exempt=True, deferrable=False, in_digest=False),
    TOOL_RUN_SCRIPT: ToolTraits(deferrable=False, in_digest=False),
    TOOL_FIND_TOOLS: ToolTraits(exempt=True, deferrable=False, in_digest=False, pure=True),
    TOOL_LIST_TOOLS: ToolTraits(exempt=True, deferrable=False, in_digest=False, pure=True),
    # Only carries a call to the tool it names, whom the policy governs.
    TOOL_CALL_TOOL: ToolTraits(deferrable=False, in_digest=False),
    REREAD_TOOL: ToolTraits(exempt=True, deferrable=False, in_digest=False),
    "plan_tasks": ToolTraits(deferrable=False),
}
"""The traits of each tool a channel serves itself, by its name."""

ORCHESTRATION_TRAITS = ToolTraits(deferrable=False, always_declared=True)
"""An orchestration tool: the agent is told to call it, so it is never hidden."""


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """One tool a channel serves: its definition, its server, its origin, its traits."""

    definition: AITool
    serve: ToolServe | None
    """``None`` for a tool the channel runs on a path of its own (a realtime
    channel's skills and Tool Search)."""

    source: ToolSource
    traits: ToolTraits = field(default_factory=ToolTraits)

    @property
    def name(self) -> str:
        return self.definition.name


def schema_tool(schema: dict[str, Any]) -> AITool:
    """The definition a tool schema given as a dict describes."""
    return AITool(
        name=schema["name"],
        description=schema.get("description", ""),
        parameters=schema.get("parameters", {}),
    )


def channel_tool(definition: AITool, serve: ToolServe | None) -> ToolEntry:
    """A tool the channel serves itself, with the traits its name carries."""
    traits = CHANNEL_TOOL_TRAITS.get(definition.name, ToolTraits())
    return ToolEntry(definition, serve, ToolSource.CHANNEL, traits)


def orchestration_tool(definition: AITool, serve: ToolServe, **traits: bool) -> ToolEntry:
    """A tool orchestration sets up: declared always, never deferred."""
    return ToolEntry(
        definition, serve, ToolSource.ORCHESTRATION, replace(ORCHESTRATION_TRAITS, **traits)
    )


class ToolNameCollisionError(ValueError):
    """A tool given under a name the channel already serves in that scope (RFC §21.1)."""


@dataclass(slots=True)
class _Held:
    entry: ToolEntry
    owner: object | None


@dataclass(slots=True)
class _Slot[T]:
    value: T
    owner: object | None


class ChannelRegistry:
    """The tools a channel serves, for all its rooms or one, and what
    orchestration installs on it per room."""

    def __init__(self, channel_id: str, host_names: Callable[[], Iterable[str]]) -> None:
        self._channel_id = channel_id
        # The names the host's own tools carry, read at each registration.
        self._host_names = host_names
        self._channel: dict[str, _Held] = {}
        self._rooms: dict[str, dict[str, _Held]] = {}
        self._turn_runners: dict[str, _Slot[TurnRunner]] = {}
        self._session_source: _Slot[SessionSource] | None = None

    # -- Tools -------------------------------------------------------------

    def register(
        self, entry: ToolEntry, *, room_id: str | None = None, owner: object | None = None
    ) -> None:
        """Serve *entry* in *room_id*, or in every room when it is ``None``.

        Refuses a name something else already serves where the entry would be
        declared (:class:`ToolNameCollisionError`); the same *owner* registering the
        same name in the same scope again replaces its own entry.
        """
        held = (self._channel if room_id is None else self._rooms.get(room_id, {})).get(entry.name)
        if held is None or owner is None or held.owner is not owner:
            clash = self._clash(entry.name, room_id, host=entry.source is ToolSource.ORCHESTRATION)
            if clash is not None:
                raise ToolNameCollisionError(
                    f"Tool {entry.name!r} is already {clash} on channel {self._channel_id!r}: "
                    "rename one of them (RFC §21.1)"
                )
        scope = self._channel if room_id is None else self._rooms.setdefault(room_id, {})
        scope[entry.name] = _Held(entry, owner)

    def _clash(self, name: str, room_id: str | None, *, host: bool) -> str | None:
        """What already serves *name* where an entry for *room_id* would be declared.

        The host's tools count for an orchestration entry; a host tool under
        one of the channel's own names is refused where the host gives it.
        """
        if host and name in set(self._host_names()):
            return "a tool of the host"
        if name in self._channel:
            return "served by the channel"
        if room_id is None:
            rooms = list(self._rooms.items())
        else:
            rooms = [(room_id, self._rooms.get(room_id, {}))]
        for rid, scope in rooms:
            if name in scope:
                return f"served in room {rid!r}"
        return None

    def unregister(self, name: str, *, room_id: str | None = None, owner: object) -> None:
        """Stop serving the entry *owner* registered under *name* in that scope."""
        scope = self._channel if room_id is None else self._rooms.get(room_id)
        if scope is None:
            return
        held = scope.get(name)
        if held is not None and held.owner is owner:
            del scope[name]
        if room_id is not None and not scope:
            self._rooms.pop(room_id, None)

    def lookup(self, name: str, room_id: str | None) -> ToolEntry | None:
        """The entry serving *name* in *room_id* (the channel's, outside any room)."""
        held = self._rooms.get(room_id, {}).get(name) if room_id is not None else None
        if held is None:
            held = self._channel.get(name)
        return held.entry if held is not None else None

    def entries(self, room_id: str | None, *, source: ToolSource | None = None) -> list[ToolEntry]:
        """The entries declared in *room_id*: the channel's, then the room's,
        each in the order they were registered."""
        held = [*self._channel.values()]
        if room_id is not None:
            held.extend(self._rooms.get(room_id, {}).values())
        return [h.entry for h in held if source is None or h.entry.source is source]

    def names(
        self, room_id: str | None, where: Callable[[ToolTraits], bool] | None = None
    ) -> set[str]:
        """The names of the entries declared in *room_id* whose traits *where* admits."""
        return {e.name for e in self.entries(room_id) if where is None or where(e.traits)}

    def traits(self, name: str, room_id: str | None = None) -> ToolTraits | None:
        """The traits of the tool serving *name* in *room_id*, if an entry serves it."""
        entry = self.lookup(name, room_id)
        return entry.traits if entry is not None else None

    def serves_orchestration(self) -> bool:
        """Whether orchestration set up any tool here, for any room."""
        scopes = [self._channel, *self._rooms.values()]
        return any(
            h.entry.source is ToolSource.ORCHESTRATION for scope in scopes for h in scope.values()
        )

    # -- What orchestration installs per room ------------------------------

    def set_turn_runner(self, room_id: str, runner: TurnRunner, *, owner: object) -> None:
        """Take *room_id*'s turns with *runner*; one runner per room."""
        held = self._turn_runners.get(room_id)
        if held is not None and held.owner is not owner:
            raise ValueError(
                f"Room {room_id!r} already has its turns taken by another strategy "
                f"on channel {self._channel_id!r}"
            )
        self._turn_runners[room_id] = _Slot(runner, owner)

    def turn_runner(self, room_id: str | None) -> TurnRunner | None:
        """The runner taking *room_id*'s turns, if one was installed there."""
        held = self._turn_runners.get(room_id) if room_id is not None else None
        return held.value if held is not None else None

    def set_session_source(self, source: SessionSource, *, owner: object) -> None:
        """Read a new realtime session's configuration from *source*."""
        held = self._session_source
        if held is not None and held.owner is not owner:
            raise ValueError(
                f"Channel {self._channel_id!r} already reads its sessions' configuration "
                "from another source"
            )
        self._session_source = _Slot(source, owner)

    @property
    def session_source(self) -> SessionSource | None:
        """Where a new session's configuration comes from, when orchestration set one."""
        return self._session_source.value if self._session_source is not None else None
