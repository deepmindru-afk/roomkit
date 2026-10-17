"""The turn runners a room's owner keeps (RFC §10.6.1, §19.7).

A strategy that takes an agent's turns in a room (a loop's produce and
review, a supervisor's delegation passes) is kept here, by the room's owner,
per channel and room, never on the shared channel object: the runner runs on
the owner, and each generation it asks of the agent is a turn of its own
through the turn backend.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

TurnRunner = Callable[["RoomEvent", "ChannelBinding", "RoomContext"], Awaitable["ChannelOutput"]]
"""What takes a channel's turns in a room in place of the channel's own answer."""


@dataclass(slots=True)
class _Held:
    runner: TurnRunner
    owner: object


class TurnRunners:
    """The runners of the strategies installed in this owner's rooms."""

    def __init__(self) -> None:
        self._held: dict[tuple[str, str], _Held] = {}

    def take(self, room_id: str, channel_id: str, runner: TurnRunner, *, owner: object) -> None:
        """Take *channel_id*'s turns in *room_id* with *runner*; one runner per
        channel and room, which only the install that set it replaces."""
        held = self._held.get((room_id, channel_id))
        if held is not None and held.owner is not owner:
            raise ValueError(
                f"Room {room_id!r} already has the turns of {channel_id!r} taken by "
                "another strategy"
            )
        self._held[(room_id, channel_id)] = _Held(runner, owner)

    def get(self, room_id: str | None, channel_id: str) -> TurnRunner | None:
        """The runner taking *channel_id*'s turns in *room_id*, if one does."""
        held = self._held.get((room_id, channel_id)) if room_id is not None else None
        return held.runner if held is not None else None

    def release(self, room_id: str, channel_id: str) -> None:
        """Stop taking *channel_id*'s turns in *room_id* (an uninstall, RFC §19.7)."""
        self._held.pop((room_id, channel_id), None)

    def channels(self, room_id: str) -> set[str]:
        """The channels whose turns a runner takes in *room_id*."""
        return {channel_id for rid, channel_id in self._held if rid == room_id}
