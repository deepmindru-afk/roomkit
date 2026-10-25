"""The hooks on a trigger a turn fires registered on the kit (RFC §10.6.6).

Registered on the kit, such a hook is its own executor's: it fires for the
turns of the channels the kit holds live, and for the calls of the kit's
process outside a turn (realtime voice, conference). A channel the kit holds
by its declaration alone runs its turns on another executor, where the hook
would never fire: such a hook is refused while such a channel could fire it,
and such a channel while such a hook is registered, in either order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from roomkit.channels.declared import DeclaredChannel
from roomkit.core.hooks import TYPED_TURN_TRIGGERS

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine, HookRegistration
    from roomkit.models.enums import HookTrigger


class KitTurnHooks:
    """The kit's turn hooks, given to its own executor's engine.

    The engine's side, synchronous and trusted: it holds no store and reads
    no room. The kit's :meth:`RoomKit.add_room_hook` and
    :meth:`RoomKit.remove_room_hook` read the room with the caller's
    organization (RFC §17.2) before they reach it.

    Args:
        engine: The kit's own executor's hook engine.
        channels: The channels the kit registers, by id.
    """

    def __init__(self, engine: HookEngine, channels: Mapping[str, Channel]) -> None:
        self._engine = engine
        self._channels = channels
        # What the kit gave its executor: each hook, and its room (None: global).
        self._given: list[tuple[HookRegistration, str | None]] = []

    def register(self, hook: HookRegistration) -> HookRegistration:
        self._refuse_where_declared(hook, global_hook=True)
        # The engine may name the hook (RFC §9.1): keep what it holds.
        registered = self._engine.register(hook)
        self._given.append((registered, None))
        return registered

    def add_room_hook(self, room_id: str, hook: HookRegistration) -> HookRegistration:
        self._refuse_where_declared(hook, global_hook=False)
        registered = self._engine.add_owner_room_hook(room_id, hook)
        self._given.append((registered, room_id))
        return registered

    def remove_global_hook(self, name: str) -> bool:
        return self._remove(name, None)

    def remove_room_hook(self, room_id: str, name: str) -> bool:
        return self._remove(name, room_id)

    def global_hook_names(self) -> list[str]:
        return [hook.name for hook, given_room in self._still_given() if given_room is None]

    def room_hook_names(self, room_id: str) -> list[str]:
        return [hook.name for hook, given_room in self._still_given() if given_room == room_id]

    @property
    def engine(self) -> HookEngine:
        return self._engine

    @property
    def fail_closed(self) -> frozenset[HookTrigger]:
        return self._engine.FAIL_CLOSED_TRIGGERS

    def set_fail_closed(self, triggers: frozenset[HookTrigger]) -> None:
        self._engine.FAIL_CLOSED_TRIGGERS = triggers

    def refuse_declaration(self, channel: DeclaredChannel) -> None:
        """Refuse to hold *channel* by its declaration while a turn hook the
        kit registered could fire for it: its turns run elsewhere.

        Raises:
            ValueError: such a hook is registered.
        """
        for hook, room_id in self._still_given():
            if room_id is None and not _could_fire(hook, channel):
                continue
            where = "for a room" if room_id is not None else "globally"
            raise ValueError(
                f"Channel {channel.channel_id!r} is held by its declaration alone, its turns "
                f"on another executor, where hook {hook.name!r} on {hook.trigger}, "
                f"registered {where} on the kit, never fires (RFC §10.6.6): register that "
                "hook on the executor that runs the channel (executor.hook(...))"
                + ("" if room_id is not None else ", or leave the channel out of channel_ids")
            )

    def _refuse_where_declared(self, hook: HookRegistration, *, global_hook: bool) -> None:
        """Refuse *hook* when a channel the kit holds by its declaration
        could fire it (a room hook: any such channel, the room's channels
        being known only later)."""
        for channel in self._channels.values():
            if not isinstance(channel, DeclaredChannel):
                continue
            if global_hook and not _could_fire(hook, channel):
                continue
            raise ValueError(
                f"Hook {hook.name!r} on {hook.trigger}: channel {channel.channel_id!r} is held "
                "by its declaration alone, its turns on another executor, where a hook on "
                "the kit never fires (RFC §10.6.6): register it on that executor "
                "(executor.hook(...))"
                + (", or leave the channel out of channel_ids" if global_hook else "")
            )

    def _still_given(self) -> list[tuple[HookRegistration, str | None]]:
        """What the kit gave its executor and the executor still holds (one
        removed on the executor's engine itself is gone)."""
        self._given = [given for given in self._given if self._engine.holds(given[0])]
        return self._given

    def _remove(self, name: str, room_id: str | None) -> bool:
        for given in self._still_given():
            hook, given_room = given
            if hook.name == name and given_room == room_id:
                self._given.remove(given)
                return self._engine.remove_hook(hook, room_id)
        return False


def _could_fire(hook: HookRegistration, channel: Channel) -> bool:
    """Whether *hook*'s channel filter lets *channel*'s turns fire it: its
    ``channel_types`` filters only where the payload carries the agent's
    type (RFC §9.1)."""
    types_filter = hook.channel_types is not None and hook.trigger in TYPED_TURN_TRIGGERS
    return (hook.channel_ids is None or channel.channel_id in hook.channel_ids) and (
        not types_filter or channel.channel_type in (hook.channel_types or ())
    )
