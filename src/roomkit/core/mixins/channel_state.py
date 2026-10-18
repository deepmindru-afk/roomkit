"""ChannelStateMixin — what a room keeps for a channel, and its per-room verbs
(RFC §10.6.3, §6.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import ChannelNotFoundError, ChannelNotRegisteredError
from roomkit.models.enums import ChannelCategory
from roomkit.turns._error_codec import rebuild_error
from roomkit.turns._rooms import room_state_key, stored_room_state
from roomkit.turns.wire import ChannelCall, RoomDetached

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.models.channel import ChannelBinding
    from roomkit.store.base import ConversationStore
    from roomkit.turns.owner import TurnClient


class ChannelStateMixin:
    """``channel_call``, and the room's record of what it keeps for a channel:
    written from a verb's answer, forgotten when the channel is detached."""

    _store: ConversationStore
    _channels: dict[str, Channel]
    _turns: TurnClient
    get_room: Any  # see RoomLifecycleMixin

    async def channel_call(
        self,
        channel_id: str,
        room_id: str,
        verb: str,
        *,
        organization_id: str | None = None,
        **arguments: Any,
    ) -> Any:
        """Run a per-room verb of a channel attached to a room, on the
        executor holding the room's state for it (RFC §10.6.3): its result.

        An ACP agent's are its session's: ``set_config_option`` (``config_id``,
        ``value``), ``close_session``, ``session_config`` and
        ``config_options``. What the room keeps for the channel, when the verb
        changes it (the values a host set on the session), is stored with the
        room (RFC §6.4), so the room's next executor finds it::

            await kit.channel_call("claude", room.id, "set_config_option",
                                   config_id="model", value="opus")

        *organization_id* scopes the call to one tenant (RFC §17.2); a room
        belonging to another organization is reported as not found.

        Raises:
            RoomNotFoundError: no room *room_id*.
            ChannelNotRegisteredError: no channel *channel_id*.
            ChannelNotFoundError: the channel is not attached to the room.
            ValueError: a verb the channel does not have.
            TurnLinkError: ``no_executor``, no executor serves the channel.
        """
        room = await self.get_room(room_id, organization_id=organization_id)
        if channel_id not in self._channels:
            raise ChannelNotRegisteredError(f"Channel {channel_id} not registered")
        if await self._store.get_binding(room_id, channel_id) is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not attached to room {room_id}")
        call = ChannelCall(
            channel_id=channel_id,
            room_id=room_id,
            verb=verb,
            arguments=arguments,
            room_state=stored_room_state(room.metadata, channel_id),
        )
        answer = await self._turns.control(call)
        if answer is None:
            return None
        if answer.room_state is not None:
            await self._store_room_state(room_id, channel_id, answer.room_state)
        if answer.error is not None:
            raise rebuild_error(answer.error)
        return answer.result

    async def _store_room_state(
        self, room_id: str, channel_id: str, state: dict[str, Any]
    ) -> None:
        """Keep *state* with the room as what it holds for *channel_id*
        (empty: nothing)."""
        key = room_state_key(channel_id)
        if state:
            await self._store.patch_room_metadata(room_id, {key: state})
        else:
            await self._store.patch_room_metadata(room_id, {}, unset=[key])

    async def _on_room_detached(self, room_id: str, binding: ChannelBinding) -> None:
        """Let the channel of *binding* let go of the room: an intelligence
        channel on every executor declaring it, the room forgetting what it
        kept for the channel even when the executors could not be told (RFC
        §10.6.3, §6.4)."""
        channel_id = binding.channel_id
        if binding.category is not ChannelCategory.INTELLIGENCE:
            channel = self._channels.get(channel_id)
            if channel is not None:
                await channel.on_room_detached(room_id)
            return
        try:
            await self._turns.control(RoomDetached(channel_id=channel_id, room_id=room_id))
        finally:
            room = await self.get_room(room_id)
            if room_state_key(channel_id) in room.metadata:
                await self._store_room_state(room_id, channel_id, {})
