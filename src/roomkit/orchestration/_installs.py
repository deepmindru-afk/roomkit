"""Where a voice strategy installed for a room sets its tool up (RFC §19.7).

A voice strategy (a supervisor or a loop in async delivery) hands its work to
the realtime channel the room's callers talk to. The strategy is installed when
the room is created, before the host attaches that channel, so its tool is set
up for the room on every realtime channel of the kit: a room's sessions may
open on any of them, and the tool is declared in that room's sessions only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from roomkit.channels._tool_registry import ToolEntry

if TYPE_CHECKING:
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration")


def realtime_channels(kit: RoomKit) -> list[RealtimeVoiceChannel]:
    """The realtime voice channels registered on *kit*."""
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel

    return [ch for ch in kit.channels.values() if isinstance(ch, RealtimeVoiceChannel)]


def set_up_for_voice_room(
    kit: RoomKit,
    room_id: str,
    owner: object,
    entry_for: Callable[[RealtimeVoiceChannel], ToolEntry],
) -> None:
    """Set up, on every realtime channel of *kit*, the tool *entry_for* builds
    for that channel, served in *room_id*."""
    channels = realtime_channels(kit)
    if not channels:
        logger.warning("async_delivery=True but no RealtimeVoiceChannel found")
    for channel in channels:
        channel._registry.register(entry_for(channel), room_id=room_id, owner=owner)
