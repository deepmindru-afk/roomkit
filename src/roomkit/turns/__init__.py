"""Turn execution behind a turn backend (RFC §10.6).

Every turn of an intelligence channel runs through a :class:`TurnBackend`
between the room owner and a :class:`TurnExecutor`, as serialized messages.
The default :class:`InMemoryTurnBackend` keeps it in the owner's process.
"""

from __future__ import annotations

from roomkit.core._turn_scope import current_turn_id
from roomkit.turns.base import ExecutorLink, OwnerTools, TurnBackend, TurnLink, TurnTimeouts
from roomkit.turns.executor import TurnExecutor
from roomkit.turns.host import TurnHost, current_turn_host
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.turns.owner import TurnClient
from roomkit.turns.wire import (
    ChannelCall,
    ChannelControl,
    ChannelResult,
    OwnerTool,
    PreviousTurn,
    RoomAttached,
    RoomDetached,
    TurnRequest,
)

__all__ = [
    "ChannelCall",
    "ChannelControl",
    "ChannelResult",
    "ExecutorLink",
    "InMemoryTurnBackend",
    "OwnerTool",
    "OwnerTools",
    "PreviousTurn",
    "RoomAttached",
    "RoomDetached",
    "TurnBackend",
    "TurnClient",
    "TurnExecutor",
    "TurnHost",
    "TurnLink",
    "TurnRequest",
    "TurnTimeouts",
    "current_turn_host",
    "current_turn_id",
]
