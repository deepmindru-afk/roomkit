"""Turn execution behind a turn backend (RFC §10.6).

Every turn of an intelligence channel runs through a :class:`TurnBackend`
between the room owner and a :class:`TurnExecutor`, as serialized messages.
The default :class:`InMemoryTurnBackend` keeps it in the owner's process.
"""

from __future__ import annotations

from roomkit.turns.base import ExecutorLink, OwnerTools, TurnBackend, TurnLink, TurnTimeouts
from roomkit.turns.executor import TurnExecutor
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.turns.owner import TurnClient
from roomkit.turns.wire import OwnerTool, TurnRequest

__all__ = [
    "ExecutorLink",
    "InMemoryTurnBackend",
    "OwnerTool",
    "OwnerTools",
    "TurnBackend",
    "TurnClient",
    "TurnExecutor",
    "TurnLink",
    "TurnRequest",
    "TurnTimeouts",
]
