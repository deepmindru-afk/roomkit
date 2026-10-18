"""Where a room's turns of a channel go on the Redis turn backend (RFC §10.6.7).

The room owner chooses: the executor that holds the room's state for the
channel, while it lives and takes the turn, else one that takes it, one of
the owner's own process first (carried in the process), else one alive
elsewhere, the rooms no executor holds yet spread over them. The holder is a
key every owner process reads, so a room's turns follow its session state
whichever owner submits them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from roomkit.core.exceptions import TurnLinkError

if TYPE_CHECKING:
    from roomkit.turns._redis_presence import LiveExecutors
    from roomkit.turns.executor import TurnExecutor

# How long a room's holder is kept once no turn renews it.
_HOLDER_TTL_S = 7 * 24 * 3600


class Elsewhere(NamedTuple):
    """An executor of another process."""

    executor_id: str
    node: str


class Placement:
    """The executors of this process, and where each turn goes.

    Args:
        client: The ``redis.asyncio`` client.
        prefix: The keys' prefix.
        node: This process's node id.
        live: The executors alive, as the owner reads them.
        loop_back: Tests only: this process's executors are reached across
            Redis too, as executors elsewhere would be.
    """

    def __init__(
        self, client: Any, prefix: str, node: str, live: LiveExecutors, *, loop_back: bool
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._node = node
        self._live = live
        self._loop_back = loop_back
        self.here: list[TurnExecutor] = []
        """The executors this process serves."""
        self._next = 0

    async def place(
        self, channel_id: str, room_id: str, room_hooks: Sequence[str] = ()
    ) -> TurnExecutor | Elsewhere:
        """The executor a room's turn or channel call of *channel_id* goes
        to, recorded as the one holding the room's state from now."""
        placed = await self.hold(channel_id, room_id, room_hooks)
        executor_id = placed.executor_id
        await self._client.set(
            self._holder_key(channel_id, room_id), executor_id, ex=_HOLDER_TTL_S
        )
        return placed

    async def hold(
        self, channel_id: str, room_id: str, room_hooks: Sequence[str] = ()
    ) -> TurnExecutor | Elsewhere:
        """The executor holding the room's state of *channel_id* while it
        lives and takes it, else one that takes it, one of this process
        first; the holder is left as it was (RFC §10.6.3)."""
        here = {e.executor_id: e for e in self.here if e.takes(channel_id, room_hooks)}
        elsewhere = await self._takers_elsewhere(channel_id, room_hooks, set(here))
        holder = await self._client.get(self._holder_key(channel_id, room_id))
        holder_id = _text(holder) if holder is not None else None
        if self._loop_back and here:
            chosen = here[holder_id] if holder_id in here else next(iter(here.values()))
            return Elsewhere(chosen.executor_id, self._node)
        if holder_id in here:
            return here[holder_id]
        if holder_id in elsewhere:
            return Elsewhere(holder_id, elsewhere[holder_id])
        if here:
            return next(iter(here.values()))
        if not elsewhere:
            missing = f" with room hooks {list(room_hooks)}" if room_hooks else ""
            raise TurnLinkError(
                "no_executor", f"No executor serves channel {channel_id!r}{missing}"
            )
        # A room no executor holds yet: its turns spread over the takers.
        ids = sorted(elsewhere)
        self._next = (self._next + 1) % len(ids)
        return Elsewhere(ids[self._next], elsewhere[ids[self._next]])

    async def forget(self, channel_id: str, room_id: str) -> None:
        """The room detached the channel: nothing holds its state any more."""
        await self._client.delete(self._holder_key(channel_id, room_id))

    async def declaring_elsewhere(self, channel_id: str) -> list[Elsewhere]:
        """The executors of other processes alive that declare *channel_id*."""
        return [
            Elsewhere(executor_id, declared.node)
            for executor_id, declared in (await self._live.current()).items()
            if declared.node != self._node and channel_id in declared.channels
        ]

    async def _takers_elsewhere(
        self, channel_id: str, room_hooks: Sequence[str], here: set[str]
    ) -> dict[str, str]:
        """The executors of other processes alive that take the turn, by id:
        their nodes, read afresh, so one just started counts and one that
        stopped does not."""
        return {
            executor_id: declared.node
            for executor_id, declared in (await self._live.current()).items()
            if declared.node != self._node
            and executor_id not in here
            and declared.takes(channel_id, room_hooks)
        }

    def _holder_key(self, channel_id: str, room_id: str) -> str:
        return f"{self._prefix}:holder:{channel_id}:{room_id}"


def _text(value: Any) -> str:
    return value if isinstance(value, str) else bytes(value).decode()
