"""Which executors live, and what they declare, on Redis (RFC §10.6.7, §10.6.8).

An executor's process writes what each of its executors declares (the
channels, the names of its room hooks, its node) to a hash, and beats for
them from a thread of its own, with a connection of its own: its turns' CPU
never stalls the beat. A beat scores the executor in a sorted set with the
Redis server's clock, so processes on several machines agree on its age. The
room owner reads the set each heartbeat interval: an executor is alive while
its last beat is younger than the heartbeat timeout, and the turns it took
count its beats as messages (RFC §10.6.8).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.turns.executor import TurnExecutor

logger = logging.getLogger("roomkit.turns")

# KEYS: the alive set, the node's inbox. ARGV: the inbox's ttl in ms, then
# the executor ids. Scores each id with the server's time, keeps the inbox.
_BEAT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
for i = 2, #ARGV do
  redis.call('ZADD', KEYS[1], now, ARGV[i])
end
redis.call('PEXPIRE', KEYS[2], ARGV[1])
return now
"""

# KEYS: the alive set, the declarations. ARGV: the age past which an
# executor is forgotten, in ms. The server's time, then each id and score.
_ALIVE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local stale = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now - tonumber(ARGV[1]))
for _, id in ipairs(stale) do
  redis.call('ZREM', KEYS[1], id)
  redis.call('HDEL', KEYS[2], id)
end
local alive = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
table.insert(alive, 1, now)
return alive
"""


@dataclass(frozen=True, slots=True)
class Declaration:
    """What an executor declares to the owners (RFC §10.6.7)."""

    node: str
    channels: frozenset[str]
    room_hooks: frozenset[str]

    def takes(self, channel_id: str, room_hooks: Iterable[str]) -> bool:
        return channel_id in self.channels and all(h in self.room_hooks for h in room_hooks)

    def dump(self) -> str:
        return json.dumps(
            {
                "node": self.node,
                "channels": sorted(self.channels),
                "room_hooks": sorted(self.room_hooks),
            }
        )

    @classmethod
    def load(cls, raw: Any) -> Declaration:
        data = json.loads(raw)
        return cls(data["node"], frozenset(data["channels"]), frozenset(data["room_hooks"]))


class Registry:
    """The executors' declarations and their beats, on Redis."""

    def __init__(self, client: Any, prefix: str) -> None:
        self._client = client
        self.alive_key = f"{prefix}:alive"
        self.declarations_key = f"{prefix}:executors"
        self._alive_script = client.register_script(_ALIVE)
        self._beat_script = client.register_script(_BEAT)

    async def beat(self, executor_ids: Iterable[str], inbox_key: str, inbox_ttl_ms: int) -> None:
        """Beat once for *executor_ids* from this task: an executor counts as
        alive from its first beat."""
        await self._beat_script(
            keys=[self.alive_key, inbox_key], args=[inbox_ttl_ms, *sorted(executor_ids)]
        )

    async def declare(self, executor_id: str, declaration: Declaration) -> None:
        await self._client.hset(self.declarations_key, executor_id, declaration.dump())

    async def withdraw(self, executor_id: str) -> None:
        pipe = self._client.pipeline(transaction=False)
        pipe.hdel(self.declarations_key, executor_id)
        pipe.zrem(self.alive_key, executor_id)
        await pipe.execute()

    async def alive(self, forget_after_ms: int) -> tuple[int, dict[str, int]]:
        """The server's time, and each executor's last beat (both in ms);
        executors silent for *forget_after_ms* are forgotten."""
        answer = await self._alive_script(
            keys=[self.alive_key, self.declarations_key], args=[forget_after_ms]
        )
        now, rest = int(answer[0]), answer[1:]
        beats = {_text(rest[i]): int(float(rest[i + 1])) for i in range(0, len(rest), 2)}
        return now, beats

    async def declarations(self, executor_ids: Iterable[str]) -> dict[str, Declaration]:
        ids = list(executor_ids)
        if not ids:
            return {}
        raws = await self._client.hmget(self.declarations_key, ids)
        found: dict[str, Declaration] = {}
        for executor_id, raw in zip(ids, raws, strict=True):
            if raw is None:
                continue
            try:
                found[executor_id] = Declaration.load(raw)
            except (ValueError, KeyError, TypeError):
                logger.warning("The declaration of executor %s is malformed", executor_id)
        return found


def _text(value: Any) -> str:
    return value if isinstance(value, str) else bytes(value).decode()


class LiveExecutors:
    """The owner's view of the executors alive, refreshed each heartbeat
    interval: their declarations, and how many beats each was seen to make."""

    def __init__(self, registry: Registry, interval: float, timeout: float) -> None:
        self._registry = registry
        self.interval = interval
        """How often the view is read again."""
        self._timeout_ms = int(timeout * 1000)
        self._declared: dict[str, Declaration] = {}
        self._beats: dict[str, int] = {}
        self._refresher: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()

    def start(self) -> None:
        if self._refresher is None:
            self._refresher = asyncio.get_running_loop().create_task(
                self._refresh_each_interval(), name="roomkit-turns-live-executors"
            )

    async def current(self) -> dict[str, Declaration]:
        """The executors alive and what they declare, read now."""
        await self.refresh()
        return dict(self._declared)

    async def declared(self) -> dict[str, Declaration]:
        """The executors alive and what they declare, as last read (read now
        the first time)."""
        if not self._ready.is_set():
            await self.refresh()
        return dict(self._declared)

    def last_beat(self, executor_id: str) -> int | None:
        """The server time of *executor_id*'s last beat seen, in ms."""
        return self._beats.get(executor_id)

    async def refresh(self) -> None:
        now, beats = await self._registry.alive(forget_after_ms=10 * self._timeout_ms)
        alive = {eid: at for eid, at in beats.items() if now - at < self._timeout_ms}
        self._declared = await self._registry.declarations(alive)
        self._beats = beats
        self._ready.set()

    async def close(self) -> None:
        if self._refresher is not None:
            self._refresher.cancel()
            await asyncio.gather(self._refresher, return_exceptions=True)
            self._refresher = None

    async def _refresh_each_interval(self) -> None:
        while True:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Reading the turn executors alive failed", exc_info=True)
            await asyncio.sleep(self.interval)


class BeatThread:
    """Beats for this process's executors from a thread of its own, on a
    connection of its own (RFC §10.6.8), and keeps the process's inbox."""

    def __init__(
        self,
        connect: Callable[[], Any],
        alive_key: str,
        inbox_key: str,
        interval: float,
        inbox_ttl_ms: int,
    ) -> None:
        self._connect = connect
        self._alive_key = alive_key
        self._inbox_key = inbox_key
        self._interval = interval
        self._inbox_ttl_ms = inbox_ttl_ms
        self._executors: dict[str, TurnExecutor] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat_for(self, executors: Mapping[str, TurnExecutor]) -> None:
        """Beat for *executors*, by id, from now on (replacing the set): one
        that closed beats no more, so it is soon not alive."""
        self._executors = dict(executors)
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="roomkit-turns-beat", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None

    def _run(self) -> None:
        client = self._connect()
        beat = client.register_script(_BEAT)
        try:
            while True:
                ids = [eid for eid, executor in self._executors.items() if not executor.closed]
                if ids:
                    try:
                        beat(
                            keys=[self._alive_key, self._inbox_key],
                            args=[self._inbox_ttl_ms, *sorted(ids)],
                        )
                    except Exception:
                        logger.warning("A turn executor beat failed", exc_info=True)
                if self._stop.wait(self._interval):
                    return
        finally:
            client.close()
