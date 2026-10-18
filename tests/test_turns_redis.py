"""The Redis turn backend against a real Redis (RFC §10.6): set REDIS_URL.

A room owner and executors in other processes (subprocesses running
tests/redis_turn_executor.py), or several backends in this process standing
for several processes: placement, liveness, a lost executor, the work a turn
began reaching its owner.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from roomkit import (
    AgentIdentity,
    ChannelDeclaration,
    HookExecution,
    HookTrigger,
    RoomKit,
    current_turn_host,
)
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.steering import Cancel
from roomkit.providers.ai.base import StreamTextDelta
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from roomkit.turns import TurnExecutor, TurnTimeouts
from roomkit.turns.redis import RedisTurnBackend
from roomkit.turns.wire import CommittedEvent
from tests.test_framework import SimpleChannel

pytestmark = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")

TIMEOUTS = TurnTimeouts(heartbeat_interval=0.2, heartbeat_timeout=1.0, accept=3.0, cancel=1.0)
EXECUTOR = Path(__file__).with_name("redis_turn_executor.py")


@pytest.fixture
def prefix() -> str:
    return f"roomkit:test:{uuid4().hex[:12]}"


@pytest.fixture
async def redis_client(prefix: str) -> AsyncIterator[Any]:
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    yield client
    keys = [key async for key in client.scan_iter(f"{prefix}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


def _backend(prefix: str) -> RedisTurnBackend:
    return RedisTurnBackend(os.environ["REDIS_URL"], key_prefix=prefix, timeouts=TIMEOUTS)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """The SQLite store the owner and the executor processes share."""
    return tmp_path / "rooms.db"


async def _owner(prefix: str, *agents: str, store: ConversationStore | None = None) -> RoomKit:
    kit = RoomKit(store=store or InMemoryStore(), turn_backend=_backend(prefix))
    kit.register_channel(SimpleChannel("tx"))
    for agent in agents:
        kit.register_channel(ChannelDeclaration(channel_id=agent, identity=AgentIdentity()))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    for agent in agents:
        await kit.attach_channel("r", agent, category=ChannelCategory.INTELLIGENCE)
    return kit


def _said(text: str) -> InboundMessage:
    return InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body=text))


async def _answers(kit: RoomKit, channel_id: str = "worker") -> list[str]:
    return [
        e.content.body
        for e in await kit.store.list_events("r")
        if e.source.channel_id == channel_id and e.type == EventType.MESSAGE
    ]


class _ExecutorProcess:
    """An executor serving ``worker`` in a subprocess."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def start(cls, prefix: str, executor_id: str, database: Path) -> _ExecutorProcess:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(EXECUTOR),
            os.environ["REDIS_URL"],
            prefix,
            executor_id,
            str(database),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        started = cls(process)
        await started.wait_for("ready")
        return started

    async def wait_for(self, line: str) -> None:
        assert self.process.stdout is not None
        while True:
            got = await asyncio.wait_for(self.process.stdout.readline(), 20)
            assert got, f"the executor process ended before printing {line!r}"
            if got.decode().strip() == line:
                return

    async def stop(self) -> None:
        if self.process.returncode is None:
            assert self.process.stdin is not None
            self.process.stdin.write(b"stop\n")
            await self.process.stdin.drain()
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def kill(self) -> None:
        self.process.kill()
        await self.process.wait()


# -- An executor in another process -----------------------------------------------


async def test_a_turn_runs_on_an_executor_in_another_process(
    prefix: str, redis_client: Any, database: Path
) -> None:
    executor = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert await _answers(kit) == ["Hello from host-1."]
        assert dict(result.response_metadata)["turns"]["worker"]["loop_end_reason"] == "completed"
    finally:
        await kit.close()
        await executor.stop()


async def test_a_turn_whose_cpu_stalls_its_executor_keeps_beating(
    prefix: str, redis_client: Any, database: Path
) -> None:
    """The executor's loop stops three times the heartbeat timeout: its
    thread beats on, so the turn is slow, never lost (RFC §10.6.8)."""
    executor = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    try:
        started = time.monotonic()
        result = await kit.process_inbound(_said("Please stall."), room_id="r")

        assert time.monotonic() - started >= 3.0
        assert result.error is None
        assert await _answers(kit) == ["Hello from host-1."]
    finally:
        await kit.close()
        await executor.stop()


async def test_a_killed_executor_fails_its_turn_and_the_room_goes_on(
    prefix: str, redis_client: Any, database: Path
) -> None:
    first = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    second: _ExecutorProcess | None = None
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Please hold."), room_id="r"))
        await first.wait_for("streaming")
        killed = time.monotonic()
        await first.kill()
        result = await asyncio.wait_for(turn, 10)

        assert time.monotonic() - killed < 3 * TIMEOUTS.heartbeat_timeout
        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "executor_lost"
        # Another executor takes the room's next turn.
        second = await _ExecutorProcess.start(prefix, "host-2", database)
        await kit.process_inbound(_said("Hi again."), room_id="r")
        assert (await _answers(kit))[-1] == "Hello from host-2."
    finally:
        await kit.close()
        if second is not None:
            await second.stop()


# -- Several processes, as backends of this one --------------------------------------


def _agent_executor(
    executor_id: str, store: ConversationStore, *, hosts: set[str], answer: str
) -> TurnExecutor:
    return TurnExecutor(
        lambda channel_id: Agent(
            channel_id, provider=MockAIProvider(responses=[answer] * 20, streaming=True)
        ),
        hosts=hosts,
        executor_id=executor_id,
        timeouts=TIMEOUTS,
        store=store,
    )


async def test_a_turn_no_executor_declares_fails_at_once(prefix: str, redis_client: Any) -> None:
    kit = await _owner(prefix, "nobody")
    try:
        started = time.monotonic()
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert time.monotonic() - started < TIMEOUTS.accept
        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "no_executor"
    finally:
        await kit.close()


async def test_a_room_s_turns_stay_on_the_executor_holding_it(
    prefix: str, redis_client: Any
) -> None:
    """Two executors take ``worker``: the room's turns stay on the one that
    took its first, until it stops (RFC §10.6.7)."""
    store = InMemoryStore()  # one process: standing for a store they share
    hosts = []
    executors = []
    for name in ("host-a", "host-b"):
        backend = _backend(prefix)
        executor = _agent_executor(name, store, hosts={"worker"}, answer=f"From {name}.")
        await backend.serve(executor)
        hosts.append(backend)
        executors.append(executor)
    kit = await _owner(prefix, "worker", store=store)
    try:
        for i in range(4):
            await kit.process_inbound(_said(f"Hi {i}."), room_id="r")
        answers = await _answers(kit)
        assert len(set(answers)) == 1
        holder = 0 if answers[0] == "From host-a." else 1
        await hosts[holder].close()
        await executors[holder].close()
        await kit.process_inbound(_said("Hi again."), room_id="r")

        assert (await _answers(kit))[-1] == f"From {['host-a', 'host-b'][1 - holder]}."
    finally:
        await kit.close()
        for backend, executor in zip(hosts, executors, strict=True):
            await backend.close()
            await executor.close()


async def test_an_executor_without_the_room_s_hook_is_not_given_its_turns(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    plain_backend, hooked_backend = _backend(prefix), _backend(prefix)
    plain = _agent_executor("plain", store, hosts={"worker"}, answer="From plain.")
    hooked = _agent_executor("hooked", store, hosts={"worker"}, answer="From hooked.")
    seen: list[str] = []

    @hooked.room_hook("audit", HookTrigger.ON_AI_RESPONSE)
    async def audit(event: Any, context: Any) -> None:
        seen.append("audited")

    await plain_backend.serve(plain)
    await hooked_backend.serve(hooked)
    kit = await _owner(prefix, "worker", store=store)
    try:
        kit.enable_room_hook("r", "audit")
        for i in range(3):
            await kit.process_inbound(_said(f"Hi {i}."), room_id="r")

        assert await _answers(kit) == ["From hooked."] * 3
    finally:
        await kit.close()
        for backend, executor in ((plain_backend, plain), (hooked_backend, hooked)):
            await backend.close()
            await executor.close()


# -- Channel controls, work after a turn, steering ----------------------------------


class _Worker(Agent):
    """An agent with a per-room verb, that keeps the events handed to it."""

    def __init__(self, channel_id: str, provider: Any) -> None:
        super().__init__(channel_id, provider=provider)
        self.handed: list[str] = []

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        return {"verb": verb, "room": room_id, **arguments}

    async def on_room_event(self, room_id: str, event: RoomEvent) -> None:
        assert isinstance(event.content, TextContent)
        self.handed.append(event.content.body)


async def _hosted(
    prefix: str, store: ConversationStore, channel: Any, name: str = "host-a"
) -> tuple[RedisTurnBackend, TurnExecutor]:
    backend = _backend(prefix)
    executor = TurnExecutor(
        lambda channel_id: channel if channel_id == channel.channel_id else None,
        hosts={channel.channel_id},
        executor_id=name,
        timeouts=TIMEOUTS,
        store=store,
    )
    await backend.serve(executor)
    return backend, executor


async def test_a_channel_call_and_a_committed_event_reach_the_executor_elsewhere(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        answer = await kit.channel_call("worker", "r", "config_options", key="model")
        event = RoomEvent(
            room_id="r",
            source=EventSource(channel_id="tx", channel_type=ChannelType.SMS),
            content=TextContent(body="A message for the memory."),
        )
        await kit._turns.backend.control(
            CommittedEvent(channel_id="worker", room_id="r", event=event)
        )
        async with asyncio.timeout(5):
            while not worker.handed:
                await asyncio.sleep(0.02)

        assert answer == {"verb": "config_options", "room": "r", "key": "model"}
        assert worker.handed == ["A message for the memory."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_work_after_a_turn_reaches_the_owner_that_submitted_it(
    prefix: str, redis_client: Any
) -> None:
    """Two owners share the backend: a status the executor posts once the
    turn ended reaches the one whose turn began it (RFC §10.6.3)."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)

    @executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def later(event: Any, context: Any) -> None:
        await asyncio.sleep(0.2)  # the turn has ended
        turn_host = current_turn_host()
        assert turn_host is not None
        await turn_host.post_status("worker", "thought", "info", detail="after the turn")

    asking, other = await _owner(prefix, "worker", store=store), await _owner(prefix, "worker")
    posted: dict[str, list[str]] = {"asking": [], "other": []}

    async def on(name: str, entry: Any) -> None:
        posted[name].append(entry.detail)

    await asking.status_bus.subscribe(lambda entry: on("asking", entry))
    await other.status_bus.subscribe(lambda entry: on("other", entry))
    try:
        await asking.process_inbound(_said("Hi."), room_id="r")
        async with asyncio.timeout(5):
            while not posted["asking"]:
                await asyncio.sleep(0.02)

        assert posted == {"asking": ["after the turn"], "other": []}
    finally:
        await asking.close()
        await other.close()
        await host.close()
        await executor.close()


async def test_a_steering_cancel_reaches_a_turn_elsewhere(prefix: str, redis_client: Any) -> None:
    store = InMemoryStore()
    started = asyncio.Event()

    class _Slow(MockAIProvider):
        """Streams a long answer, a word every 50 ms: the loop reads the
        cancel at its next delta."""

        async def generate_structured_stream(self, context: Any) -> Any:
            started.set()
            for i in range(200):
                yield StreamTextDelta(text=f"w{i} ")
                await asyncio.sleep(0.05)

    worker = _Worker("worker", _Slow(streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Hi."), room_id="r"))
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.sleep(0.1)
        reached = await kit.steer(Cancel(), "worker", room_id="r")
        result = await asyncio.wait_for(turn, 5)

        assert reached == 1
        assert dict(result.response_metadata)["turns"]["worker"]["loop_end_reason"] == "cancelled"
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_turn_for_an_executor_that_closed_fails_at_once(
    prefix: str, redis_client: Any
) -> None:
    """Its backend still runs and its beat is not stale yet: the request
    reaches its process, which answers at once that it no longer serves."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await executor.close()
        started = time.monotonic()
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert time.monotonic() - started < TIMEOUTS.accept
        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "no_executor"
    finally:
        await kit.close()
        await host.close()
