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
    """Two owners share the backend: a status the executor posts once its
    turn ended (its host no longer open) reaches the one whose turn began
    it, through the owner port, never the other (RFC §10.6.3)."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    later_work: list[asyncio.Task[None]] = []
    open_when_posted: list[bool] = []

    async def post_later(turn_host: Any) -> None:
        await asyncio.sleep(0.3)  # the turn has returned
        open_when_posted.append(turn_host.open)
        await turn_host.post_status("worker", "thought", "info", detail="after the turn")

    @executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def later(event: Any, context: Any) -> None:
        later_work.append(asyncio.get_running_loop().create_task(post_later(current_turn_host())))

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

        assert open_when_posted == [False]
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


# -- What the review of 1b7af07f found -----------------------------------------------


@pytest.mark.parametrize("frame", ["output", "end"])
async def test_a_frame_redis_refuses_fails_the_turn_rather_than_waits(
    prefix: str, redis_client: Any, frame: str
) -> None:
    """The executor's process still beats, but no longer runs the turn: its
    owner gives it up within the heartbeat window, never waits for good."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    post = host._outbox.post
    refused: list[str] = []

    def refusing(node: str, envelope: Any, failed: Any = None) -> None:
        if envelope.kind == "frame" and f'"type":"{frame}"' in envelope.body and not refused:
            refused.append(frame)
            failed(ConnectionError("OOM command not allowed"))
            return
        post(node, envelope, failed)

    host._outbox.post = refusing  # type: ignore[method-assign]
    kit = await _owner(prefix, "worker", store=store)
    try:
        result = await asyncio.wait_for(
            kit.process_inbound(_said("Hi."), room_id="r"), 6 * TIMEOUTS.heartbeat_timeout
        )

        assert refused == [frame]
        if frame == "output":
            assert isinstance(result.error, TurnLinkError)
        else:  # its answer was read whole: only its end is missing
            assert await _answers(kit) == ["Hi."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_many_rooms_at_once_keep_every_turn_s_order(prefix: str, redis_client: Any) -> None:
    """Sixty rooms' turns at once, owner and executor writing concurrently:
    every message arrives in its turn's order, no connection runs out."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 100, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = RoomKit(store=store, turn_backend=_backend(prefix))
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(ChannelDeclaration(channel_id="worker", identity=AgentIdentity()))
    rooms = [f"r{i}" for i in range(60)]
    for room in rooms:
        await kit.create_room(room_id=room)
        await kit.attach_channel(room, "tx")
        await kit.attach_channel(room, "worker", category=ChannelCategory.INTELLIGENCE)
    try:
        results = await asyncio.gather(
            *(kit.process_inbound(_said("Hi."), room_id=room) for room in rooms)
        )

        assert [r.error for r in results] == [None] * len(rooms)
        assert host._routes._executor_links == {}
        assert kit._turns.backend._routes._owner_links == {}
    finally:
        await kit.close()
        await host.close()
        await executor.close()


class _SlowWrites:
    """A Redis client whose pipelines take 2 ms to write: a cut lands while
    a frame is being written."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def pipeline(self, **kwargs: Any) -> Any:
        pipe = self._client.pipeline(**kwargs)
        execute = pipe.execute

        async def slow() -> Any:
            await asyncio.sleep(0.002)
            return await execute()

        pipe.execute = slow
        return pipe

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


async def test_a_cut_while_frames_are_written_leaves_no_hole(
    prefix: str, redis_client: Any
) -> None:
    """Cancelled deliveries whose frames were being written each end with
    their turn's end: nothing is reported incomplete (RFC §10.6.8)."""
    store = InMemoryStore()
    started = asyncio.Event()

    class _Words(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            started.set()
            for i in range(500):
                yield StreamTextDelta(text=f"w{i} ")
                await asyncio.sleep(0.001)

    worker = _Worker("worker", _Words(streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    host._outbox._client = _SlowWrites(host._outbox._client)
    kit = await _owner(prefix, "worker", store=store)
    try:
        for _ in range(10):
            started.clear()
            result = await kit.process_inbound(_said("Talk."), room_id="r", defer_delivery=True)
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.05)
            assert result.delivery is not None
            await asyncio.wait_for(result.delivery.cancel(), 10)
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_an_executor_restarted_under_its_id_does_not_hold_the_dead_one_s_turn(
    prefix: str, redis_client: Any, database: Path
) -> None:
    """A supervisor restarts the killed process under the same executor id:
    the turn the dead process ran is lost, not kept alive by the new one's
    beats."""
    first = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    second: _ExecutorProcess | None = None
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Please hold."), room_id="r"))
        await first.wait_for("streaming")
        await first.kill()
        second = await _ExecutorProcess.start(prefix, "host-1", database)
        result = await asyncio.wait_for(turn, 6 * TIMEOUTS.heartbeat_timeout)

        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "executor_lost"
    finally:
        await kit.close()
        if second is not None:
            await second.stop()


async def test_a_declaration_lost_from_redis_is_written_again(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await redis_client.delete(f"{prefix}:executors")  # a restart without persistence
        await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert result.error is None
        assert await _answers(kit) == ["Hi."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_room_hook_defined_after_serving_is_declared(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)

    @executor.room_hook("audit", HookTrigger.ON_AI_RESPONSE)
    async def audit(event: Any, context: Any) -> None: ...

    kit = await _owner(prefix, "worker", store=store)
    kit.enable_room_hook("r", "audit")
    try:
        await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert result.error is None
    finally:
        await kit.close()
        await host.close()
        await executor.close()


class _Remembers(Agent):
    """An agent whose attach and memory take time, logging what it does."""

    def __init__(self, channel_id: str, log: list[str]) -> None:
        super().__init__(channel_id, provider=MockAIProvider(responses=["ok"] * 5, streaming=True))
        self._log = log

    async def on_room_attached(self, room_id: str, binding: Any) -> None:
        await asyncio.sleep(0.2)
        self._log.append("attached")

    async def on_room_event(self, room_id: str, event: RoomEvent) -> None:
        await asyncio.sleep(0.2)  # a memory store's ingest
        assert isinstance(event.content, TextContent)
        self._log.append(f"memory:{event.content.body}")

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        assert isinstance(event.content, TextContent)
        self._log.append(f"turn:{event.content.body}")
        return await super().on_event(event, binding, context)


async def test_channel_controls_apply_before_the_turns_that_follow(
    prefix: str, redis_client: Any
) -> None:
    """As in memory: the attach, then a committed event, are applied on the
    executor before the room's next turn runs (RFC §10.6.3)."""
    store = InMemoryStore()
    log: list[str] = []
    host, executor = await _hosted(prefix, store, _Remembers("worker", log))
    kit = await _owner(prefix, "worker", store=store)
    event = RoomEvent(
        room_id="r",
        source=EventSource(channel_id="tx", channel_type=ChannelType.SMS),
        content=TextContent(body="earlier"),
    )
    try:
        await kit._turns.backend.control(
            CommittedEvent(channel_id="worker", room_id="r", event=event)
        )
        await kit.process_inbound(_said("later"), room_id="r")

        assert log == ["attached", "memory:earlier", "turn:later"]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_room_whose_executor_closed_moves_to_another(
    prefix: str, redis_client: Any
) -> None:
    """Its holder's executor closed, its process still running: the room's
    next turn goes to another executor, as in memory."""
    store = InMemoryStore()
    backends = []
    executors = []
    for name in ("host-a", "host-b"):
        backend = _backend(prefix)
        executor = _agent_executor(name, store, hosts={"worker"}, answer=f"From {name}.")
        await backend.serve(executor)
        backends.append(backend)
        executors.append(executor)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await kit.process_inbound(_said("Hi."), room_id="r")
        holder = 0 if (await _answers(kit))[0] == "From host-a." else 1
        await executors[holder].close()
        result = await kit.process_inbound(_said("Hi again."), room_id="r")

        assert result.error is None
        assert (await _answers(kit))[-1] == f"From {['host-a', 'host-b'][1 - holder]}."
    finally:
        await kit.close()
        for backend, executor in zip(backends, executors, strict=True):
            await backend.close()
            await executor.close()


async def test_a_request_redis_refuses_leaves_no_link(prefix: str, redis_client: Any) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    backend = kit._turns.backend
    post = backend._outbox.post

    def refusing(node: str, envelope: Any, failed: Any = None) -> None:
        if envelope.kind == "request":
            failed(ConnectionError("READONLY You can't write against a read only replica."))
            return
        post(node, envelope, failed)

    backend._outbox.post = refusing
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert isinstance(result.error, TurnLinkError)
        assert backend._routes._owner_links == {}
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_an_executor_s_turn_its_owner_gave_up_leaves_no_link(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()

    class _Holds(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            yield StreamTextDelta(text="Hello ")
            await asyncio.sleep(60)

    host, executor = await _hosted(prefix, store, _Worker("worker", _Holds(streaming=True)))
    kit = await _owner(prefix, "worker", store=store)
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Hi."), room_id="r"))
        async with asyncio.timeout(5):
            while not host._routes._executor_links:
                await asyncio.sleep(0.02)
        await kit.close()  # the owner goes: the executor cuts the turn
        await asyncio.gather(turn, return_exceptions=True)
        async with asyncio.timeout(6 * TIMEOUTS.heartbeat_timeout):
            while host._routes._executor_links:
                await asyncio.sleep(0.05)

        assert host._routes.running == frozenset()
    finally:
        await host.close()
        await executor.close()
