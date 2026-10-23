"""A discussion uninstalled, then installed again, across processes (RFC §19.7.5 rule 16).

A process that read the discussion uninstalled leaves it, even when another
process installed one again before it looked; its host installing again
starts the discussion afresh, never in the room it left. Until it looks, it
writes nothing back into the room, so a discussion installed later starts
with an empty queue.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from roomkit import Discussion, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.enums import ChannelType, EventType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration.strategies.discussion import (
    SpeakQueueChange,
    SpeakQueueEvent,
    _driver,
    _shared,
)
from roomkit.orchestration.strategies.discussion._config import DiscussionConfig
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.orchestration.test_discussion_processes import (
    AGENTS,
    KEPT,
    _Cluster,
    _say,
    _turn_started,
    _until,
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll and lease on a test's time scale."""
    monkeypatch.setattr(_driver, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(_shared, "LEASE_SECONDS", 0.6)
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", 0.4)


class _Holding(_Cluster):
    """Processes whose first investigator turn runs until the test ends it."""

    def __init__(self) -> None:
        super().__init__()
        self.first_turn_ends = asyncio.Event()


class _Held(MockAIProvider):
    """Says *body*, logging when it generates. The cluster's first
    investigator turn waits for ``first_turn_ends`` (five seconds at most)
    and, as a provider that does not stream, meets a ``Cancel`` only once it
    answered; any other turn takes 0.01 s."""

    def __init__(self, name: str, cluster: _Holding, body: str) -> None:
        super().__init__(ai_responses=[AIResponse(content=body)])
        self.who, self.log, self.ends = name, cluster.log, cluster.first_turn_ends

    async def generate(self, context: Any) -> AIResponse:
        first = f"+{self.who}" not in self.log
        self.log.append(f"+{self.who}")
        if first and self.who == "investigator":
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.ends.wait(), 5.0)
        else:
            await asyncio.sleep(0.01)
        return await super().generate(context)


def _handing_over(cluster: _Holding, names: tuple[str, ...] = KEPT) -> Discussion:
    """investigator asks @sre in every answer."""
    bodies = {n: "@sre please check" if n == "investigator" else f"{n} here" for n in names}
    return Discussion([AIChannel(n, provider=_Held(n, cluster, bodies[n])) for n in names])


async def _created(cluster: _Holding, strategy: Discussion | None = None) -> RoomKit:
    kit = cluster.kit()
    await kit.create_room(room_id="r1", orchestration=strategy or _handing_over(cluster))
    await kit.attach_channel("r1", "ops")
    return kit


def _ends(kit: RoomKit) -> list[str]:
    """The agents whose turn *kit* announces ended, as it announces them."""
    ended: list[str] = []

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: Any) -> None:
        if event.change == SpeakQueueChange.TURN_ENDED:
            ended.extend(event.channel_ids)

    return ended


async def _turn_ended(ended: list[str], agent: str = "investigator") -> None:
    async def announced() -> bool:
        return agent in ended

    await _until(announced)


async def _stored_queue(cluster: _Cluster) -> dict[str, Any]:
    room = await cluster.store.get_room("r1")
    return dict((room.metadata if room is not None else {}).get("_speak_queue") or {})


async def _leased(cluster: _Cluster, kit: RoomKit | None = None) -> None:
    """Until *kit*'s driver, else any, took the lease: it read the queue once,
    and reads it again only once POLL_SECONDS passed or it is woken."""

    async def leased() -> bool:
        holder = (await _stored_queue(cluster)).get("lease_holder")
        return holder is not None and (kit is None or holder == kit._instance_id)

    await _until(leased)


async def _cut_by_an_uninstall_elsewhere(cluster: _Holding, a: RoomKit, b: RoomKit) -> None:
    """A gives investigator's long turn; B uninstalls and A's watch cuts it."""
    await _say(a, "@investigator take your time")
    await _turn_started(cluster)
    held = a._discussions["r1"]
    assert await b.uninstall_strategy("r1") is True

    async def seen_gone() -> bool:  # A's watch has read the discussion gone
        return held.forgotten

    await _until(seen_gone)


async def _sre_answers_investigator(cluster: _Cluster, kit: RoomKit) -> None:
    """investigator's turn ends recorded: the @sre it says gives sre a turn."""
    mark = len(cluster.log)
    await _say(kit, "@investigator go on")

    async def sre_answered() -> bool:
        return "+sre" in cluster.log[mark:]

    await _until(sre_answered)


@pytest.mark.parametrize("host_asks", [False, True], ids=["driver_looks", "host_asks"])
async def test_a_process_that_saw_an_uninstall_leaves_the_discussion_installed_again(
    host_asks: bool,
) -> None:
    cluster = _Holding()
    a, b = await _created(cluster), cluster.kit()
    ended = _ends(a)
    await _cut_by_an_uninstall_elsewhere(cluster, a, b)
    await b.install_strategy("r1", _handing_over(cluster))  # before A looks again

    if host_asks:
        assert await a.room_strategy_is_current("r1", Discussion) is False
    cluster.first_turn_ends.set()

    async def left() -> bool:
        return a.room_strategy("r1") is None

    await _until(left)
    await _turn_ended(ended)  # past A's cut turn
    await _sre_answers_investigator(cluster, b)
    await cluster.close()


async def test_installing_again_over_a_discussion_seen_uninstalled_starts_it_afresh() -> None:
    cluster = _Holding()
    a, b = await _created(cluster), cluster.kit()
    ended = _ends(a)
    await _cut_by_an_uninstall_elsewhere(cluster, a, b)
    # A's host still wants the discussion: it installs before A looks again.
    await a.install_strategy("r1", _handing_over(cluster))

    assert await a.room_strategy_is_current("r1", Discussion) is True
    cluster.first_turn_ends.set()
    await _turn_ended(ended)  # past A's cut turn
    await _sre_answers_investigator(cluster, a)
    await cluster.close()


class _SlowContext(InMemoryStore):
    """A store whose next room context, once armed, takes a while to come
    back: the round trip a Postgres store takes. ``driver_read`` is set
    each time a discussion's driver reads the room."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: tuple[asyncio.Event, asyncio.Event] | None = None
        self.driver_read = asyncio.Event()

    async def get_room(self, room_id: str) -> Any:
        room = await super().get_room(room_id)
        task = asyncio.current_task()
        if task is not None and task.get_name().startswith("roomkit-discussion-"):
            self.driver_read.set()
        return room

    def arm(self) -> tuple[asyncio.Event, asyncio.Event]:
        self.gate = (asyncio.Event(), asyncio.Event())
        return self.gate

    async def load_room_context(self, room_id: str) -> Any:
        context = await super().load_room_context(room_id)
        gate, self.gate = self.gate, None
        if gate is not None:
            entered, release = gate
            entered.set()
            await release.wait()
        return context


@pytest.mark.parametrize("reader", ["driver", "host"])
async def test_installing_again_after_an_uninstall_elsewhere_gives_turns(
    reader: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A does not look again before its host installs; the install's round
    # trips leave time for A's driver or another call of its host to read
    # the discussion gone.
    monkeypatch.setattr(_driver, "POLL_SECONDS", 5.0)
    cluster = _Cluster()
    store = cluster.store = _SlowContext()
    a = await cluster.process(install=True, create=True, delay=0.05)
    await _leased(cluster, a)
    b = await cluster.process(install=False)
    assert await b.uninstall_strategy("r1") is True
    held = a._discussions["r1"]

    entered, release = store.arm()
    install = asyncio.create_task(a.install_strategy("r1", cluster.strategy(delay=0.05)))
    await asyncio.wait_for(entered.wait(), 2.0)
    if reader == "driver":
        store.driver_read.clear()
        held.wake()

        async def looked() -> bool:  # it read the room, unless the install stopped it
            driver = held.driver
            return store.driver_read.is_set() or driver is None or driver._task is None

        await _until(looked)
    else:
        assert await a.room_strategy_is_current("r1", Discussion) is False
    release.set()
    await install

    assert await a.room_strategy_is_current("r1", Discussion) is True
    await _say(a, "@investigator are you there?")

    async def answered() -> bool:
        return await cluster.answers() == ["investigator"]

    await _until(answered)
    await cluster.close()


class _LateRead(InMemoryStore):
    """A store where one task's next room read, once armed, comes back late
    with the room as it was when asked: a slow round trip."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: tuple[str, asyncio.Event, asyncio.Event] | None = None
        self.came_back = asyncio.Event()

    def arm(self, task_name: str) -> tuple[asyncio.Event, asyncio.Event]:
        entered, release = asyncio.Event(), asyncio.Event()
        self.gate = (task_name, entered, release)
        return entered, release

    async def get_room(self, room_id: str) -> Any:
        room = await super().get_room(room_id)
        task = asyncio.current_task()
        if self.gate is None or task is None or not task.get_name().startswith(self.gate[0]):
            return room
        _, entered, release = self.gate
        self.gate = None
        entered.set()
        await release.wait()
        self.came_back.set()
        return room


@pytest.mark.parametrize("reader", ["driver", "host"])
async def test_a_read_begun_before_an_install_here_does_not_undo_it(
    reader: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A's read finds the discussion gone (B uninstalled it), and comes back
    # once B installed it again and A's host installed over it: the read
    # tells nothing of A's install.
    monkeypatch.setattr(_driver, "POLL_SECONDS", 5.0)
    cluster = _Cluster()
    store = cluster.store = _LateRead()
    a = await cluster.process(install=True, create=True, delay=0.05)
    await _leased(cluster, a)
    b = cluster.kit()
    assert await b.uninstall_strategy("r1") is True

    if reader == "driver":
        entered, release = store.arm("roomkit-discussion-")
        a._discussions["r1"].wake()
        reading: asyncio.Task[Any] | None = None
    else:
        entered, release = store.arm("late-read")
        reading = asyncio.create_task(
            a.room_strategy_is_current("r1", Discussion), name="late-read"
        )
    await asyncio.wait_for(entered.wait(), 2.0)
    await b.install_strategy("r1", cluster.strategy(delay=0.05))
    await a.install_strategy("r1", cluster.strategy(delay=0.05))
    release.set()
    if reading is not None:
        await reading
    else:
        # Back once the driver, done with what it read, waits again.
        await asyncio.wait_for(store.came_back.wait(), 2.0)

    assert isinstance(a.room_strategy("r1"), Discussion)
    assert await a.room_strategy_is_current("r1", Discussion) is True
    await b.close()  # A alone gives the turns from here (B's close gives its lease up)
    await _say(a, "@investigator are you there?")

    async def answered() -> bool:
        return await cluster.answers() == ["investigator"]

    await _until(answered)
    await cluster.close()


async def test_a_message_before_a_process_sees_an_uninstall_writes_no_queue_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bound investigator itself, so it stays bound once B uninstalls; A
    # does not look again before a person writes to it.
    monkeypatch.setattr(_driver, "POLL_SECONDS", 5.0)
    cluster = _Cluster()
    a = await cluster.hosted("investigator")
    await _leased(cluster, a)
    b = await cluster.process(install=False)
    assert await b.uninstall_strategy("r1") is True

    await _say(a, "@investigator are you there?")
    room = await cluster.store.get_room("r1")
    assert room is not None and room.metadata.get("_speak_queue") is None

    # A discussion installed later starts with nothing asked of it: once a
    # driver read the queue (B's, or A's woken by the message, which never
    # saw the uninstall), nothing is there to give, and nothing else writes.
    await b.install_strategy("r1", cluster.strategy(delay=0.05))
    await _leased(cluster)
    stored = await _stored_queue(cluster)
    assert (stored["entries"], stored["dispatching"], stored["speaking"]) == ([], [], None)
    assert await cluster.answers() == []
    await cluster.close()


class _Contended(InMemoryLockManager):
    """A lock manager that tells when a discussion's driver waits for a room
    lock another holds."""

    def __init__(self) -> None:
        super().__init__()
        self.driver_waits = asyncio.Event()

    @contextlib.asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        task, lock = asyncio.current_task(), self._locks.get(room_id)
        driver = task is not None and task.get_name().startswith("roomkit-discussion-")
        if driver and lock is not None and lock.locked():
            self.driver_waits.set()
        async with super().locked(room_id):
            yield


async def test_a_turn_looked_for_as_another_process_uninstalls_leaves_no_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every look of A's driver renews its lease, under the room's lock.
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", _shared.LEASE_SECONDS)
    cluster = _Cluster()
    locks = cluster.locks = _Contended()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=False)
    await _leased(cluster, a)
    async with locks.locked("r1"):
        # A's driver has read the queue and waits for the lock.
        await asyncio.wait_for(locks.driver_waits.wait(), 2.0)
        assert await b.uninstall_strategy("r1") is True

    async def left() -> bool:  # A's next look found the discussion gone
        return a.room_strategy("r1") is None

    await _until(left)

    room = await cluster.store.get_room("r1")
    assert room is not None and room.metadata.get("_speak_queue") is None
    await cluster.close()


async def _installed_again_elsewhere(cluster: _Holding) -> RoomKit:
    """A's watch read B's uninstall; B installed every agent again since."""
    a = cluster.kit()
    await a.create_room(room_id="r1", orchestration=_handing_over(cluster, AGENTS))
    await a.attach_channel("r1", "ops")
    b = cluster.kit()
    await _cut_by_an_uninstall_elsewhere(cluster, a, b)
    await b.install_strategy("r1", _handing_over(cluster, AGENTS))
    return a


async def test_installing_again_over_one_installed_elsewhere_meanwhile_takes_its_place() -> None:
    cluster = _Holding()
    a = await _installed_again_elsewhere(cluster)

    # A's host installs without comms before A looks again: it takes the place
    # of B's discussion, and comms, which an install attached, goes.
    await a.install_strategy("r1", _handing_over(cluster, KEPT))

    stored = await cluster.stored()
    assert stored is not None and stored.agents == list(KEPT)
    assert "comms" not in await cluster.bound()
    assert await a.room_strategy_is_current("r1", Discussion) is True
    await cluster.close()


async def test_installing_again_over_one_installed_elsewhere_keeps_what_it_attached() -> None:
    cluster = _Holding()
    a = await _installed_again_elsewhere(cluster)
    attached = (await cluster.stored() or DiscussionConfig()).attached

    await a.install_strategy("r1", _handing_over(cluster, AGENTS))
    stored = await cluster.stored()
    assert stored is not None and stored.attached == attached

    follower = await cluster.process(install=False)
    assert await follower.uninstall_strategy("r1") is True
    assert await cluster.bound() == {"ops"}
    await cluster.close()


async def test_a_failed_install_over_a_discussion_left_here_keeps_no_record_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _Holding()
    a = await _installed_again_elsewhere(cluster)

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("refused by the test")

    monkeypatch.setattr(a, "_prepare_discussion", refuse)
    with pytest.raises(ValueError, match="refused by the test"):
        await a.install_strategy("r1", _handing_over(cluster, AGENTS))

    # A left its discussion: no record of it says the room still runs it.
    assert a.room_strategy("r1") is None
    assert await a.room_strategy_is_current("r1", Discussion) is False
    await cluster.close()


async def test_a_result_or_instruction_reaching_a_process_after_an_uninstall_is_not_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A does not look again before B installs the discussion back; what
    # reaches A in between has no queue to go to.
    monkeypatch.setattr(_driver, "POLL_SECONDS", 3.0)
    monkeypatch.setattr(_shared, "LEASE_SECONDS", 6.0)
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", 4.0)
    cluster = _Holding()
    agents = [AIChannel(n, provider=_Held(n, cluster, f"{n} answers")) for n in AGENTS]
    a = await _created(cluster, Discussion(agents))
    b = cluster.kit()
    seen: list[tuple[SpeakQueueChange, tuple[str, ...]]] = []

    @a.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: Any) -> None:
        seen.append((event.change, event.channel_ids))

    await _say(a, "@investigator take your time")
    await _turn_started(cluster)
    assert await b.uninstall_strategy("r1") is True
    result = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sre", channel_type=ChannelType.AI),
        content=TextContent(body="Result of the task"),
        addressed_to=[],
    )
    assert await a._queue_discussion_post("r1", result) is True
    await _say(a, "summarise", addressed_to=["comms"], event_type=EventType.INSTRUCTION)
    await b.install_strategy("r1", cluster.strategy(delay=0.05))
    cluster.first_turn_ends.set()  # the result waits for the floor A's turn holds

    async def handed_back() -> bool:
        events = await cluster.store.list_events("r1", limit=200)
        posted = any(e.content == result.content for e in events)
        return posted and (SpeakQueueChange.INSTRUCTION_DROPPED, ("comms",)) in seen

    await _until(handed_back)
    await cluster.close()
