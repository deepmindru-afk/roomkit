"""A discussion uninstalled, then installed again, across processes (RFC §19.7.5 rule 16).

A process that read the discussion uninstalled leaves it, even when another
process installed one again before it looked; its host installing again
starts the discussion afresh, never in the room it left.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import Discussion, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.orchestration.strategies.discussion import _driver, _shared
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.orchestration.test_discussion_processes import (
    KEPT,
    _Cluster,
    _say,
    _turn_started,
    _until,
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll and lease on a test's time scale."""
    monkeypatch.setattr(_driver, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(_shared, "LEASE_SECONDS", 0.6)
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", 0.4)


class _HandsOver(MockAIProvider):
    """investigator asks @sre in every answer, the first of the cluster slow."""

    def __init__(self, name: str, log: list[str]) -> None:
        body = "@sre please check" if name == "investigator" else f"{name} here"
        super().__init__(ai_responses=[AIResponse(content=body)])
        self.who, self.log = name, log

    async def generate(self, context: Any) -> AIResponse:
        first = f"+{self.who}" not in self.log
        self.log.append(f"+{self.who}")
        await asyncio.sleep(1.0 if first and self.who == "investigator" else 0.05)
        return await super().generate(context)


def _handing_over(cluster: _Cluster) -> Discussion:
    return Discussion([AIChannel(n, provider=_HandsOver(n, cluster.log)) for n in KEPT])


async def _created(cluster: _Cluster) -> RoomKit:
    kit = cluster.kit()
    await kit.create_room(room_id="r1", orchestration=_handing_over(cluster))
    await kit.attach_channel("r1", "ops")
    return kit


async def _cut_by_an_uninstall_elsewhere(cluster: _Cluster, a: RoomKit, b: RoomKit) -> None:
    """A gives investigator's slow turn; B uninstalls and A's watch cuts it."""
    await _say(a, "@investigator take your time")
    await _turn_started(cluster)
    assert await b.uninstall_strategy("r1") is True
    await asyncio.sleep(0.2)  # A's watch has read the discussion gone


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
    cluster = _Cluster()
    a, b = await _created(cluster), cluster.kit()
    await _cut_by_an_uninstall_elsewhere(cluster, a, b)
    await b.install_strategy("r1", _handing_over(cluster))  # before A looks again

    if host_asks:
        assert await a.room_strategy_is_current("r1", Discussion) is False

    async def left() -> bool:
        return a.room_strategy("r1") is None

    await _until(left)
    await asyncio.sleep(1.0)  # past A's cut turn
    await _sre_answers_investigator(cluster, b)
    await cluster.close()


async def test_installing_again_over_a_discussion_seen_uninstalled_starts_it_afresh() -> None:
    cluster = _Cluster()
    a, b = await _created(cluster), cluster.kit()
    await _cut_by_an_uninstall_elsewhere(cluster, a, b)
    # A's host still wants the discussion: it installs before A looks again.
    await a.install_strategy("r1", _handing_over(cluster))

    assert await a.room_strategy_is_current("r1", Discussion) is True
    await asyncio.sleep(1.0)  # past A's cut turn
    await _sre_answers_investigator(cluster, a)
    await cluster.close()


class _SlowContext(InMemoryStore):
    """A store whose next room context, once armed, takes a while to come
    back: the round trip a Postgres store takes."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: tuple[asyncio.Event, asyncio.Event] | None = None

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
    await asyncio.sleep(0.1)
    b = await cluster.process(install=False)
    assert await b.uninstall_strategy("r1") is True
    held = a._discussions["r1"]

    entered, release = store.arm()
    install = asyncio.create_task(a.install_strategy("r1", cluster.strategy(delay=0.05)))
    await asyncio.wait_for(entered.wait(), 2.0)
    if reader == "driver":
        held.wake()
        await asyncio.sleep(0.1)
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
