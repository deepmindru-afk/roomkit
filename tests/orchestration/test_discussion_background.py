"""Talking and working: a discussion's long work runs as tasks, off the floor
(RFC §19.7.5 rule 19, RMK-668).

A turn talks; the work runs apart. An agent that starts a task says it is on
it and its turn ends, so a question asked while the task runs is answered in
its turn. The result comes back at the front of the queue, as the agent's own
turn (relayed) or as its message, posted with no generation, never cutting a
turn under way, its names asking no one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import Agent, RoomKit
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.enums import EventStatus, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.models.hook import HookResult
from roomkit.orchestration.strategies.discussion import (
    Discussion,
    SpeakQueueChange,
    SpeakQueueEvent,
    _driver,
)
from roomkit.orchestration.strategies.discussion._posts import RESULTS_KEY, ResultPosts
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.tasks.background import BACKGROUND_TOOL
from roomkit.tools.context import _current_loop_ctx
from tests.orchestration.test_strategy_discussion import _provider, _say, _timeline
from tests.test_framework import SimpleChannel

Wait = float | Callable[[], Awaitable[Any]]


class _Paced(MockAIProvider):
    """Answers in order, answer *i* once *waits[i]* is over: that many
    seconds, or what it awaits, for a second at most (a test then fails on
    the order it reads rather than hangs). ``asked`` is set once a
    generation begins."""

    def __init__(self, answers: list[AIResponse], waits: list[Wait] | None = None) -> None:
        super().__init__(ai_responses=answers)
        self.waits = waits or []
        self.asked = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        turn = len(self.calls)
        self.asked.set()
        if turn < len(self.waits):
            wait = self.waits[turn]
            if isinstance(wait, float):
                await asyncio.sleep(wait)
            else:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(wait(), 1.0)
        return await super().generate(context)


def _starts_task(after: str = "On it.", *then: str) -> list[AIResponse]:
    """An agent that starts a background task, then says *after*, then *then*."""
    call = AIToolCall(id="c1", name=BACKGROUND_TOOL, arguments={"task": "go through the tickets"})
    return [
        AIResponse(content="", tool_calls=[call]),
        AIResponse(content=after),
        *(AIResponse(content=t) for t in then),
    ]


async def _room(
    kit: RoomKit,
    comms: MockAIProvider,
    sre: MockAIProvider,
    helper: MockAIProvider,
    *,
    hand_back: str = "relay",
    **options: Any,
) -> None:
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(Agent("helper", provider=helper))
    agents = [Agent("comms", provider=comms), Agent("sre", provider=sre)]
    kit.setup_background(agents[0], worker="helper", hand_back=hand_back)
    await kit.create_room(room_id="r1", orchestration=Discussion(agents, **options))
    await kit.attach_channel("r1", "ops")


async def _until_said(kit: RoomKit, count: int, timeout: float = 5.0) -> list[RoomEvent]:
    async with asyncio.timeout(timeout):
        while True:
            said = [
                e
                for e in await _timeline(kit)
                if e.source.channel_id != "ops" and e.status == EventStatus.DELIVERED
            ]
            if len(said) >= count:
                return said
            await asyncio.sleep(0.01)


def _handed_back(kit: RoomKit) -> asyncio.Event:
    """Set once a task's result was handed to the room's discussion, which
    queues it or, over, posts it once no turn runs."""
    handed = asyncio.Event()
    queue = kit._queue_discussion_post

    async def spy(room_id: str, event: RoomEvent) -> bool:
        try:
            return await queue(room_id, event)
        finally:
            handed.set()

    kit._queue_discussion_post = spy  # type: ignore[method-assign]
    return handed


async def _over(kit: RoomKit) -> None:
    while not (queue := kit.speak_queue("r1")) or not queue.over:
        await asyncio.sleep(0.01)


def _lines(events: list[RoomEvent]) -> list[tuple[str, str]]:
    return [(e.source.channel_id, e.content.body) for e in events]


async def test_a_question_asked_while_a_task_runs_is_answered_in_its_turn() -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it.", "12 tickets, all Canadian."))
    # The task ends once sre answered, or a second later if sre cannot.
    helper = _Paced([AIResponse(content="Findings: 12 tickets.")], [lambda: _until_said(kit, 2)])
    await _room(kit, comms, _provider("6 percent."), helper)

    await _say(kit, "@comms please go through all the support tickets")
    await _until_said(kit, 1)
    await _say(kit, "@sre quick one: the error rate?")
    said = await _until_said(kit, 3)

    assert _lines(said) == [
        ("comms", "On it."),
        ("sre", "6 percent."),
        ("comms", "12 tickets, all Canadian."),
    ]
    # The relayed result opened comms's own turn, told as the worker's output.
    told = str(comms.calls[-1].messages[-1].content)
    assert "Findings: 12 tickets." in told and "“go through the tickets”" in told
    await kit.close()


async def test_a_posted_result_is_the_agent_s_message_without_a_generation() -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    helper = MockAIProvider(ai_responses=[AIResponse(content="Findings: 12 tickets.")])
    await _room(kit, comms, _provider("sre here"), helper, hand_back="post")

    await _say(kit, "@comms please go through all the support tickets")
    said = await _until_said(kit, 2)

    assert _lines(said) == [
        ("comms", "On it."),
        ("comms", "Result of the task “go through the tickets”:\n\nFindings: 12 tickets."),
    ]
    assert len(comms.calls) == 2  # the tool call and its sentence: no turn for the result
    await kit.close()


async def test_a_posted_result_waits_for_the_turn_under_way() -> None:
    kit = RoomKit()
    # The task ends while comms is still saying it is on it.
    comms = _Paced(_starts_task("On it."), [0.0, _handed_back(kit).wait])
    helper = MockAIProvider(ai_responses=[AIResponse(content="Findings.")])
    await _room(kit, comms, _provider("sre here"), helper, hand_back="post")

    await _say(kit, "@comms please go through all the support tickets")
    said = await _until_said(kit, 2)

    assert [who for who, _ in _lines(said)] == ["comms", "comms"]
    assert said[0].content.body == "On it."
    assert said[0].index < said[1].index
    await kit.close()


async def test_the_names_a_posted_result_carries_ask_no_one() -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    sre = _provider("sre here")
    helper = MockAIProvider(ai_responses=[AIResponse(content="Ask @sre about the 422s.")])
    await _room(kit, comms, sre, helper, hand_back="post")
    ended: list[tuple[str, ...]] = []

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: Any) -> None:
        if event.change == SpeakQueueChange.TURN_ENDED:
            ended.append(event.channel_ids)

    await _say(kit, "@comms please go through all the support tickets")
    async with asyncio.timeout(5):
        while len(ended) < 2:  # comms's turn, then its result's: both ends recorded
            await asyncio.sleep(0.01)

    queue = kit.speak_queue("r1")
    assert queue is not None and queue.queue == () and queue.speaking is None
    assert sre.calls == []
    await kit.close()


async def test_a_result_still_lands_once_the_discussion_is_over() -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    helper = _Paced([AIResponse(content="Findings.")], [lambda: _over(kit)])
    await _room(kit, comms, _provider("sre here"), helper, hand_back="post", max_turns=1)

    await _say(kit, "@comms please go through all the support tickets")
    said = await _until_said(kit, 2)

    assert kit.speak_queue("r1").over
    assert said[-1].content.body.startswith("Result of the task")
    assert said[-1].type == EventType.MESSAGE
    await kit.close()


async def test_a_result_back_once_over_waits_for_the_last_turn() -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    # The result comes back while sre, the last turn, speaks.
    sre = _Paced([AIResponse(content="sre here")], [_handed_back(kit).wait])
    helper = _Paced([AIResponse(content="Findings.")], [sre.asked.wait])
    await _room(kit, comms, sre, helper, hand_back="post", max_turns=2)

    await _say(kit, "@comms please go through all the support tickets")
    await _until_said(kit, 1)
    await _say(kit, "@sre quick one")
    said = await _until_said(kit, 3)

    assert [who for who, _ in _lines(said)] == ["comms", "sre", "comms"]
    assert said[2].content.body.startswith("Result of the task")
    await kit.close()


async def test_results_held_when_another_process_uninstalls_are_posted() -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    sre = _Paced([AIResponse(content="sre here")], [5.0])
    # The result comes back while sre speaks, and waits behind its turn.
    helper = _Paced([AIResponse(content="Findings.")], [sre.asked.wait])
    handed = _handed_back(kit)
    await _room(kit, comms, sre, helper, hand_back="post")

    await _say(kit, "@comms please go through all the support tickets")
    await _until_said(kit, 1)
    await _say(kit, "@sre quick one")
    discussion = kit._discussions["r1"]
    await asyncio.wait_for(handed.wait(), 5)  # the result, queued behind sre's turn
    await kit._forget_discussion("r1", discussion, installs=discussion.installs)
    posting = set(kit._background_runs)  # the posts the forget left to the kit
    said = await _until_said(kit, 2)

    posted = [e for e in said if e.content.body.startswith("Result of the task")]
    assert len(posted) == 1
    await asyncio.wait(posting, timeout=5)
    assert [e.id for e in await _timeline(kit) if e.content.body == posted[0].content.body] == [
        posted[0].id
    ]
    await kit.close()


async def _until_cut() -> None:
    """Until the turn is cut, as a streaming provider's next chunk would
    reach the steering check."""
    loop = _current_loop_ctx.get()
    if loop is not None:
        await loop.cancel_event.wait()


async def _bindings(kit: RoomKit) -> set[str]:
    return {b.channel_id for b in await kit.store.list_bindings("r1")}


def _troubles(caplog: pytest.LogCaptureFixture) -> list[str]:
    """What a run logged as an error, or as a result lost."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR
        or (r.levelno >= logging.WARNING and "not posted" in r.getMessage())
    ]


def _posted(events: list[RoomEvent]) -> list[tuple[str, EventStatus]]:
    return [
        (e.source.channel_id, e.status)
        for e in events
        if e.content.body.startswith("Result of the task")
    ]


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "under_the_lock"])
async def test_a_result_queued_when_the_strategy_is_uninstalled_is_posted_before_its_agent_leaves(
    locked: bool, caplog: pytest.LogCaptureFixture
) -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    sre = _Paced([AIResponse(content="sre here")], [_until_cut])
    # The result comes back while sre speaks, and waits behind its turn.
    helper = _Paced([AIResponse(content="Findings.")], [sre.asked.wait])
    handed = _handed_back(kit)
    await _room(kit, comms, sre, helper, hand_back="post")
    await _say(kit, "@comms please go through all the support tickets")
    await _until_said(kit, 1)
    await _say(kit, "@sre quick one")
    await asyncio.wait_for(handed.wait(), 5)

    async with kit.lock_manager.locked("r1") if locked else contextlib.nullcontext():
        assert await kit.uninstall_strategy("r1") is True

    # Posted as comms's message, then comms and sre, which the install
    # attached, detached (RFC §19.7.5 rule 19, §19.7).
    assert _posted(await _timeline(kit)) == [("comms", EventStatus.DELIVERED)]
    assert await _bindings(kit) == {"ops"}
    await kit.close()
    assert _troubles(caplog) == []


@pytest.mark.parametrize(
    "case", ["unlocked", "under_the_lock", "under_the_lock_after_its_binding_read"]
)
async def test_a_result_being_posted_when_the_strategy_is_uninstalled_is_posted_once_first(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    helper = MockAIProvider(ai_responses=[AIResponse(content="Findings.")])
    await _room(kit, comms, _provider("sre here"), helper, hand_back="post")
    discussion: Any = None
    stopped, read, locked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    send_event, get_binding = kit.send_event, kit._get_binding

    def posting() -> bool:
        return discussion is not None and discussion.driver.posting is not None

    async def slow_send(*args: Any, **kwargs: Any) -> Any:
        if posting() and case != "under_the_lock_after_its_binding_read":
            # The post turn's commit has not read its agent's binding yet
            # once the discussion stopped.
            await asyncio.wait_for(stopped.wait(), 5)
        return await send_event(*args, **kwargs)

    async def slow_binding(room_id: str, channel_id: str) -> Any:
        binding = await get_binding(room_id, channel_id)
        if channel_id == "comms" and posting() and case.endswith("binding_read"):
            # It read the binding, and waits for the room lock the host takes.
            read.set()
            await asyncio.wait_for(locked.wait(), 5)
        return binding

    kit.send_event = slow_send  # type: ignore[method-assign]
    kit._get_binding = slow_binding  # type: ignore[method-assign]
    await _say(kit, "@comms please go through all the support tickets")
    discussion = kit._discussions["r1"]
    async with asyncio.timeout(5):
        while not posting():
            await asyncio.sleep(0.005)
    stop = discussion.driver.stop

    async def stopping() -> None:
        await stop()
        stopped.set()

    discussion.driver.stop = stopping
    async with kit.lock_manager.locked("r1") if case != "unlocked" else contextlib.nullcontext():
        locked.set()
        if case.endswith("binding_read"):
            await asyncio.wait_for(read.wait(), 5)
        # Under the lock, the commit under way cannot be waited for: it needs it.
        async with asyncio.timeout(1):
            assert await kit.uninstall_strategy("r1") is True

    assert _posted(await _timeline(kit)) == [("comms", EventStatus.DELIVERED)]
    assert await _bindings(kit) == {"ops"}
    await kit.close()  # the commit under way has ended: a replay
    assert _posted(await _timeline(kit)) == [("comms", EventStatus.DELIVERED)]
    assert _troubles(caplog) == []


async def _hold_in_a(a: RoomKit, held: str, monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Run *a*'s discussion until it holds comms's result as *held*: queued
    behind sre's turn, waiting for the floor once the discussion is over (sre
    speaks last), or being committed by its post turn, which reads comms's
    binding only once the returned callback is called."""
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    sre = _Paced([AIResponse(content="sre here")], [_until_cut])
    helper = _Paced([AIResponse(content="Findings.")], [sre.asked.wait])
    release = asyncio.Event()
    if held == "posting":
        sre = _provider("sre here")
        helper = _Paced([AIResponse(content="Findings.")], [0.02])
        get_binding = a._get_binding
        reading = asyncio.Event()

        async def slow_binding(room_id: str, channel_id: str) -> Any:
            discussion = a._discussions.get(room_id)
            if channel_id == "comms" and discussion is not None and discussion.driver.posting:
                reading.set()
                await asyncio.wait_for(release.wait(), 5)
            return await get_binding(room_id, channel_id)

        monkeypatch.setattr(a, "_get_binding", slow_binding)
    handed = _handed_back(a)
    over = {"max_turns": 2} if held == "waiting_for_the_floor" else {}
    await _room(a, comms, sre, helper, hand_back="post", **over)
    await _say(a, "@comms please go through all the support tickets")
    await _until_said(a, 1)
    if held == "posting":
        await asyncio.wait_for(reading.wait(), 5)
    else:
        await _say(a, "@sre quick one")
        await asyncio.wait_for(handed.wait(), 5)
    return release.set


@pytest.mark.parametrize("held", ["queued", "waiting_for_the_floor", "posting"])
async def test_a_result_another_process_holds_is_posted_once_by_the_one_that_uninstalls(
    held: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(_driver, "POLL_SECONDS", 0.05)
    store, locks = InMemoryStore(), InMemoryLockManager()
    a = RoomKit(store=store, lock_manager=locks)
    release = await _hold_in_a(a, held, monkeypatch)
    # b serves the room too and only follows the discussion (rule 16).
    b = RoomKit(store=store, lock_manager=locks)
    b.register_channel(SimpleChannel("ops"))

    assert await b.uninstall_strategy("r1") is True

    # Posted by b, as comms's message, before it detached the agents.
    assert _posted(await _timeline(a)) == [("comms", EventStatus.DELIVERED)]
    assert await _bindings(a) == {"ops"}
    release()
    async with asyncio.timeout(5):
        while "r1" in a._discussions:  # a reads the uninstall and leaves
            await asyncio.sleep(0.01)
    await a.close()  # a's own post of the result has ended by then: a replay
    await b.close()
    assert _posted(await _timeline(a)) == [("comms", EventStatus.DELIVERED)]
    assert _troubles(caplog) == []


async def test_a_result_no_list_holds_is_posted_by_its_holder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def lists_nothing(self: ResultPosts, posted: RoomEvent) -> None:
        """A process of an earlier version keeps its results in memory only."""

    monkeypatch.setattr(ResultPosts, "keep", lists_nothing)
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    handed = _handed_back(kit)
    # sre speaks until the result is queued behind its turn.
    sre = _Paced([AIResponse(content="sre here")], [handed.wait])
    helper = _Paced([AIResponse(content="Findings.")], [sre.asked.wait])
    await _room(kit, comms, sre, helper, hand_back="post")
    await _say(kit, "@comms please go through all the support tickets")
    await _until_said(kit, 1)
    await _say(kit, "@sre quick one")
    said = await _until_said(kit, 3)

    assert [who for who, _ in _lines(said)] == ["comms", "sre", "comms"]
    assert _posted(said) == [("comms", EventStatus.DELIVERED)]
    await kit.close()


_SECRET = "Findings: customer 4521 owes 9,300 dollars."


@pytest.mark.parametrize("hidden", ["by_the_agent_s_binding", "by_a_hook"])
async def test_a_result_waiting_to_be_posted_is_read_nowhere_its_message_is_not(
    hidden: str,
) -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    sre = _Paced([AIResponse(content="sre here")], [_until_cut])
    helper = _Paced([AIResponse(content=_SECRET)], [sre.asked.wait])
    handed = _handed_back(kit)
    await _room(kit, comms, sre, helper, hand_back="post")
    if hidden == "by_the_agent_s_binding":
        # comms's messages reach the intelligence channels only, never ops.
        await kit.set_visibility("r1", "comms", "intelligence")
    else:

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def moderate(event: RoomEvent, context: Any) -> HookResult:
            body = getattr(event.content, "body", "")
            return HookResult.block("account data") if "owes" in body else HookResult.allow()

    await _say(kit, "@comms please go through all the support tickets")
    await _until_said(kit, 1)
    await _say(kit, "@sre quick one")
    await asyncio.wait_for(handed.wait(), 5)

    # Queued behind sre's turn: the room, as any reader of it gets it, holds
    # nothing of it (RFC §17.2).
    assert _SECRET not in (await kit.get_room("r1")).model_dump_json()
    assert await kit.uninstall_strategy("r1") is True

    posted = [e for e in await _timeline(kit) if e.content.body.endswith(_SECRET)]
    if hidden == "by_the_agent_s_binding":
        assert [(e.status, e.visibility) for e in posted] == [
            (EventStatus.DELIVERED, "intelligence")
        ]
    else:
        assert [e.status for e in posted] == [EventStatus.BLOCKED]
    assert await kit.store.get_private_state("r1", RESULTS_KEY) is None
    await kit.close()


async def test_a_result_a_closing_kit_did_not_post_is_posted_by_the_next_uninstall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_driver, "POLL_SECONDS", 0.05)
    store, locks = InMemoryStore(), InMemoryLockManager()
    a = RoomKit(store=store, lock_manager=locks)
    await _hold_in_a(a, "queued", monkeypatch)
    await a.close()  # a closing kit posts nothing; the result stays kept
    assert _posted(await _timeline(a)) == []
    b = RoomKit(store=store, lock_manager=locks)
    b.register_channel(SimpleChannel("ops"))

    assert await b.uninstall_strategy("r1") is True

    assert _posted(await _timeline(b)) == [("comms", EventStatus.DELIVERED)]
    assert await _bindings(b) == {"ops"}
    await b.close()
