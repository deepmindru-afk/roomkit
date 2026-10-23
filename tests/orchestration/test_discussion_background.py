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
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit import Agent, RoomKit
from roomkit.models.enums import EventStatus, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.orchestration.strategies.discussion import (
    Discussion,
    SpeakQueueChange,
    SpeakQueueEvent,
)
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.background import BACKGROUND_TOOL
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


async def test_a_result_being_posted_when_the_discussion_stops_is_still_posted() -> None:
    kit = RoomKit()
    comms = MockAIProvider(ai_responses=_starts_task("On it."))
    helper = MockAIProvider(ai_responses=[AIResponse(content="Findings.")])
    await _room(kit, comms, _provider("sre here"), helper, hand_back="post")
    send_event = kit.send_event
    stopped = asyncio.Event()

    async def slow(*args: Any, **kwargs: Any) -> Any:
        if (kwargs.get("metadata") or {}).get("task_id"):
            # The result is still being posted once the discussion stopped.
            await asyncio.wait_for(stopped.wait(), 5)
        return await send_event(*args, **kwargs)

    kit.send_event = slow  # type: ignore[method-assign]
    await _say(kit, "@comms please go through all the support tickets")
    discussion = kit._discussions["r1"]
    async with asyncio.timeout(5):
        while discussion.driver is None or discussion.driver.posting is None:
            await asyncio.sleep(0.005)
    await kit._uninstall_discussion("r1")
    stopped.set()
    said = await _until_said(kit, 2)

    assert said[-1].content.body.startswith("Result of the task")
    await kit.close()
