"""``kit.close()`` ends a turn someone awaits as it ends the same turn on the
neighbouring door (RMK-526, RFC §23.3, §10.1 step 18).

An inline delegation ends ``cancelled``, ON_TASK_COMPLETED fired before the
store is sealed, as a background one does. An awaited room turn returns its
``cancelled`` turn, as a deferred one does, never a ``CancelledError`` its
caller did not ask for. A ``close()`` called from inside work the kit holds
does not wait for that work.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, RoomStatus
from roomkit.channels.agent import Agent
from roomkit.core import framework
from roomkit.core.task_utils import hold_task
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.room import RoomTimers
from roomkit.models.trace import ProtocolTrace
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AIToolCall,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.tasks.posting import send_posted
from tests.test_framework import SimpleChannel


class _SlowWords(MockAIProvider):
    """Streams a long answer word by word, and says when it started; with
    *answers_first*, its first generation answers at once."""

    def __init__(self, started: asyncio.Event, *, answers_first: bool = False) -> None:
        super().__init__(streaming=True)
        self.started = started
        self.answers_first = answers_first

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        self.calls.append(context)
        if self.answers_first and len(self.calls) == 1:
            yield StreamTextDelta(text="Quick.")
            yield StreamDone(finish_reason="stop", usage={"input_tokens": 1, "output_tokens": 1})
            return
        yield StreamTextDelta(text="The ")
        # The reader reads what came before the test hears it (RFC §10.6.4).
        await asyncio.sleep(0.05)
        self.started.set()
        for i in range(2000):
            await asyncio.sleep(0.01)
            yield StreamTextDelta(text=f"w{i} ")
        yield StreamDone(finish_reason="stop", usage={"input_tokens": 5, "output_tokens": 50})


def _ended_tasks(kit: RoomKit) -> list[tuple[str, Any]]:
    ended: list[tuple[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_task(event: Any, context: Any) -> None:
        ended.append((str(event.metadata.get("task_status")), event.metadata.get("error")))

    return ended


# -- a delegation under kit.close() --------------------------------------------


@pytest.mark.parametrize("shared", [False, True], ids=["alone", "shared-channel"])
@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
async def test_a_delegation_ends_cancelled_with_its_hook(
    wait: bool, shared: bool, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("worker", provider=_SlowWords(started)))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    ended = _ended_tasks(kit)
    share = ["sms"] if shared else None

    delegation = asyncio.ensure_future(
        kit.delegate("r", "worker", "Go.", wait=wait, share_channels=share)
    )
    if not wait:
        await delegation
    await started.wait()
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await kit.close()
        task = await asyncio.wait_for(delegation, timeout=5.0)

    assert task.result is not None
    assert (task.result.status, task.result.error, task.result.output) == (
        "cancelled",
        "cancelled",
        None,
    )
    assert ended == [("cancelled", "cancelled")]
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_a_close_from_a_held_run_does_not_wait_for_it() -> None:
    kit = RoomKit()
    closed = asyncio.Event()

    async def run() -> None:
        await kit.close()
        closed.set()

    hold_task(kit, run())
    await asyncio.wait_for(closed.wait(), timeout=5.0)


@pytest.mark.in_process(
    "the kit closes from a tool in its own turn: an executor elsewhere holds no kit"
)
async def test_a_close_from_an_inline_workers_tool_does_not_wait_for_it() -> None:
    kit = RoomKit()
    closed = asyncio.Event()

    async def shutdown(name: str, arguments: dict[str, Any]) -> str:
        await kit.close()
        closed.set()
        return "closed"

    tool = {"name": "shutdown", "description": "Shut down.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="shutdown", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    kit.register_channel(Agent("worker", provider=provider, tools=[tool], tool_handler=shutdown))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")

    delegation = asyncio.ensure_future(kit.delegate("r", "worker", "Go.", wait=True))
    await asyncio.wait_for(closed.wait(), timeout=5.0)
    await asyncio.wait_for(asyncio.gather(delegation, return_exceptions=True), timeout=5.0)


# -- a room turn under kit.close() ----------------------------------------------


@pytest.mark.parametrize(
    ("door", "entry"),
    [
        ("awaited", "process_inbound"),
        ("deferred", "process_inbound"),
        ("awaited", "send_event"),
        ("awaited", "regenerate_response"),
    ],
)
async def test_a_room_turn_returns_cancelled_to_its_caller(
    door: str, entry: str, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("tx"))
    provider = _SlowWords(started, answers_first=entry == "regenerate_response")
    kit.register_channel(Agent("agent", provider=provider))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    responses: list[str] = []
    message = InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Go."))
    if entry == "regenerate_response":
        await kit.process_inbound(message)

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def on_response(event: Any, context: Any) -> None:
        responses.append(str(event.loop_end_reason))

    if entry == "send_event":
        call: Any = asyncio.ensure_future(kit.send_event("r", "tx", TextContent(body="Go.")))
    elif entry == "regenerate_response":
        call = asyncio.ensure_future(kit.regenerate_response("r"))
    else:
        call = asyncio.ensure_future(
            kit.process_inbound(message, defer_delivery=door == "deferred")
        )
    await started.wait()
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await kit.close()
        result = await asyncio.wait_for(call, timeout=5.0)
        if door == "deferred":
            await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    if entry != "send_event":
        turns = dict(result.response_metadata).get("turns", {})
        assert turns["agent"]["loop_end_reason"] == "cancelled"
        assert result.error is None
    assert responses == []
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []


# -- a close from inside work the kit holds -------------------------------------


def _closing_agent(kit: RoomKit, closed: asyncio.Event) -> Agent:
    """An agent whose one tool closes the kit, then answers."""

    async def shutdown(name: str, arguments: dict[str, Any]) -> str:
        await kit.close()
        closed.set()
        return "closed"

    tool = {"name": "shutdown", "description": "Shut down.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="shutdown", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    return Agent("agent", provider=provider, tools=[tool], tool_handler=shutdown)


@pytest.mark.in_process(
    "the kit closes from a tool in its own turn: an executor elsewhere holds no kit"
)
@pytest.mark.parametrize(
    "door", ["room-turn", "inline", "inline-shared", "background"], ids=lambda d: d
)
async def test_a_close_from_a_tool_returns_and_seals_the_kit(door: str) -> None:
    """A tool that closes the kit while its turn is awaited, or runs as a
    delegated task, does not wait for itself (RMK-526)."""
    kit = RoomKit()
    closed = asyncio.Event()
    kit.register_channel(_closing_agent(kit, closed))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")

    if door == "room-turn":
        await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
        message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))
        call: Any = asyncio.ensure_future(kit.process_inbound(message))
    else:
        share = ["sms"] if door == "inline-shared" else None
        call = asyncio.ensure_future(
            kit.delegate("r", "agent", "Go.", wait=door != "background", share_channels=share)
        )
    await asyncio.wait_for(closed.wait(), timeout=5.0)
    await asyncio.wait_for(asyncio.gather(call, return_exceptions=True), timeout=5.0)

    assert kit._closed


async def test_a_held_inline_run_starts_before_a_close_can_cut_it() -> None:
    """An inline task's run is never cancelled unstarted, its end never run:
    it starts as it is held."""
    kit = RoomKit()
    steps: list[str] = []

    async def work() -> None:
        steps.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("ended")
            raise

    run = hold_task(kit, work(), eager=True)
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)

    assert steps == ["started", "ended"]


@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
async def test_a_task_announced_as_the_kit_closes_ends_cancelled(
    wait: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kit's close lands between the task's announcement and its run: it
    ends cancelled on both doors, its worker never runs (RFC §23.3)."""
    started = asyncio.Event()
    kit = RoomKit()
    provider = _SlowWords(started)
    kit.register_channel(Agent("worker", provider=provider))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    announce = kit._announce_task

    async def announce_then_close(handle: Any) -> None:
        await announce(handle)
        await kit.close()

    monkeypatch.setattr(kit, "_announce_task", announce_then_close)
    task = await asyncio.wait_for(kit.delegate("r", "worker", "Go.", wait=wait), timeout=5.0)

    assert task.result is not None and task.result.status == "cancelled"
    assert provider.calls == []


# -- what the kit holds as it closes --------------------------------------------


async def test_a_task_started_while_the_kit_drains_ends_with_its_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An observer still running as the kit closes commits into a room whose
    timer fell due: the timer's hooks start while close() drains the
    observers, and close() returns with them ended, never left running
    against a closed kit."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.create_room(room_id="r2", timers=RoomTimers(inactive_after_seconds=0))
    draining = asyncio.Event()
    drain = kit.hook_engine.drain_observers

    async def draining_observers() -> None:
        draining.set()
        await drain()

    monkeypatch.setattr(kit.hook_engine, "drain_observers", draining_observers)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: RoomEvent, ctx: RoomContext) -> None:
        await draining.wait()
        record = RoomEvent(
            room_id="r2",
            source=EventSource(channel_id="sms", channel_type="sms"),
            content=TextContent(body="audit record"),
        )
        await kit.commit_event("r2", record)

    @kit.hook(HookTrigger.ON_ROOM_PAUSED, name="paused")
    async def paused(event: RoomEvent, ctx: RoomContext) -> None:
        await asyncio.Event().wait()

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hi")),
        room_id="r1",
    )
    await kit.close()

    running = [t.get_name() for t in asyncio.all_tasks() if not t.done()]
    assert [name for name in running if name.startswith("roomkit-")] == []


async def test_a_background_run_started_while_the_kit_drains_ends_with_its_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An observer still running as the kit closes starts work the kit holds
    (a hand-off greeting's turn): started while close() drains the
    observers, it is cut with the kit's other background work, never left
    running past close()."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    draining = asyncio.Event()
    drain = kit.hook_engine.drain_observers

    async def draining_observers() -> None:
        draining.set()
        await drain()

    monkeypatch.setattr(kit.hook_engine, "drain_observers", draining_observers)
    runs: list[asyncio.Task[None]] = []

    async def greeting() -> None:
        await asyncio.Event().wait()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="greet")
    async def greet(event: RoomEvent, ctx: RoomContext) -> None:
        await draining.wait()
        runs.append(hold_task(kit, greeting()))

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hi")),
        room_id="r1",
    )
    await kit.close()

    assert [run.done() for run in runs] == [True]


class _HoldsAsItCloses(SimpleChannel):
    """A channel whose close starts work the kit holds (a hand-back its
    session's end starts)."""

    def __init__(self, channel_id: str, hold: Callable[[], None]) -> None:
        super().__init__(channel_id)
        self._hold = hold

    async def close(self) -> None:
        self._hold()
        await super().close()


@pytest.mark.parametrize(
    "when", ["before-its-first-step", "as-a-channel-closes", "after-the-close"]
)
async def test_a_run_held_around_the_close_is_cut_with_its_end_run(when: str) -> None:
    """Work the kit holds (a strategy's background run, a hand-back, a
    greeting's turn) started however late in the kit's close, or after it,
    is cut as close cuts the runs it holds: once it started, so its own end
    runs (a run's terminal entry, a task's completion), and never left
    running past close() (RFC §19.7.3, §23.3)."""
    kit = RoomKit()
    steps: list[str] = []
    runs: list[asyncio.Task[None]] = []

    async def work() -> None:
        steps.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("ended")
            raise

    def hold() -> None:
        runs.append(hold_task(kit, work()))

    if when == "as-a-channel-closes":
        kit.register_channel(_HoldsAsItCloses("sms", hold))
    if when == "before-its-first-step":
        hold()
    await kit.close()
    if when == "after-the-close":
        hold()
        await asyncio.wait(runs, timeout=1.0)

    assert [run.cancelled() for run in runs] == [True]
    assert steps == ["started", "ended"]


class _TracesAsItCloses(SimpleChannel):
    """A channel whose close emits a protocol trace (a SIP BYE), and returns
    once the trace's hook has started."""

    def __init__(self, channel_id: str, hook_started: asyncio.Event) -> None:
        super().__init__(channel_id)
        self._hook_started = hook_started

    async def close(self) -> None:
        self.emit_trace(
            ProtocolTrace(
                channel_id=self.channel_id,
                direction="outbound",
                protocol="sip",
                summary="BYE",
                room_id="r1",
            )
        )
        await asyncio.wait_for(self._hook_started.wait(), 1.0)
        await super().close()


class _SlowRecords(InMemoryStore):
    """A store whose commit of a ``record`` takes a few steps, and that says
    how many were still in flight when it closed."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.in_flight_at_close: list[int] = []

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        if getattr(event.content, "body", None) != "record":
            return await super().commit_event(room_id, event, **kw)
        self.in_flight += 1
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            return await super().commit_event(room_id, event, **kw)
        finally:
            self.in_flight -= 1

    async def close(self) -> None:
        self.in_flight_at_close.append(self.in_flight)
        await super().close()


def _record() -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sip", channel_type="sms"),
        content=TextContent(body="record"),
    )


@pytest.mark.parametrize("its_end", ["ends", "holds-a-run", "commits-unawaited"])
async def test_a_hook_task_started_as_a_channel_closes_ends_with_the_close(its_end: str) -> None:
    """A channel's close starts a hook of the kit's (its last trace's
    ON_PROTOCOL_TRACE): the hook is cut before close() returns, its end run,
    and what its end starts (a run the kit holds, a record it commits
    without awaiting it) ends before the store is released."""
    store = _SlowRecords()
    kit = RoomKit(store=store)
    hook_started = asyncio.Event()
    kit.register_channel(_TracesAsItCloses("sip", hook_started))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sip")
    steps: list[str] = []

    async def late_run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)  # its end: a terminal entry it posts
            steps.append("run ended")
            raise

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE, execution=HookExecution.ASYNC, name="audit")
    async def audit(trace: ProtocolTrace, ctx: RoomContext) -> None:
        steps.append("started")
        hook_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("ended")
            if its_end == "holds-a-run":
                hold_task(kit, late_run())
            if its_end == "commits-unawaited":
                asyncio.get_running_loop().create_task(kit.commit_event("r1", _record()))
            raise

    await kit.close()

    assert steps == ["started", "ended", *(["run ended"] if its_end == "holds-a-run" else [])]
    assert store.in_flight_at_close == [0]


class _SaysByeAsItCloses(SimpleChannel):
    """A channel whose close sends a last message into its room."""

    kit: RoomKit | None = None

    async def close(self) -> None:
        assert self.kit is not None
        await self.kit.send_event("r1", self.channel_id, TextContent(body="bye"))
        await super().close()


async def test_an_observer_started_as_a_channel_closes_ends_before_the_close_returns() -> None:
    """A channel's close sends into its room, and the ASYNC observers of that
    event's BEFORE_BROADCAST chain start then: close() waits for them, as it
    waits for the observers already running, before it releases the store."""
    kit = RoomKit()
    channel = _SaysByeAsItCloses("sms")
    channel.kit = kit
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    observed: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: RoomEvent, ctx: RoomContext) -> None:
        for _ in range(5):
            await asyncio.sleep(0)  # an observer that takes a few steps
        observed.append(getattr(event.content, "body", ""))

    await kit.close()

    assert observed == ["bye"]


class _WritesSlowlyAsItCloses(SimpleChannel):
    """A channel whose close starts a record and takes longer than the kit's
    closing budget; the record's commit ends just after."""

    kit: RoomKit | None = None
    store: _GatedRecords | None = None

    async def close(self) -> None:
        assert self.kit is not None and self.store is not None
        asyncio.get_running_loop().create_task(self.kit.commit_event("r1", _record()))
        await asyncio.sleep(0.06)  # past the 0.05 s budget
        asyncio.get_running_loop().call_later(0.01, self.store.answer.set)
        await super().close()


class _GatedRecords(_SlowRecords):
    """A ``record``'s commit that waits for *answer*."""

    def __init__(self) -> None:
        super().__init__()
        self.answer = asyncio.Event()

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        if getattr(event.content, "body", None) == "record":
            self.in_flight += 1
            try:
                await self.answer.wait()
            finally:
                self.in_flight -= 1
        return await InMemoryStore.commit_event(self, room_id, event, **kw)


@pytest.mark.wallclock  # the channels' closes spend the kit's closing budget
async def test_what_a_slow_channel_close_starts_has_a_budget_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The channels' closes outlast the kit's closing budget, and one of
    them started a decided write: close() waits for it on a budget of its
    own, counted from the channels' end, before it releases the store."""
    monkeypatch.setattr(framework, "CLOSE_WAIT_S", 0.05)
    store = _GatedRecords()
    kit = RoomKit(store=store)
    channel = _WritesSlowlyAsItCloses("sip")
    channel.kit, channel.store = kit, store
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sip")

    await kit.close()

    assert store.in_flight_at_close == [0]


async def test_a_caller_waiting_for_its_injected_events_returns_when_the_close_cuts_them() -> None:
    """process_inbound committed its event and waits for the pass of what
    its hook injected when the kit closes and cuts that pass: the caller
    returns its committed event, never a CancelledError it did not ask for."""
    kit = RoomKit()
    for channel_id in ("sms1", "sms2"):
        kit.register_channel(SimpleChannel(channel_id))
    await kit.create_room(room_id="r1")
    for channel_id in ("sms1", "sms2"):
        await kit.attach_channel("r1", channel_id)
    in_pass = asyncio.Event()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="gate")
    async def gate(event: RoomEvent, ctx: RoomContext) -> HookResult:
        body = getattr(event.content, "body", "")
        if body == "injected":
            in_pass.set()
            await asyncio.Event().wait()
        if body != "hello":
            return HookResult.allow()
        injected = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms2", channel_type="sms"),
            content=TextContent(body="injected"),
        )
        return HookResult(
            action="allow",
            injected_events=[InjectedEvent(event=injected, target_channel_ids=["sms2"])],
        )

    message = InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hello"))
    caller = asyncio.create_task(kit.process_inbound(message, room_id="r1"))
    await asyncio.wait_for(in_pass.wait(), 1)
    await kit.close()

    result = await asyncio.wait_for(caller, 1)
    assert result.event is not None and result.event.content == TextContent(body="hello")


async def test_a_timer_check_waiting_for_its_hooks_returns_when_the_close_cuts_them() -> None:
    """check_room_timers() paused the room and waits for its ON_ROOM_PAUSED
    hook when the kit closes and cuts it: the check returns the room as
    stored, never a CancelledError its caller did not ask for."""
    kit = RoomKit()
    await kit.create_room(room_id="r1", timers=RoomTimers(inactive_after_seconds=0))
    in_hook = asyncio.Event()

    @kit.hook(HookTrigger.ON_ROOM_PAUSED, name="slow")
    async def slow(event: RoomEvent, ctx: RoomContext) -> None:
        in_hook.set()
        await asyncio.Event().wait()

    check = asyncio.create_task(kit.check_room_timers("r1"))
    await asyncio.wait_for(in_hook.wait(), 1)
    await kit.close()

    room = await asyncio.wait_for(check, 1)
    assert room.status is RoomStatus.PAUSED


async def test_a_result_post_held_just_before_the_close_posts_nothing() -> None:
    """A task's result posted as its agent's message, in work the kit holds
    (a discussion's post turn), is held just before the kit closes: cut once
    started, it posts nothing on the closing kit (RFC §19.7.5 rule 19)."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("ai1", provider=MockAIProvider(responses=["hello"])))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    result = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ai1", channel_type="ai"),
        content=TextContent(body="result"),
    )
    hold_task(kit, send_posted(kit, result))
    await kit.close()

    assert "result" not in [
        getattr(e.content, "body", None) for e in await kit.store.list_events("r1")
    ]
