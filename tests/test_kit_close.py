"""The kit's ``close()``, rule by rule (RMK-1030, RFC §12.10.4).

1. Admission stops at once: every call made once ``close()`` started raises
   ``KitClosedError``, whoever makes it.
2. Work not started is cut at once: a queued delivery, a call waiting for its
   room's lock.
3. A model turn is cut at once, its stream and its tool calls: what it
   streamed is stored marked cancelled, an open tool call gets its end row.
4. Other running work (a hook, a store write, a transport send) gets one
   grace, then is cut; nothing is written or delivered after ``close()``.
5. Every call the close cuts returns one result, ``kit_closed``, with what it
   delivered and the answers already stored; never ``lane_cancelled``.
6. Kept: a close from the kit's own work does not wait for itself, concurrent
   closes share one shutdown (``tests/test_kit_close_doors.py``,
   ``tests/test_framework.py::TestOneShutdown``).
7. Bound: the grace plus closing the channels.

The grace is 50 ms under test (the ``close_grace`` fixture).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    KitClosedError,
    RoomKit,
    TextContent,
)
from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelOutput
from roomkit.models.event import EventSource, RoomEvent
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
from tests.test_framework import AILikeChannel, SimpleChannel


class _Transport(SimpleChannel):
    """A transport whose send of a body in *holds* waits: its seconds, or for
    ever when ``None``."""

    def __init__(self, channel_id: str, holds: dict[str, float | None] | None = None) -> None:
        super().__init__(channel_id)
        self.holds = holds or {}
        self.sending: dict[str, asyncio.Event] = {}
        self.sent: list[str] = []

    def entered(self, body: str) -> asyncio.Event:
        return self.sending.setdefault(body, asyncio.Event())

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        body = getattr(event.content, "body", "")
        self.entered(body).set()
        if body in self.holds:
            hold = self.holds[body]
            await (asyncio.Event().wait() if hold is None else asyncio.sleep(hold))
        self.sent.append(body)
        return await super().deliver(event, binding, context)


async def _kit(*targets: SimpleChannel, rooms: tuple[str, ...] = ("r",), **kw: Any) -> RoomKit:
    kit = RoomKit(**kw)
    kit.register_channel(SimpleChannel("sms"))
    for target in targets:
        kit.register_channel(target)
    for room_id in rooms:
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "sms")
        for target in targets:
            await kit.attach_channel(room_id, target.channel_id)
    return kit


def _message(body: str) -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body=body))


def _record(body: str) -> RoomEvent:
    return RoomEvent(
        room_id="r",
        source=EventSource(channel_id="sms", channel_type="sms"),
        content=TextContent(body=body),
    )


async def _bodies(kit: RoomKit, room_id: str = "r") -> list[str]:
    return [getattr(e.content, "body", "") for e in await kit.store.list_events(room_id)]


async def _a_send_held_for_ever(kit: RoomKit, transport: _Transport, room_id: str) -> None:
    """Keep the close in its grace: a send that never ends."""
    await kit.process_inbound(_message("held"), room_id=room_id, defer_delivery=True)
    await transport.entered("held").wait()


class _SlowWords(MockAIProvider):
    """Streams a long answer word by word; *started* once its first words are
    read."""

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__(streaming=True)
        self.started = started

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        self.calls.append(context)
        yield StreamTextDelta(text="The ")
        await asyncio.sleep(0.01)
        self.started.set()
        for i in range(2000):
            await asyncio.sleep(0.01)
            yield StreamTextDelta(text=f"w{i} ")
        yield StreamDone(finish_reason="stop", usage={"input_tokens": 5, "output_tokens": 50})


# -- 1. Admission stops at once -------------------------------------------------

_DOORS: dict[str, Callable[[RoomKit], Awaitable[Any]]] = {
    "process_inbound": lambda kit: kit.process_inbound(_message("late"), room_id="r"),
    "send_event": lambda kit: kit.send_event("r", "sms", TextContent(body="late")),
    "commit_event": lambda kit: kit.commit_event("r", _record("late")),
    "create_room": lambda kit: kit.create_room(room_id="late"),
    "get_timeline": lambda kit: kit.get_timeline("r"),
}


@pytest.mark.parametrize("door", list(_DOORS))
@pytest.mark.parametrize("when", ["during", "after"])
async def test_a_call_once_the_close_started_is_refused(
    door: str, when: str, close_grace: float
) -> None:
    held = _Transport("ws", holds={"held": None})
    kit = await _kit(held)
    await _a_send_held_for_ever(kit, held, "r")
    closing = asyncio.ensure_future(kit.close())
    await asyncio.sleep(0)  # the close starts
    if when == "after":
        await closing

    with pytest.raises(KitClosedError):
        await _DOORS[door](kit)

    await closing
    assert "late" not in await _bodies(kit)


async def test_a_call_from_a_hook_in_the_close_s_grace_is_refused(close_grace: float) -> None:
    """A hook still running as the kit closes calls into the kit: refused,
    while the call it runs for commits within the grace."""
    kit = await _kit(_Transport("ws"))
    entered, closing = asyncio.Event(), asyncio.Event()
    refused: list[type[BaseException]] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="late")
    async def late(event: RoomEvent, ctx: Any) -> HookResult:
        if getattr(event.content, "body", "") == "in hook":
            entered.set()
            await closing.wait()
            try:
                await kit.send_event("r", "sms", TextContent(body="from the hook"))
            except Exception as error:
                refused.append(type(error))
        return HookResult.allow()

    call = asyncio.ensure_future(kit.process_inbound(_message("in hook"), room_id="r"))
    await entered.wait()
    closed = asyncio.ensure_future(kit.close())
    closing.set()
    await closed

    assert refused == [KitClosedError]
    result = await call
    assert result.cancellation_reason == "kit_closed"
    assert "from the hook" not in await _bodies(kit)


async def test_the_close_returns_while_calls_keep_coming(close_grace: float) -> None:
    """Calls arriving with no gap do not hold the close open (RMK-1016):
    they are refused, and the close ends with its grace."""
    held = _Transport("ws", holds={"held": None})
    kit = await _kit(held)
    await _a_send_held_for_ever(kit, held, "r")
    stop = False
    refused: list[type[BaseException]] = []

    async def traffic() -> None:
        while not stop:
            try:
                await kit.process_inbound(_message("traffic"), room_id="r", defer_delivery=True)
            except KitClosedError as error:
                refused.append(type(error))
            await asyncio.sleep(0)

    producers = [asyncio.ensure_future(traffic()) for _ in range(4)]
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(kit.close(), 1)
    finally:
        stop = True
        await asyncio.gather(*producers)

    assert refused


# -- 2. Work not started is cut at once -----------------------------------------


async def test_a_delivery_queued_as_the_kit_closes_is_never_sent(close_grace: float) -> None:
    transport = _Transport("ws", holds={"held": close_grace / 5})
    kit = await _kit(transport)
    held = await kit.process_inbound(_message("held"), room_id="r", defer_delivery=True)
    await transport.entered("held").wait()
    queued = await kit.process_inbound(_message("queued"), room_id="r", defer_delivery=True)
    assert held.delivery is not None and queued.delivery is not None

    await kit.close()

    dropped = await queued.delivery.wait()
    assert (dropped.cancellation_reason, dropped.delivery_results) == ("kit_closed", {})
    assert transport.sent == ["held"]


@pytest.mark.parametrize("door", ["process_inbound", "send_event"])
async def test_a_call_waiting_for_its_room_lock_is_cut_at_once(
    door: str, close_grace: float
) -> None:
    kit = await _kit(_Transport("ws"))
    in_hook = asyncio.Event()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="holds-the-lock")
    async def holds(event: RoomEvent, ctx: Any) -> HookResult:
        if getattr(event.content, "body", "") == "locker":
            in_hook.set()
            await asyncio.Event().wait()
        return HookResult.allow()

    locker = asyncio.ensure_future(kit.process_inbound(_message("locker"), room_id="r"))
    await in_hook.wait()
    if door == "process_inbound":
        waiter: asyncio.Future[Any] = asyncio.ensure_future(
            kit.process_inbound(_message("waiter"), room_id="r")
        )
    else:
        waiter = asyncio.ensure_future(kit.send_event("r", "sms", TextContent(body="waiter")))
    for _ in range(10):
        await asyncio.sleep(0)
    assert not waiter.done()

    closing = asyncio.ensure_future(kit.close())
    await asyncio.wait({waiter})
    assert not closing.done()  # the locker's hook still has its grace

    if door == "process_inbound":
        result = waiter.result()
        assert (result.blocked, result.reason, result.cancellation_reason) == (
            True,
            "kit_closed",
            "kit_closed",
        )
    else:
        assert isinstance(waiter.exception(), KitClosedError)
    await closing
    cut = await locker
    assert (cut.blocked, cut.cancellation_reason) == (True, "kit_closed")
    assert [b for b in await _bodies(kit) if b in ("locker", "waiter")] == []


# -- 3. A model turn is cut at once ----------------------------------------------


async def test_a_streaming_turn_is_cut_at_once_its_text_stored_cancelled(
    close_grace: float,
) -> None:
    held = _Transport("ws", holds={"held": None})
    started = asyncio.Event()
    kit = await _kit(held, rooms=("r", "busy"))
    kit.register_channel(Agent("agent", provider=_SlowWords(started)))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("Go."), room_id="r"))
    await started.wait()
    await _a_send_held_for_ever(kit, held, "busy")

    closing = asyncio.ensure_future(kit.close())
    result = await call
    assert not closing.done()  # the send in the other room still has its grace
    await closing

    assert result.cancellation_reason == "kit_closed"
    assert dict(result.response_metadata)["turns"]["agent"]["loop_end_reason"] == "cancelled"
    answers = [e for e in await kit.store.list_events("r") if e.source.channel_id == "agent"]
    assert [getattr(e.content, "body", "") for e in answers] == ["The "]
    assert answers[0].metadata.get("cancelled") is True


async def test_a_running_tool_call_is_cut_at_once_its_end_row_stored(close_grace: float) -> None:
    held = _Transport("ws", holds={"held": None})
    entered = asyncio.Event()
    ran: list[str] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            ran.append("cut")
            raise
        return "found"

    tool = {"name": "lookup", "description": "Look up.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
            ),
            AIResponse(content="Found it.", finish_reason="stop"),
        ]
    )
    kit = await _kit(held, rooms=("r", "busy"))
    kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=lookup))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("Go."), room_id="r"))
    await entered.wait()
    await _a_send_held_for_ever(kit, held, "busy")

    closing = asyncio.ensure_future(kit.close())
    result = await call
    assert not closing.done()
    await closing

    assert ran == ["cut"]
    assert result.cancellation_reason == "kit_closed"
    ends = [e for e in await kit.store.list_events("r") if e.type == "tool_call_end"]
    assert [e.content.status for e in ends] == ["failed"]


# -- 4. Other running work gets one grace, then is cut ----------------------------


async def test_a_send_ending_within_the_grace_is_delivered(close_grace: float) -> None:
    transport = _Transport("ws", holds={"send": close_grace / 5})
    kit = await _kit(transport)
    call = await kit.process_inbound(_message("send"), room_id="r", defer_delivery=True)
    await transport.entered("send").wait()

    await kit.close()

    assert call.delivery is not None
    result = await call.delivery.wait()
    assert result.delivery_results["ws"].status == "sent"
    assert result.cancellation_reason == "kit_closed"


async def test_a_send_outlasting_the_grace_is_cut_and_reported_failed(close_grace: float) -> None:
    quick, held = _Transport("a"), _Transport("b", holds={"both": None})
    kit = await _kit(quick, held)
    call = await kit.process_inbound(_message("both"), room_id="r", defer_delivery=True)
    await held.entered("both").wait()

    await kit.close()

    assert call.delivery is not None
    result = await call.delivery.wait()
    assert result.cancellation_reason == "kit_closed"
    assert result.delivery_results["a"].status == "sent"
    cut = result.delivery_results["b"]
    assert cut.status == "failed" and cut.error is not None
    assert (cut.error.code, cut.error.retryable) == ("kit_closed", True)
    assert held.sent == []


async def test_a_hook_outlasting_the_grace_is_cut(close_grace: float) -> None:
    kit = await _kit(_Transport("ws"))
    started = asyncio.Event()
    steps: list[str] = []

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: RoomEvent, ctx: Any) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("cut")
            raise

    call = asyncio.ensure_future(kit.process_inbound(_message("hi"), room_id="r"))
    await started.wait()
    await kit.close()

    assert steps == ["cut"]
    assert (await call).cancellation_reason == "kit_closed"


class _NeverWrites(InMemoryStore):
    """A store whose commit of ``slow`` never ends, and says if it did."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.landed = False

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        if getattr(event.content, "body", None) == "slow":
            self.entered.set()
            await asyncio.Event().wait()
            self.landed = True
        return await super().commit_event(room_id, event, **kw)


async def test_a_write_outlasting_the_grace_is_cut_and_never_lands(close_grace: float) -> None:
    """A commit past its caller's decision (the caller cancelled) gets the
    grace, then is cut: nothing of it is written after close()."""
    store = _NeverWrites()
    kit = await _kit(_Transport("ws"), store=store)
    call = asyncio.ensure_future(kit.process_inbound(_message("slow"), room_id="r"))
    await store.entered.wait()
    call.cancel()

    await kit.close()

    await asyncio.gather(call, return_exceptions=True)
    assert not store.landed
    assert "slow" not in await _bodies(kit)


# -- 5. One result per cut call, kit_closed, with what it delivered ---------------


class _Batch(_Transport):
    """``m0`` is sent at once, ``m1``'s send never ends."""

    def __init__(self) -> None:
        super().__init__("ws", holds={"m1": None})


async def test_each_call_the_close_cuts_reads_kit_closed_with_what_it_delivered(
    close_grace: float,
) -> None:
    transport = _Batch()
    kit = await _kit(transport)
    handles = []
    for n in range(3):
        result = await kit.process_inbound(_message(f"m{n}"), room_id="r", defer_delivery=True)
        assert result.delivery is not None
        handles.append(result.delivery)
    await transport.entered("m1").wait()

    await kit.close()

    results = [await handle.wait() for handle in handles]
    assert [r.cancellation_reason for r in results] == ["kit_closed"] * 3
    assert [{k: v.status for k, v in r.delivery_results.items()} for r in results] == [
        {"ws": "sent"},
        {"ws": "failed"},
        {},
    ]


class _StuckOnAnswers(AILikeChannel):
    """An agent whose turn on another agent's answer never ends."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.started = asyncio.Event()

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        if event.source.channel_id == "ai":
            self.started.set()
            await asyncio.Event().wait()
        return ChannelOutput.empty()


async def test_a_cut_call_carries_the_answers_already_stored(close_grace: float) -> None:
    kit = await _kit(_Transport("ws"))
    stuck = _StuckOnAnswers("stuck")
    kit.register_channel(AILikeChannel("ai", response="hello"))
    kit.register_channel(stuck)
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r", "stuck", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("hi"), room_id="r"))
    await stuck.started.wait()

    await kit.close()

    result = await call
    assert result.cancellation_reason == "kit_closed"
    assert [getattr(e.content, "body", None) for e in result.response_events] == ["hello"]


# -- 7. Bound ------------------------------------------------------------------


@pytest.mark.wallclock
async def test_the_close_takes_its_grace_with_every_kind_of_work_stuck(
    close_grace: float,
) -> None:
    """A send, a decided write, an observer and a hook holding its room's
    lock, none ever ending: the close cuts them all once its grace ran out."""
    held = _Transport("ws", holds={"held": None})
    store = _NeverWrites()
    rooms = ("send", "write", "observer", "lock")
    kit = await _kit(held, rooms=rooms, store=store)
    observing, locking = asyncio.Event(), asyncio.Event()

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="observer")
    async def observer(event: RoomEvent, ctx: Any) -> None:
        if getattr(event.content, "body", "") == "observed":
            observing.set()
            await asyncio.Event().wait()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="locker")
    async def locker(event: RoomEvent, ctx: Any) -> HookResult:
        if getattr(event.content, "body", "") == "locked":
            locking.set()
            await asyncio.Event().wait()
        return HookResult.allow()

    await _a_send_held_for_ever(kit, held, "send")
    calls = [
        asyncio.ensure_future(kit.process_inbound(_message("observed"), room_id="observer")),
        asyncio.ensure_future(kit.process_inbound(_message("locked"), room_id="lock")),
    ]
    write = asyncio.ensure_future(kit.process_inbound(_message("slow"), room_id="write"))
    await asyncio.gather(observing.wait(), locking.wait(), store.entered.wait())
    write.cancel()
    loop = asyncio.get_running_loop()
    started = loop.time()

    await asyncio.wait_for(kit.close(), 1)

    assert loop.time() - started < close_grace + 0.5
    await asyncio.gather(*calls, write, return_exceptions=True)
