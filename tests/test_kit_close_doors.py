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
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from roomkit import (
    ChannelBinding,
    ChannelType,
    HookExecution,
    HookTrigger,
    KitClosedError,
    RealtimeVoiceChannel,
    RoomKit,
    RoomStatus,
    SpeakQueueChange,
    SpeakQueueEvent,
    VideoChannel,
    VoiceChannel,
)
from roomkit.channels.agent import Agent
from roomkit.core.task_utils import hold_task, run_to_its_end
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
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.stt.mock import MockSTTProvider
from tests.orchestration.test_strategy_discussion import _provider, _say
from tests.orchestration.test_strategy_discussion import _room as _discussion
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


# -- a close from a delivery, a session's end, an announcement -----------------


class _Witness(SimpleChannel):
    """A channel that says whether it was closed."""

    closed = False

    async def close(self) -> None:
        self.closed = True


class _Closer:
    """The kit's own work closing the kit, once let go."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.go = asyncio.Event()
        self.closed = asyncio.Event()
        self.door: asyncio.Future[Any] | None = None
        """What opened the door, when it waits for the work: awaited last."""

    async def __call__(self, kit: RoomKit) -> None:
        if self.entered.is_set():
            return
        self.entered.set()
        await self.go.wait()
        await kit.close()
        self.closed.set()


async def _delivery_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A delivery's ASYNC AFTER_BROADCAST hook closes the kit: it runs in the
    room's delivery lane."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", witness.channel_id)

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="quit")
    async def quit(event: RoomEvent, ctx: RoomContext) -> None:
        await work(kit)

    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Bye."))
    # Its delivery waits for the hook, which may be held.
    work.door = asyncio.ensure_future(kit.process_inbound(message, room_id="r"))
    entered = asyncio.ensure_future(work.entered.wait())
    await asyncio.wait({work.door, entered}, return_when=asyncio.FIRST_COMPLETED)
    entered.cancel()
    return kit


async def _realtime_session_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """The client hangs up a realtime session; the ``voice_session_ended``
    handler, run in the session's teardown, closes the kit."""
    kit = RoomKit()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=MockRealtimeProvider(), transport=transport)
    kit.register_channel(channel)
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    session = await channel.start_session("r", "p", "fake-ws")

    @kit.on("voice_session_ended")
    async def ended(event: Any) -> None:
        await work(kit)

    await transport.simulate_client_disconnect(session)
    return kit


async def _voice_session_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """The client hangs up a voice session; the ``voice_session_ended``
    handler, run in one of the voice channel's tasks, closes the kit."""
    backend = MockVoiceBackend()
    kit = RoomKit(voice=backend)
    kit.register_channel(VoiceChannel("voice", stt=MockSTTProvider(), backend=backend))
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "voice")
    session = await kit.join("r", "voice", participant_id="u")

    @kit.on("voice_session_ended")
    async def ended(event: Any) -> None:
        await work(kit)

    await backend.simulate_client_disconnected(session)
    return kit


async def _video_session_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A video session is unbound; the ``video_session_ended`` handler, run
    in one of the video channel's tasks, closes the kit."""
    backend = MockVideoBackend()
    kit = RoomKit()
    video = VideoChannel("video", backend=backend)
    kit.register_channel(video)
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "video")
    session = await backend.connect("r", "u", "video")
    binding = ChannelBinding(room_id="r", channel_id="video", channel_type=ChannelType.VIDEO)
    video.bind_session(session, "r", binding)

    @kit.on("video_session_ended")
    async def ended(event: Any) -> None:
        await work(kit)

    video.unbind_session(session)
    return kit


async def _delegation_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A background delegation ends; its ``on_complete``, run in the task's
    end, closes the kit."""
    kit = RoomKit()
    kit.register_channel(witness)
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["Done."])))
    await kit.create_room(room_id="r")

    async def on_complete(result: Any) -> None:
        await work(kit)

    await kit.delegate("r", "worker", "Go.", on_complete=on_complete)
    return kit


async def _held_write_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A decided write a background run waits for (a strategy's or a
    delegation's, which its caller never cuts) closes the kit, from a hook
    it runs."""
    kit = RoomKit()
    kit.register_channel(witness)

    async def run() -> None:
        await run_to_its_end(work(kit), held=kit._decided_writes, name="write")

    hold_task(kit, run())
    return kit


async def _speak_queue_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A discussion's turn ends; an ASYNC ON_SPEAK_QUEUE hook, run under the
    announcement of that change, closes the kit."""
    kit = RoomKit()
    kit.register_channel(witness)

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def quit(event: SpeakQueueEvent, ctx: RoomContext) -> None:
        if event.change == SpeakQueueChange.TURN_ENDED:
            await work(kit)

    await _discussion(kit, {"a": _provider("Done.")})
    await _say(kit, "@a go")
    return kit


@pytest.mark.parametrize(
    "door",
    [
        _delivery_door,
        _realtime_session_end_door,
        _voice_session_end_door,
        _video_session_end_door,
        _speak_queue_door,
        _delegation_end_door,
        _held_write_door,
    ],
    ids=[
        "delivery-hook",
        "realtime-session-end",
        "voice-session-end",
        "video-session-end",
        "speak-queue-hook",
        "delegation-end",
        "held-write",
    ],
)
async def test_a_close_from_the_kit_s_own_work_returns_and_closes_the_kit(
    door: Callable[[_Witness, _Closer], Awaitable[RoomKit]],
) -> None:
    """The kit's own work closes the kit: its close neither waits for nor
    cuts the work it runs in, and returns once every channel is closed
    (RMK-1015)."""
    witness = _Witness("witness")
    work = _Closer()
    work.go.set()

    # A realtime channel's attach compiles its marks' copies, ~1 s on its own.
    kit = await asyncio.wait_for(door(witness, work), 5)
    await asyncio.wait_for(work.closed.wait(), 1)

    assert witness.closed
    await kit.close()
    if work.door is not None:
        await asyncio.wait_for(work.door, 1)


@pytest.mark.parametrize(
    "door",
    [
        _delivery_door,
        _realtime_session_end_door,
        _speak_queue_door,
        _delegation_end_door,
        _held_write_door,
    ],
    ids=[
        "delivery-hook",
        "realtime-session-end",
        "speak-queue-hook",
        "delegation-end",
        "held-write",
    ],
)
async def test_a_close_from_the_kit_s_own_work_joining_another_spares_that_work(
    door: Callable[[_Witness, _Closer], Awaitable[RoomKit]],
) -> None:
    """Another close() already runs, and waits for the work (the lane's
    drain, the session's teardown) when that work closes the kit too: from
    then on it is spared, and both calls return once the kit is closed
    (RMK-1015)."""
    witness = _Witness("witness")
    work = _Closer()
    kit = await asyncio.wait_for(door(witness, work), 5)
    await asyncio.wait_for(work.entered.wait(), 1)

    first = asyncio.ensure_future(kit.close())
    # The other close's shutdown reaches the work, and waits for it, first.
    for _ in range(5):
        await asyncio.sleep(0)
    work.go.set()

    await asyncio.wait_for(work.closed.wait(), 1)
    await asyncio.wait_for(first, 1)
    assert witness.closed
    if work.door is not None:
        await asyncio.wait_for(work.door, 1)


async def test_two_sessions_ending_at_once_each_close_the_kit() -> None:
    """Two realtime sessions' clients hang up together, and each one's
    ``voice_session_ended`` handler closes the kit: the second joins the
    shutdown the first started, which spares both (RMK-1015)."""
    kit = RoomKit()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=MockRealtimeProvider(), transport=transport)
    witness = _Witness("witness")
    kit.register_channel(channel)
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    sessions = [await channel.start_session("r", f"p{n}", f"ws-{n}") for n in (1, 2)]
    returned: list[str] = []
    both = asyncio.Event()

    @kit.on("voice_session_ended")
    async def ended(event: Any) -> None:
        await kit.close()
        returned.append(event.room_id)
        if len(returned) == 2:
            both.set()

    await asyncio.wait_for(
        asyncio.gather(*(transport.simulate_client_disconnect(s) for s in sessions)), 1
    )
    await asyncio.wait_for(both.wait(), 1)
    assert witness.closed


async def test_the_deliveries_queued_behind_a_lane_a_close_runs_in_are_dropped() -> None:
    """A delivery's hook closes the kit while the room's next message waits
    behind it in the lane: the lane is spared, not drained, so that message
    is never delivered, and its caller is not held (RMK-1015)."""
    witness = _Witness("witness")
    work = _Closer()
    kit = await asyncio.wait_for(_delivery_door(witness, work), 5)
    await asyncio.wait_for(work.entered.wait(), 1)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Again."))
    queued = asyncio.ensure_future(kit.process_inbound(message, room_id="r"))
    # Queued behind the delivery whose hook runs.
    for _ in range(20):
        await asyncio.sleep(0)
    work.go.set()

    await asyncio.wait_for(work.closed.wait(), 1)
    await asyncio.wait_for(asyncio.gather(*(f for f in (work.door, queued) if f)), 1)
    assert [event.content.body for event in witness.delivered] == ["Bye."]


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


async def test_a_call_an_observer_makes_as_the_kit_closes_is_refused(close_grace: float) -> None:
    """An observer still running as the kit closes commits into a room whose
    timer fell due: the call is refused (RMK-1030), no timer hook starts,
    and close() returns with nothing of the kit left running."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.create_room(room_id="r2", timers=RoomTimers(inactive_after_seconds=0))
    closing = asyncio.Event()
    refused: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: RoomEvent, ctx: RoomContext) -> None:
        await closing.wait()
        record = RoomEvent(
            room_id="r2",
            source=EventSource(channel_id="sms", channel_type="sms"),
            content=TextContent(body="audit record"),
        )
        try:
            await kit.commit_event("r2", record)
        except KitClosedError:
            refused.append("commit_event")

    @kit.hook(HookTrigger.ON_ROOM_PAUSED, name="paused")
    async def paused(event: RoomEvent, ctx: RoomContext) -> None:
        await asyncio.Event().wait()

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hi")),
        room_id="r1",
    )
    closed = asyncio.ensure_future(kit.close())
    closing.set()
    await closed

    assert refused == ["commit_event"]
    running = [t.get_name() for t in asyncio.all_tasks() if not t.done()]
    assert [name for name in running if name.startswith("roomkit-")] == []


async def test_a_background_run_started_in_the_close_s_grace_ends_with_its_close() -> None:
    """An observer still running as the kit closes starts work the kit holds
    (a hand-off greeting's turn) in the close's grace: it is cut with the
    kit's other background work, never left running past close()."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    closing = asyncio.Event()
    runs: list[asyncio.Task[None]] = []

    async def greeting() -> None:
        await asyncio.Event().wait()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="greet")
    async def greet(event: RoomEvent, ctx: RoomContext) -> None:
        await closing.wait()
        runs.append(hold_task(kit, greeting()))

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hi")),
        room_id="r1",
    )
    closed = asyncio.ensure_future(kit.close())
    closing.set()
    await closed

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
    refused: BaseException | None = None

    async def close(self) -> None:
        assert self.kit is not None
        try:
            await self.kit.send_event("r1", self.channel_id, TextContent(body="bye"))
        except KitClosedError as refusal:
            self.refused = refusal
        await super().close()


async def test_a_call_a_channel_s_close_makes_is_refused() -> None:
    """A channel's close sends a last message into its room: the call is
    refused, as every call once close() started (RMK-1030), and nothing is
    stored."""
    kit = RoomKit()
    channel = _SaysByeAsItCloses("sms")
    channel.kit = kit
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")

    await kit.close()

    assert isinstance(channel.refused, KitClosedError)
    assert "bye" not in [
        getattr(e.content, "body", None) for e in await kit.store.list_events("r1")
    ]


async def test_a_caller_waiting_for_its_injected_events_returns_when_the_close_cuts_them(
    close_grace: float,
) -> None:
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


async def test_a_timer_check_waiting_for_its_hooks_returns_when_the_close_cuts_them(
    close_grace: float,
) -> None:
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
