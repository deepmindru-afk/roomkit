"""Work handed to a task of its own, the task cancelled before its first step
(a close or a teardown right after the work was scheduled): nothing of the
work is left unawaited, a RuntimeWarning at an unrelated moment that fails a
host's tests under ``-W error``, and what the work held is let go (RMK-909).

The owner calls' own case is ``tests/test_turns_calls.py`` (RMK-901)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.channels import VoiceChannel
from roomkit.channels._conference_activity import RoomActivity
from roomkit.channels._conference_realtime import ConferenceRealtime
from roomkit.channels.base import _safe_invoke
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.task_utils import held_runs
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.providers.anam.avatar import AnamAvatarProvider
from roomkit.turns import TurnExecutor
from roomkit.turns.base import ExecutorLink
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import (
    AudioPipeline,
    AudioPipelineConfig,
    MockAudioRecorder,
    MockVADProvider,
    RecordingConfig,
)
from roomkit.voice.realtime.bridge import RealtimeAVBridge
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_framework import SimpleChannel
from tests.test_turns_calls import _request
from tests.unawaited import cancelled_before_start, never_awaited
from tests.voice.realtime.test_realtime_bridge import _bridge_with_a_call


async def _published() -> None:
    """A room's piece of work: a chunk published on the backend."""


# -- a conference room's work --------------------------------------------------


async def test_a_room_s_work_cancelled_before_it_starts_is_let_go() -> None:
    """The room holds the work from before its task starts; cancelled before
    its first step, the work is closed and the room holds it no more, so a
    teardown's drain returns at once instead of waiting out its budget."""
    activity = RoomActivity()

    unawaited = await cancelled_before_start([activity.spawn("r1", _published())])
    started = time.monotonic()
    await activity.drain("r1", timeout=2.0)

    assert unawaited == []
    assert time.monotonic() - started < 0.5


async def test_a_room_s_work_that_ran_is_let_go_once_done() -> None:
    activity = RoomActivity()

    await activity.spawn("r1", _published())
    started = time.monotonic()
    await activity.drain("r1", timeout=2.0)

    assert time.monotonic() - started < 0.5


# -- a conference realtime report ---------------------------------------------


def _realtime() -> ConferenceRealtime:
    # Reports read none of these.
    nothing = cast(Any, None)
    return ConferenceRealtime(
        channel_id="conference",
        bot_identity="bot",
        voice=nothing,
        operations=nothing,
        ensure_bot=nothing,
        binding_of=nothing,
    )


async def test_a_report_cancelled_before_it_starts_leaves_nothing_unawaited() -> None:
    realtime = _realtime()

    realtime._track_report(_published())

    assert await cancelled_before_start(list(realtime._reports)) == []


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager-task-factory"])
async def test_a_report_is_named_to_the_code_it_runs_from_its_first_step(eager: bool) -> None:
    """What runs under a report reads the report as its own, from its first
    step whatever the loop's task factory, so a hook that disconnects from
    inside it never waits for itself."""
    loop = asyncio.get_running_loop()
    realtime = _realtime()
    named: list[bool] = []

    async def report() -> None:
        named.append(asyncio.current_task() in held_runs())
        await asyncio.sleep(0)
        named.append(asyncio.current_task() in held_runs())

    if eager:
        loop.set_task_factory(asyncio.eager_task_factory)
    try:
        realtime._track_report(report())
    finally:
        loop.set_task_factory(None)
    await asyncio.gather(*realtime._reports)

    assert named == [True, True]
    assert held_runs() == ()


# -- a superseded voice turn ----------------------------------------------------


def _superseded(handle: Any) -> VoiceChannel:
    """A voice channel whose user's ended speech supersedes *handle*'s turn."""

    class _OneTurn:
        def take_superseded(self, session_id: str, min_speech_ms: float) -> Any:
            return handle

    channel = VoiceChannel("voice", backend=MockVoiceBackend())
    channel._unheard_turns = cast(Any, _OneTurn())
    return channel


class _Draining:
    """A routed turn still draining as the user's speech supersedes it."""

    def __init__(self, result: InboundResult | None = None) -> None:
        self.result = result
        self.drained = asyncio.Event()

    async def cancel(self, *, reason: str) -> InboundResult:
        raise TimeoutError

    async def wait(self) -> InboundResult:
        await self.drained.wait()
        return self.result or InboundResult(blocked=False)


_SESSION = VoiceSession(id="s1", room_id="r1", participant_id="p", channel_id="voice")


async def test_a_superseded_turn_s_marking_cancelled_before_it_starts_leaves_nothing() -> None:
    """The turn still draining, its marking waits for it on a task of its own,
    cancelled as the channel closes."""
    channel = _superseded(_Draining())

    assert await channel._supersede_unheard_turn(_SESSION)
    marking = [t for t in channel._scheduled_tasks if t.get_name() == "superseded:s1"]

    assert len(marking) == 1
    assert await cancelled_before_start(marking) == []


async def test_a_superseded_turn_still_draining_is_marked_once_drained() -> None:
    event = RoomEvent(
        room_id="r1",
        type=EventType.MESSAGE,
        source=EventSource(channel_id="ai", channel_type=ChannelType.AI),
        content=TextContent(body="never heard"),
    )
    draining = _Draining(InboundResult(blocked=False, response_events=[event]))
    marked: list[tuple[str, str | None]] = []

    class _Kit:
        async def update_event(
            self, room_id: str, event_id: str, *, metadata: dict[str, Any]
        ) -> None:
            marked.append((event_id, metadata.get("cancellation_reason")))

    channel = _superseded(draining)
    channel._framework = cast(Any, _Kit())

    assert await channel._supersede_unheard_turn(_SESSION)
    draining.drained.set()
    for _ in range(5):
        await asyncio.sleep(0)

    assert marked == [(event.id, "superseded")]


# -- a realtime bridge's callbacks ---------------------------------------------


async def test_a_detection_callback_s_coroutine_cancelled_before_it_starts_leaves_nothing() -> (
    None
):
    bridge = RealtimeAVBridge(MockRealtimeProvider(), MockVoiceBackend())

    async def on_detection(event: Any) -> None:
        pass

    before = asyncio.all_tasks()
    bridge._notify_detection(on_detection, cast(Any, None))
    callbacks = list(asyncio.all_tasks() - before)

    assert len(callbacks) == 1
    assert await cancelled_before_start(callbacks) == []


async def test_a_call_end_callback_returning_a_future_is_awaited() -> None:
    """``on_call_ended`` may return any awaitable, as ``on_video_detection``
    may: a future is awaited on a task of its own, not refused."""
    ended = asyncio.get_running_loop().create_future()
    bridge, _, session = await _bridge_with_a_call(frames=0, on_call_ended=lambda *_: ended)

    bridge._on_backend_disconnect(session)
    ended.set_result(None)
    await asyncio.sleep(0)
    await bridge.close()


# -- an executor's turn --------------------------------------------------------


class _Link(ExecutorLink):
    """A turn's link to its owner; one that cannot carry a frame when *gone*."""

    def __init__(self, *, gone: bool = False) -> None:
        self.request = _request()
        self.gone = gone

    async def receive(self) -> Any:
        await asyncio.Event().wait()

    async def send(self, frame: Any) -> int:
        if self.gone:
            raise ConnectionError("the link is gone")
        return 1


@pytest.mark.parametrize("gone", [False, True], ids=["closed-at-once", "accepted-not-carried"])
async def test_a_turn_its_executor_never_ran_leaves_its_channel_unbuilt(gone: bool) -> None:
    """The turn's channel is built by the turn's task, never before: a close
    as the turn arrives, or an ``Accepted`` its link cannot carry, leaves no
    channel build unawaited."""
    executor = TurnExecutor(lambda channel_id: SimpleChannel(channel_id))
    loop = asyncio.get_running_loop()
    handler = loop.get_exception_handler()
    # The refused Accepted fails the turn's task: a handler that keeps what it
    # reports would keep the turn, its channel's build included, past the block.
    loop.set_exception_handler(lambda _loop, _context: None)
    try:
        with never_awaited() as unawaited:
            executor.take(_Link(gone=gone))
            if gone:
                # The turn sends its Accepted, which the link refuses.
                for _ in range(20):
                    await asyncio.sleep(0.01)
            await executor.close()
            del executor
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(handler)

    assert unawaited == []


# -- a recording's announcement -------------------------------------------------


async def _announcement(session: VoiceSession, handle: object) -> None:
    """An integrator's consent announcement."""


@pytest.mark.parametrize("at", ["before-its-first-step", "between-two-subscribers"])
async def test_a_recording_announcement_cut_short_leaves_no_subscriber_unawaited(at: str) -> None:
    """The announcement's subscribers are heard one after the other on a task
    of its own; cut before it starts, or while the first is still heard,
    the subscribers it never reached are closed."""
    first_heard = asyncio.Event()

    async def slow(session: VoiceSession, handle: object) -> None:
        first_heard.set()
        await asyncio.sleep(10)

    pipeline = AudioPipeline(
        AudioPipelineConfig(
            vad=MockVADProvider(), recorder=MockAudioRecorder(), recording_config=RecordingConfig()
        )
    )
    pipeline.on_recording_started(slow)
    pipeline.on_recording_started(_announcement)

    with never_awaited() as unawaited:
        before = asyncio.all_tasks()
        pipeline.on_session_active(_SESSION)
        announcing = list(asyncio.all_tasks() - before)
        if at == "between-two-subscribers":
            await asyncio.wait_for(first_heard.wait(), 5)
        for task in announcing:
            task.cancel()
        await asyncio.gather(*announcing, return_exceptions=True)
        del announcing, task
        await asyncio.sleep(0)

    assert unawaited == []


# -- a realtime session's start --------------------------------------------------


async def _answered() -> str:
    """A SIP call's answer, which never comes."""
    await asyncio.Event().wait()
    return "carrier"


async def _realtime_kit() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(
        RealtimeVoiceChannel(
            "rt", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )
    )
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "rt")
    return kit


# Cut at once, the caller's own task never entered ``join``: no code of the
# kit's ran, and the coroutine is the caller's to close.
@pytest.mark.parametrize("ticks", [1, 2, 3, 5])
async def test_a_session_start_cut_before_its_connection_is_awaited_closes_it(ticks: int) -> None:
    """A caller hands the start its connection as a coroutine (a SIP call
    still ringing) and cancels the start before the transport takes it: the
    caller's coroutine is closed, not left never awaited."""
    kit = await _realtime_kit()

    with never_awaited() as unawaited:
        start = asyncio.ensure_future(kit.join("room", "rt", connection=_answered()))
        for _ in range(ticks):
            await asyncio.sleep(0)
        start.cancel()
        await asyncio.gather(start, return_exceptions=True)
        del start
        await asyncio.sleep(0)
    await kit.close()

    assert unawaited == []


async def test_a_session_start_refused_closes_its_connection() -> None:
    kit = await _realtime_kit()
    channel = cast(RealtimeVoiceChannel, kit.channels["rt"])

    with never_awaited() as unawaited, pytest.raises(RoomNotFoundError):
        await channel.start_session("elsewhere", "p", _answered(), organization_id="org")
    await kit.close()

    assert unawaited == []


# -- work returned where no loop runs ------------------------------------------


async def _sent(*_: Any) -> None:
    """A host's callback work, or an avatar's send."""


def test_a_trace_callback_s_coroutine_returned_where_no_loop_runs_is_closed() -> None:
    with never_awaited() as unawaited:
        _safe_invoke(_sent, cast(Any, None))

    assert unawaited == []


def test_an_avatar_s_send_scheduled_where_no_loop_runs_is_closed() -> None:
    with never_awaited() as unawaited:
        AnamAvatarProvider._schedule_async(_sent())

    assert unawaited == []
