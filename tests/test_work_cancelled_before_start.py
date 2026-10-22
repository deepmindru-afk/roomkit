"""Work handed to a task of its own, the task cancelled before its first step
(a close or a teardown right after the work was scheduled): nothing of the
work is left unawaited, a RuntimeWarning at an unrelated moment that fails a
host's tests under ``-W error``, and what the work held is let go (RMK-909).

The owner calls' own case is ``tests/test_turns_calls.py`` (RMK-901)."""

from __future__ import annotations

import asyncio
import gc
import time
import warnings
from typing import Any, cast

from roomkit.channels import VoiceChannel
from roomkit.channels._conference_activity import RoomActivity
from roomkit.channels._conference_realtime import _RUNNING_REPORT, ConferenceRealtime
from roomkit.models.delivery import InboundResult
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.bridge import RealtimeAVBridge
from roomkit.voice.realtime.mock import MockRealtimeProvider


async def _cancelled_before_start(tasks: list[asyncio.Task[Any]]) -> list[str]:
    """Cancel *tasks* before their first step and free them: the coroutines
    collected unawaited meanwhile."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # The tasks freed, the loop's name included, what they never awaited
        # is collected here, inside.
        tasks.clear()
        del task
        await asyncio.sleep(0)
        gc.collect()
    return [str(w.message) for w in caught if "never awaited" in str(w.message)]


async def _published() -> None:
    """A room's piece of work: a chunk published on the backend."""


async def test_a_room_s_work_cancelled_before_it_starts_is_let_go() -> None:
    """The room holds the work from before its task starts; cancelled before
    its first step, the work is closed and the room holds it no more, so a
    teardown's drain returns at once instead of waiting out its budget."""
    activity = RoomActivity()

    unawaited = await _cancelled_before_start([activity.spawn("r1", _published())])
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

    assert await _cancelled_before_start(list(realtime._reports)) == []


async def test_a_report_is_named_to_the_code_it_runs() -> None:
    """What runs under a report reads the report as its own, so a hook that
    disconnects from inside it never waits for itself."""
    realtime = _realtime()
    named: list[bool] = []

    async def report() -> None:
        named.append(_RUNNING_REPORT.get() is asyncio.current_task())

    realtime._track_report(report())
    await asyncio.gather(*realtime._reports)

    assert named == [True]
    assert _RUNNING_REPORT.get() is None


class _Draining:
    """A routed turn still draining as the user's speech supersedes it."""

    async def cancel(self, *, reason: str) -> InboundResult:
        raise TimeoutError

    async def wait(self) -> InboundResult:  # pragma: no cover - never reached
        return InboundResult(blocked=False)


class _SupersededTurn:
    def take_superseded(self, session_id: str, min_speech_ms: float) -> _Draining:
        return _Draining()


async def test_a_superseded_turn_s_marking_cancelled_before_it_starts_leaves_nothing() -> None:
    """The turn still draining, its marking waits for it on a task of its own,
    cancelled as the channel closes."""
    channel = VoiceChannel("voice", backend=MockVoiceBackend())
    channel._unheard_turns = cast(Any, _SupersededTurn())
    session = VoiceSession(id="s1", room_id="r1", participant_id="p", channel_id="voice")

    assert await channel._supersede_unheard_turn(session)
    marking = [t for t in channel._scheduled_tasks if t.get_name() == "superseded:s1"]

    assert len(marking) == 1
    assert await _cancelled_before_start(marking) == []


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
    assert await _cancelled_before_start(callbacks) == []
