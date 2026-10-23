"""One playback is interrupted at most once, however slow the barge-in is.

Until ``interrupt()`` pops the playback, every trigger path still sees the bot
talking. Building the barge-in's context waits on the store, and the user keeps
talking in the meantime: without a claim on the playback, the energy check
re-fires every 100 ms of continued speech and each firing runs ON_BARGE_IN and
``interrupt()`` again.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from roomkit.channels.voice import TTSPlaybackState
from tests.test_interruption_semantic_wait import _LOUD, _room, _SlowPartialSTT


def _held_store(kit: Any) -> tuple[list[str], asyncio.Event]:
    """Make every context build wait until the returned event is set, as a
    loaded store does; each build asked for while it is held is noted."""
    build = kit._build_context  # noqa: SLF001
    building: list[str] = []
    release = asyncio.Event()

    async def held(room_id: str, *args: Any, **kwargs: Any) -> Any:
        if not release.is_set():
            building.append(room_id)
            await asyncio.wait_for(release.wait(), 5)
        return await build(room_id, *args, **kwargs)

    kit._build_context = held  # noqa: SLF001
    return building, release


def _count_interrupts(channel: Any) -> list[str]:
    calls: list[str] = []
    interrupt = channel.interrupt

    async def counting(session: Any, *, reason: str = "explicit") -> bool:
        calls.append(reason)
        return await interrupt(session, reason=reason)

    channel.interrupt = counting
    return calls


async def _until(condition: Any, timeout: float = 5.0) -> None:
    """Wait for ``condition()`` rather than a fixed sleep a slow runner outlasts."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


class TestBargeInOnce:
    async def test_continued_speech_during_a_slow_barge_in_fires_it_once(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT("attends, stop", delay=0), vad=False
        )
        building, release = _held_store(kit)
        interrupts = _count_interrupts(channel)

        # Speech until the first barge-in is building its context (its speech
        # runs past min_speech_ms, a wall-clock measure)...
        async with asyncio.timeout(5):
            while not building:
                await backend.simulate_audio_received(session, _LOUD)
                await asyncio.sleep(0.01)
        # ...then five more energy runs land while it is still building.
        for _ in range(25):
            await backend.simulate_audio_received(session, _LOUD)
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)  # a barge-in they started is building too by now
        release.set()
        # Every barge-in that started ends in its interrupt.
        await _until(lambda: len(interrupts) >= len(building) and len(seen["barge_in"]) >= 1)

        assert len(building) == 1
        assert len(seen["barge_in"]) == 1
        assert interrupts == ["barge_in"]
        await kit.close()

    async def test_two_paths_deciding_on_the_same_playback_fire_once(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT(None, delay=0), vad=False
        )
        playback = channel._playing_sessions[session.id]  # noqa: SLF001

        # The partial path and the energy path both decided on this speech.
        await asyncio.gather(
            channel._handle_barge_in(session, playback, "r1"),  # noqa: SLF001
            channel._handle_barge_in(session, playback, "r1"),  # noqa: SLF001
        )
        await asyncio.sleep(0.05)

        assert len(seen["barge_in"]) == 1
        await kit.close()

    async def test_the_next_playback_can_still_be_interrupted(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT(None, delay=0), vad=False
        )
        first = channel._playing_sessions[session.id]  # noqa: SLF001
        await channel._handle_barge_in(session, first, "r1")  # noqa: SLF001

        second = TTSPlaybackState(
            session_id=session.id,
            text="As I was saying, Tuesday at ten.",
            started_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        channel._playing_sessions[session.id] = second  # noqa: SLF001
        await channel._handle_barge_in(session, second, "r1")  # noqa: SLF001
        await asyncio.sleep(0.05)

        assert len(seen["barge_in"]) == 2
        assert session.id not in channel._playing_sessions  # noqa: SLF001
        await kit.close()

    async def test_a_barge_in_whose_hooks_fail_still_cuts_the_playback(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT(None, delay=0), vad=False
        )
        playback = channel._playing_sessions[session.id]  # noqa: SLF001

        async def broken(room_id: str, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store unavailable")

        kit._build_context = broken  # noqa: SLF001
        await channel._handle_barge_in(session, playback, "r1")  # noqa: SLF001

        # Claimed and then abandoned, the playback would never be cut again.
        assert session.id not in channel._playing_sessions  # noqa: SLF001
        await kit.close()
