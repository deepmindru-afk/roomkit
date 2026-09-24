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
from tests.test_interruption_semantic_wait import _room, _SlowPartialSTT, _talk


def _slow_store(kit: Any, delay: float) -> None:
    """Make every context build wait ``delay`` seconds, as a loaded store does."""
    build = kit._build_context  # noqa: SLF001

    async def slow(room_id: str, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(delay)
        return await build(room_id, *args, **kwargs)

    kit._build_context = slow  # noqa: SLF001


def _count_interrupts(channel: Any) -> list[str]:
    calls: list[str] = []
    interrupt = channel.interrupt

    async def counting(session: Any, *, reason: str = "explicit") -> bool:
        calls.append(reason)
        return await interrupt(session, reason=reason)

    channel.interrupt = counting
    return calls


class TestBargeInOnce:
    async def test_continued_speech_during_a_slow_barge_in_fires_it_once(self) -> None:
        kit, channel, session, backend, seen = await _room(
            _SlowPartialSTT("attends, stop", delay=0.2), vad=False
        )
        _slow_store(kit, 0.3)
        interrupts = _count_interrupts(channel)

        # 600 ms of speech: five more energy runs land while the first
        # barge-in is still building its context.
        await _talk(backend, session, 0.6)
        await asyncio.sleep(0.4)

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
