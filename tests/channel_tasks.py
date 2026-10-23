"""The work a voice or realtime channel scheduled, awaited instead of a pause.

A channel runs what a callback starts (a segment's transcription and routing,
a barge-in, a reply's speech, a tool call) in a task it tracks. A test that fed
it audio or a provider event waits for those tasks to have run. Work that lasts
as long as the session or outlives a reply on purpose (a continuous STT loop, a
playback's echo tail) is named in *lasting* and left out, by its task's name
prefix or its coroutine's name.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from typing import Any


def _lasts(task: asyncio.Task[Any], lasting: Collection[str]) -> bool:
    coroutine = getattr(task.get_coro(), "__name__", "")
    return any(task.get_name().startswith(name) or coroutine == name for name in lasting)


async def settled(channel: Any, *, lasting: Collection[str] = (), timeout: float = 5.0) -> None:
    """Until every task *channel* scheduled has run, those *lasting* left out."""
    async with asyncio.timeout(timeout):
        while running := [
            task
            for task in channel._scheduled_tasks
            if not task.done() and not _lasts(task, lasting)
        ]:
            await asyncio.wait(running)
