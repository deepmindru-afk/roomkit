"""Drive a blocking audio generator from a worker thread, cancellable."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Generator
from typing import Any


async def iterate_in_thread(
    frames: Generator[bytes, None, None], cancel: threading.Event
) -> AsyncGenerator[bytes, None]:
    """Drive a blocking generator from a worker thread.

    Closing this iterator (a barge-in) sets *cancel* and waits for the thread
    to stop, even if the waiting task is cancelled again meanwhile, so a lock
    held around it is never released while the model is still busy.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    done = object()

    def run() -> None:
        # The generator stops on *cancel* itself (checked every frame), so it
        # ends through its own cleanup rather than suspended mid-frame.
        try:
            for pcm in frames:
                loop.call_soon_threadsafe(queue.put_nowait, pcm)
        except BaseException as exc:  # handed to the consumer, raised there
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            frames.close()
            loop.call_soon_threadsafe(queue.put_nowait, done)

    worker = loop.run_in_executor(None, run)
    try:
        while (item := await queue.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()
        cancelled = False
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
