"""Fish Audio streaming text input over WebSocket (``/v1/tts/live``).

One connection carries one response. Every message is msgpack: the client
sends ``start`` (the render settings), a ``text`` and a ``flush`` per sentence,
then ``stop``; the server answers ``audio`` events and one ``finish``, whose
``reason`` is ``stop`` or ``error``, and closes.

A flush per sentence is what keeps both latency and prosody (measured
2026-10-14): without one the server waits for about 300 characters or the
stop, and a flush after every token breaks the words apart. No keep-alive is
needed: a socket left idle 130 s after ``start`` or between sentences still
answered. Over the account's concurrency limit (5 sessions on the starter
tier) the extra session is closed with code 1005 when its text arrives, with
no ``finish``: that reads as a socket closed before the end of the audio.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

try:
    import msgpack
    import websockets
    from websockets.exceptions import ConnectionClosed, InvalidStatus
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "msgpack and websockets are required for Fish Audio. "
        "Install with: pip install 'roomkit[fish-audio]'"
    ) from exc

from roomkit.voice.tts._fish_audio_text import StreamedText

logger = logging.getLogger("roomkit.voice.tts.fish_audio")

CLOSE_TIMEOUT_S = 1.0


class FishAudioSocketError(Exception):
    """The socket refused the session or ended it without finishing the audio.

    ``status_code`` is the handshake's HTTP status when the server refused the
    connection (402 when the account has no API credit), else None.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


async def stream_audio(
    url: str,
    *,
    headers: dict[str, str],
    request: dict[str, Any],
    text_stream: AsyncIterator[str],
    expressive: bool,
) -> AsyncGenerator[bytes, None]:
    """Send *text_stream* over one socket and yield the audio bytes it answers.

    Each item of *text_stream* is a sentence, sent with a flush; one with no
    word to say waits for the next (see ``_fish_audio_text``). Ends on the
    server's ``finish``. A ``finish`` with an error, a refused handshake, an
    error of the text stream and a socket closed before ``finish`` raise.
    Closing the generator early (a barge-in) stops the sender and closes the
    socket without waiting on the server.
    """
    try:
        # Audio comes faster than a paced transport plays it. An unbounded
        # queue keeps the socket reading, so a close frame never waits behind
        # unread audio, and the close handshake is not given ten seconds.
        connection = await websockets.connect(
            url, additional_headers=headers, max_queue=None, close_timeout=CLOSE_TIMEOUT_S
        )
    except InvalidStatus as exc:
        status = exc.response.status_code
        detail = exc.response.body.decode(errors="replace")[:200] if exc.response.body else ""
        raise FishAudioSocketError(
            f"Fish Audio refused the socket ({status}): {detail or 'no detail'}",
            status_code=status,
        ) from exc
    async with connection as ws:
        try:
            await ws.send(msgpack.packb({"event": "start", "request": request}))
        except ConnectionClosed:
            raise FishAudioSocketError(
                f"Fish Audio closed the socket before the session started (code {ws.close_code})"
            ) from None
        failure: list[Exception] = []
        sender = asyncio.create_task(_send_text(ws, text_stream, expressive, failure))
        finished = False
        try:
            async for frame in ws:
                message = msgpack.unpackb(frame) if isinstance(frame, bytes) else {}
                event = message.get("event")
                if event == "audio":
                    yield message["audio"]
                elif event == "finish":
                    if message.get("reason") == "error":
                        raise FishAudioSocketError("Fish Audio finished the audio with an error")
                    finished = True
                    break
        except ConnectionClosed:
            if not failure:  # else the sender closed it, and its failure is raised below
                raise FishAudioSocketError(
                    f"Fish Audio closed the socket before the end of the audio "
                    f"(code {ws.close_code}; the account's concurrency limit closes "
                    "extra sessions this way)"
                ) from None
        finally:
            await _stop(sender)
        # Raised after the finally, not in it: an exception already on its way
        # out (a server error, a cancellation, a barge-in) is never replaced.
        if failure:
            raise failure[0]
        if not finished:
            raise FishAudioSocketError(
                f"Fish Audio closed the socket before the end of the audio (code {ws.close_code})"
            )


async def _send_text(
    ws: Any, text_stream: AsyncIterator[str], expressive: bool, failure: list[Exception]
) -> None:
    """Forward each sentence with a flush, then stop.

    A failure is put in *failure* before the socket is closed: the close ends
    the receive loop, which cancels this task, and an exception raised here
    would go with it. A socket the server closed is not the sender's failure:
    the receive loop sees the same close and reports it.
    """
    prepared = StreamedText(expressive=expressive)
    try:
        async for sentence in text_stream:
            await _send(ws, prepared.feed(sentence))
        await _send(ws, prepared.flush())
        await ws.send(msgpack.packb({"event": "stop"}))
    except ConnectionClosed:
        return
    except Exception as exc:
        failure.append(exc)
        await ws.close()


async def _send(ws: Any, text: str) -> None:
    """Send *text* and flush it; an empty text is not sent."""
    if text:
        await ws.send(msgpack.packb({"event": "text", "text": text}))
        await ws.send(msgpack.packb({"event": "flush"}))


async def _stop(task: asyncio.Task[Any]) -> None:
    """Cancel *task* and wait for it to end, its outcome retrieved.

    Waited, not awaited: awaiting would re-raise the task's own cancellation,
    and suppressing that would also swallow a cancellation of the caller.
    """
    task.cancel()
    await asyncio.wait({task})
    if not task.cancelled():
        task.exception()
