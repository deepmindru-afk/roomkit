"""ElevenLabs streaming text input over WebSocket.

Two sockets take text as it is produced and answer audio as it is ready:

- the Text to Speech socket (``/v1/text-to-speech/{voice}/stream-input``),
  for the v2 and v2.5 models;
- the Text to Dialogue socket (``/v1/text-to-dialogue/stream-input``), for
  v4 and v4 Turbo, which the first one refuses. It applies no voice settings.

One connection carries one response. Each chunk is flushed as it is sent: the
chunks are whole sentences, and without a flush both sockets hold the text
until enough of it has piled up, one to three seconds of a slow LLM. Neither
socket stitches a response to the previous ones: no request id comes back and
no previous text is taken.
"""

from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from urllib.parse import quote, urlencode

BASE_URL = "wss://api.elevenlabs.io"
# Both sockets close after 20 s without a client message; an LLM that pauses
# (a tool call mid-response) is covered by a keep-alive at half that.
KEEP_ALIVE_S = 10.0
CLOSE_TIMEOUT_S = 1.0

_SPEECH_MODELS = ("eleven_multilingual_v2", "eleven_flash_v2", "eleven_turbo_v2")
_DIALOGUE_MODELS = ("eleven_v4",)
_END = object()


class ElevenLabsSocket(ABC):
    """The URL and the messages of one ElevenLabs input-streaming socket."""

    path: str
    final_key: str

    def url(self, voice_id: str, query: dict[str, Any]) -> str:
        path = self.path.format(voice_id=quote(voice_id, safe=""))
        return f"{BASE_URL}{path}?{urlencode(query)}"

    @abstractmethod
    def open_message(self, voice_id: str, voice_settings: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def text_messages(self, voice_id: str, text: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def end_message(self) -> dict[str, Any]: ...

    @abstractmethod
    def keep_alive_message(self) -> dict[str, Any]: ...


class SpeechSocket(ElevenLabsSocket):
    """Text to Speech socket: one voice in the URL, voice settings applied."""

    path = "/v1/text-to-speech/{voice_id}/stream-input"
    final_key = "isFinal"

    def open_message(self, voice_id: str, voice_settings: dict[str, Any]) -> dict[str, Any]:
        return {"text": " ", "voice_settings": voice_settings}

    def text_messages(self, voice_id: str, text: str) -> list[dict[str, Any]]:
        # The socket expects each chunk to end with a space.
        return [{"text": text if text.endswith(" ") else f"{text} ", "flush": True}]

    def end_message(self) -> dict[str, Any]:
        return {"text": ""}

    def keep_alive_message(self) -> dict[str, Any]:
        return {"text": " "}


class DialogueSocket(ElevenLabsSocket):
    """Text to Dialogue socket: voices registered on open, no voice settings."""

    path = "/v1/text-to-dialogue/stream-input"
    final_key = "is_final"

    def open_message(self, voice_id: str, voice_settings: dict[str, Any]) -> dict[str, Any]:
        return {"voices": [voice_id]}

    def text_messages(self, voice_id: str, text: str) -> list[dict[str, Any]]:
        return [{"inputs": [{"text": text, "voice_id": voice_id}]}, {"flush": True}]

    def end_message(self) -> dict[str, Any]:
        return {"close_socket": True}

    def keep_alive_message(self) -> dict[str, Any]:
        return {"keep_alive": True}


def socket_for(model_id: str) -> ElevenLabsSocket | None:
    """The socket that streams *model_id*, or None (v3 has none)."""
    if model_id.startswith(_DIALOGUE_MODELS):
        return DialogueSocket()
    if model_id.startswith(_SPEECH_MODELS):
        return SpeechSocket()
    return None


async def stream_audio(
    socket: ElevenLabsSocket,
    *,
    api_key: str,
    voice_id: str,
    query: dict[str, Any],
    voice_settings: dict[str, Any],
    text_stream: AsyncIterator[str],
    keep_alive_s: float = KEEP_ALIVE_S,
) -> AsyncGenerator[bytes, None]:
    """Send *text_stream* over one socket and yield the audio it answers.

    Ends when the server marks the audio final. A server error raises, and so
    do an error of the text stream and a socket closed before the final flag.
    Closing the generator early (a barge-in) stops the sender and closes the
    socket without waiting on the server.
    """
    try:
        import websockets
    except ImportError as exc:
        raise ImportError(
            "websockets is required for ElevenLabs streaming input. "
            "Install with: pip install 'roomkit[elevenlabs]'"
        ) from exc

    # Audio comes faster than a paced transport plays it. An unbounded queue
    # keeps the socket reading, so a close frame never waits behind unread
    # audio, and the close handshake is not given the default ten seconds.
    async with websockets.connect(
        socket.url(voice_id, query),
        additional_headers={"xi-api-key": api_key},
        max_queue=None,
        close_timeout=CLOSE_TIMEOUT_S,
    ) as ws:
        await ws.send(json.dumps(socket.open_message(voice_id, voice_settings)))
        failure: list[Exception] = []
        sender = asyncio.create_task(
            _send_text(ws, socket, voice_id, text_stream, keep_alive_s, failure)
        )
        finalized = False
        try:
            async for raw in ws:
                message = json.loads(raw)
                if message.get("error"):
                    raise RuntimeError(f"ElevenLabs streaming input failed: {message['error']}")
                audio = message.get("audio")
                if audio:
                    yield base64.b64decode(audio)
                if message.get(socket.final_key):
                    finalized = True
                    break
        except websockets.exceptions.ConnectionClosed:
            if not failure:  # else the sender closed it, and its failure is raised below
                raise
        finally:
            await _stop(sender)
        # Raised after the finally, not in it: an exception already on its way
        # out (a server error, a cancellation, a barge-in) is never replaced.
        if failure:
            raise failure[0]
        if not finalized:
            raise RuntimeError("ElevenLabs closed the socket before the end of the audio")


async def _send_text(
    ws: Any,
    socket: ElevenLabsSocket,
    voice_id: str,
    text_stream: AsyncIterator[str],
    keep_alive_s: float,
    failure: list[Exception],
) -> None:
    """Forward each chunk, keep the socket alive while none comes, then end it.

    The next chunk is awaited as one task across keep-alives: cancelling it on
    a timeout would close the text stream. A failure is put in *failure* before
    the socket is closed: the close ends the receive loop, which cancels this
    task, and an exception raised here would go with it.
    """
    iterator = aiter(text_stream)
    pending: asyncio.Task[Any] | None = None
    try:
        while True:
            pending = asyncio.create_task(_next_text(iterator))
            while not (await asyncio.wait({pending}, timeout=keep_alive_s))[0]:
                await ws.send(json.dumps(socket.keep_alive_message()))
            text = pending.result()
            if text is _END:
                break
            if text:
                for message in socket.text_messages(voice_id, text):
                    await ws.send(json.dumps(message))
        await ws.send(json.dumps(socket.end_message()))
    except Exception as exc:
        failure.append(exc)
        await ws.close()
    finally:
        if pending is not None:
            await _stop(pending)


async def _stop(task: asyncio.Task[Any]) -> None:
    """Cancel *task* and wait for it to end, its outcome retrieved.

    Waited, not awaited: awaiting would re-raise the task's own cancellation,
    and suppressing that would also swallow a cancellation of the caller.
    """
    task.cancel()
    await asyncio.wait({task})
    if not task.cancelled():
        task.exception()


async def _next_text(iterator: AsyncIterator[str]) -> Any:
    try:
        return await anext(iterator)
    except StopAsyncIteration:
        return _END
