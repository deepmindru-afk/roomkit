"""One realtime session of Mistral's Voxtral transcription WebSocket.

Drives a stream for :class:`roomkit.voice.stt.voxtral.VoxtralSTTProvider`:
open the session and set it up before any audio (its settings are locked
from the first chunk), send the audio while reading the transcript, and end
the audio once the stream does. The service drops the connection right after
``transcription.done``, so reading stops there.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from roomkit.core.task_utils import cancel_and_wait
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.voxtral_protocol import (
    VoxtralSTTError,
    event_error,
    is_done,
    stream_error,
)
from roomkit.voice.stt.voxtral_transcript import SpeakerTurns, Transcript

logger = logging.getLogger("roomkit.voice.stt.voxtral")

_APPEND_S = 0.1
"""Longest audio one append carries, so a backlog (the pre-roll) does not
leave as one large frame."""

_CLOSE_TIMEOUT_S = 0.2
"""How long closing the socket waits for the service's answer to the close
frame before dropping the connection. A session still open when its stream
ends is abandoned, with nothing left to read, and Mistral answers the frame
about 1.1 s later (measured 2026-10-21): a channel closing would wait that
long. A session that ended normally has no wait: the service drops the
connection after ``transcription.done``."""

PcmOf = Callable[[AudioChunk, int], bytes]
"""A chunk's audio as mono 16-bit PCM at the given rate."""


def _import_websockets() -> tuple[Any, Any]:
    """``(connect, exceptions)`` from ``websockets``, which streaming needs."""
    try:
        from websockets import exceptions
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise ImportError(
            "websockets is required for VoxtralSTTProvider streaming. "
            "Install with: pip install roomkit[mistral-voice]"
        ) from exc
    return connect, exceptions


@dataclass(frozen=True)
class SessionSetup:
    """What a session is opened with.

    Attributes:
        url: The realtime endpoint, the model in its query string.
        api_key: Mistral API key (sent as ``Bearer``).
        settings: The ``session`` object of the one ``session.update``.
        rate: The rate of the PCM the session is sent.
        turns: Whether the settings ask for turn detection.
        diarize: Whether they ask for speaker labels (no turn events then).
        final_wait_s: How long a turn's final waits after ``turn.end``.
        next_turn_wait_s: How long it waits at most after the next
            ``turn.start``, whose own words come a streaming delay later.
        pause_s: How long without new words ends a diarized turn.
        handshake_timeout_s: How long to wait for the session to open and
            accept its settings.
        final_timeout_s: How long to wait, once the audio ended, for
            ``transcription.done``.
    """

    url: str
    api_key: str = field(repr=False)
    settings: dict[str, Any]
    rate: int
    turns: bool
    final_wait_s: float
    next_turn_wait_s: float
    handshake_timeout_s: float
    final_timeout_s: float
    diarize: bool = False
    pause_s: float = 0.0


@dataclass
class _End:
    """Whether the wait for ``transcription.done`` ran out."""

    timed_out: bool = False


class RealtimeSession:
    """An open session, ready for audio."""

    def __init__(self, ws: Any, setup: SessionSetup, pcm_of: PcmOf) -> None:
        self._ws = ws
        self._setup = setup
        self._pcm_of = pcm_of
        self._end = _End()

    @classmethod
    async def open(cls, setup: SessionSetup, pcm_of: PcmOf) -> RealtimeSession:
        """Connect, wait for the session, and have it accept the settings.

        Raises:
            VoxtralSTTError: The service refused the connection (401 for a
                wrong key, 403 for a model the key cannot use) or a setting.
        """
        connect, _ = _import_websockets()
        try:
            # One bound for the connection and both handshake events.
            async with asyncio.timeout(setup.handshake_timeout_s):
                ws = await connect(
                    setup.url,
                    additional_headers={"Authorization": f"Bearer {setup.api_key}"},
                    open_timeout=None,
                    close_timeout=_CLOSE_TIMEOUT_S,
                )
                try:
                    created = await _await_event(ws, "session.created")
                    await ws.send(
                        json.dumps({"type": "session.update", "session": setup.settings})
                    )
                    await _await_event(ws, "session.updated")
                except BaseException:
                    await ws.close()
                    raise
        except VoxtralSTTError:
            raise
        except Exception as exc:
            raise stream_error(exc) from exc
        session = created.get("session") or {}
        logger.debug(
            "Voxtral STT session %s open (%s, %d Hz, language %s)",
            session.get("request_id"),
            session.get("model"),
            setup.rate,
            setup.settings.get("language") or "auto",
        )
        return cls(ws, setup, pcm_of)

    async def transcribe(
        self, first: AudioChunk, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Send *first* then *audio_stream*, yielding the transcript as it comes.

        The session is closed when the generator ends, however it ends.
        """
        setup = self._setup
        transcript: Transcript | SpeakerTurns = (
            SpeakerTurns(pause_s=setup.pause_s)
            if setup.diarize
            else Transcript(
                turns=setup.turns,
                final_wait_s=setup.final_wait_s,
                next_turn_wait_s=setup.next_turn_wait_s,
            )
        )
        sender = asyncio.create_task(self._send_audio(first, audio_stream))
        try:
            async with contextlib.aclosing(self._results(transcript)) as results:
                async for result in results:
                    yield result
        finally:
            try:
                await cancel_and_wait(sender)
            finally:
                await self._ws.close()

    async def _results(
        self, transcript: Transcript | SpeakerTurns
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Read the session's events into results, until ``transcription.done``.

        A turn is given when the transcript's :meth:`deadline` passes: once
        ``final_wait_s`` has passed after the model ended it, its last words
        included, or sooner when the next turn starts, before that turn's own
        words; for a diarized session, after ``pause_s`` without new words.
        """
        _, exceptions = _import_websockets()
        loop = asyncio.get_running_loop()
        close_at: float | None = None
        while True:
            try:
                async with asyncio.timeout_at(close_at):
                    message = await self._ws.recv()
            except TimeoutError:
                for result in transcript.close():
                    yield result
                # A turn that started and ended during the wait now closes.
                close_at = transcript.deadline(loop.time(), None)
                continue
            except exceptions.ConnectionClosed as exc:
                raise self._ended_early(exc) from exc
            if not isinstance(message, str):
                continue
            event = json.loads(message)
            for result in transcript.feed(event):
                yield result
            if late := transcript.take_late():
                # A kept stream lasts the whole call: say it when it happens.
                logger.warning(
                    "Voxtral STT: %d characters of a turn arrived after its final and were "
                    "dropped; raise final_wait_ms (now %.2f s) to keep them",
                    late,
                    self._setup.final_wait_s,
                )
            if is_done(event):
                return
            close_at = transcript.deadline(loop.time(), close_at)

    async def _send_audio(
        self, first: AudioChunk, audio_stream: AsyncIterator[AudioChunk]
    ) -> None:
        """Append the audio, end it once the stream ends, then bound the wait for the end.

        A failing audio source still ends the audio, so the service answers
        with the transcript of what it heard instead of waiting for more. If
        ``transcription.done`` does not come within ``final_timeout_s``, the
        socket is closed, which ends the reader.
        """
        _, exceptions = _import_websockets()
        try:
            await self._append(first)
            if not first.is_final:
                async for chunk in audio_stream:
                    if chunk.data:
                        await self._append(chunk)
                    if chunk.is_final:
                        break
        except exceptions.ConnectionClosed:
            # The reader sees the close and reports it; nothing to add here.
            logger.debug("Voxtral STT stream closed while sending audio")
            return
        except Exception:
            logger.exception("Voxtral STT audio source failed; ending the audio sent so far")
        try:
            await self._ws.send(json.dumps({"type": "input_audio.end"}))
        except exceptions.ConnectionClosed:
            return
        await asyncio.sleep(self._setup.final_timeout_s)
        self._end.timed_out = True
        await self._ws.close()

    async def _append(self, chunk: AudioChunk) -> None:
        """Send ``chunk`` as base64 PCM, in appends of at most :data:`_APPEND_S`."""
        rate = self._setup.rate
        pcm = self._pcm_of(chunk, rate)
        step = int(rate * _APPEND_S) * 2
        for start in range(0, len(pcm), step):
            piece = base64.b64encode(pcm[start : start + step]).decode("ascii")
            await self._ws.send(json.dumps({"type": "input_audio.append", "audio": piece}))

    def _ended_early(self, exc: Exception) -> VoxtralSTTError:
        """Why the session ended before ``transcription.done``: the wait ran out, or the line."""
        if self._end.timed_out:
            return VoxtralSTTError(
                f"VoxtralSTT got no end of transcript within {self._setup.final_timeout_s:g} s "
                "of the end of the audio",
                retryable=True,
            )
        return stream_error(exc)


async def _await_event(ws: Any, kind: str) -> dict[str, Any]:
    """Read until the service sends ``kind``; an error event raises."""
    while True:
        event = json.loads(await ws.recv())
        if event.get("type") == "error":
            raise event_error(event)
        if event.get("type") == kind:
            return dict(event)


async def first_audio(audio_stream: AsyncIterator[AudioChunk]) -> AudioChunk | None:
    """The first chunk carrying audio, or ``None`` if the stream ends without any.

    The session is opened only once there is audio to send, and its rate is
    set from this chunk's.
    """
    async for chunk in audio_stream:
        if chunk.data:
            return chunk
        if chunk.is_final:
            return None
    return None
