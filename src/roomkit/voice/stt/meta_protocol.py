"""Wire-level pieces of the Meta Muse Voice Transcribe protocol.

What the service sends and how it fails, kept apart from
:mod:`roomkit.voice.stt.meta`, which drives the connection: the event mapping,
the error shapes of the REST and WebSocket endpoints, and the WAV container the
REST endpoint takes. The public import path stays ``roomkit.voice.stt.meta``.
"""

from __future__ import annotations

import io
import wave
from typing import Any

from roomkit.providers.ai.base import ProviderError
from roomkit.voice.base import TranscriptionResult

# 1011 is a server fault ("Max session duration reached" included) and 1013 a
# rate limit: both are worth a new stream. 1008 is a refused request.
_RETRYABLE_CLOSE_CODES = frozenset({1011, 1013})

WAV_MIME_TYPES: frozenset[str] = frozenset(
    {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
)
"""Media types a ``data:`` URI may declare for a clip: the endpoint takes WAV only."""


class MetaSTTError(ProviderError):
    """A request the Meta Model API refused or failed.

    Attributes:
        code: Meta's ``error.code`` / ``errorCode`` (e.g.
            ``"billing_not_configured"``), or the WebSocket close code as a
            string when the service closed without an error frame.
        error_type: Meta's ``error.type`` / ``errorType`` (e.g.
            ``"billing_error"``), when it sent one.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable, provider="MetaSTT", status_code=status_code)
        self.code = code
        self.error_type = error_type


def to_result(event: dict[str, Any]) -> TranscriptionResult | None:
    """Map one server event to a result, ``None`` for the ones RoomKit ignores.

    One mapping serves both modes, as observed on the live service: in
    ``ENDPOINTING`` a turn's final is ``speechComplete`` and the stream ends
    on an *empty* ``transcript`` marked final; in ``PUSH_TO_TALK`` there is no
    ``speechComplete`` and the final is that ``transcript``, with the text.
    ``speechEnd`` precedes ``speechComplete`` and adds nothing to it.
    """
    kind = event.get("type")
    if kind == "error":
        raise event_error(event)
    if kind == "speechStart":
        return TranscriptionResult(text="", is_final=False, is_speech_start=True)
    if kind not in ("transcript", "speechComplete"):
        return None
    text = str(event.get("transcript") or "").strip()
    if not text:
        return None
    is_final = kind == "speechComplete" or bool(event.get("final"))
    return TranscriptionResult(text=text, is_final=is_final)


def event_error(event: dict[str, Any]) -> MetaSTTError:
    """The error frame the service sends before closing a refused stream."""
    return MetaSTTError(
        f"Meta STT error: {event.get('message') or 'unknown error'}",
        code=event.get("errorCode"),
        error_type=event.get("errorType"),
    )


def stream_error(exc: Exception) -> MetaSTTError:
    """A failed connection, handshake or close, as an exception.

    Retryable unless the service refused: a server fault (1011), a rate limit
    (1013, or HTTP 429 at the upgrade), a 5xx at the upgrade, a timeout, a
    network error, and a connection dropped without a close frame may all work
    on a new stream. A refusal is a close code (1008) or a 4xx at the upgrade.
    """
    received = getattr(exc, "rcvd", None)
    code = getattr(received, "code", None)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    # Only websockets' ConnectionClosed carries ``rcvd``, and ``None`` there
    # means the peer vanished without a close frame, not that it refused.
    dropped = hasattr(exc, "rcvd") and received is None
    retryable = (
        code in _RETRYABLE_CLOSE_CODES
        or dropped
        or (status is not None and (status == 429 or status >= 500))
        or isinstance(exc, TimeoutError | OSError)
    )
    label = code if code is not None else status if status is not None else type(exc).__name__
    reason = getattr(received, "reason", "") or str(exc) or type(exc).__name__
    return MetaSTTError(
        f"Meta STT stream failed ({label}): {reason}",
        code=str(code) if code is not None else None,
        retryable=retryable,
        status_code=status,
    )


def http_error(response: Any) -> MetaSTTError:
    """A REST error body, ``{"error": {"code", "type", "message"}}``, as an exception."""
    try:
        error = response.json().get("error") or {}
    except ValueError:
        error = {}
    status = response.status_code
    return MetaSTTError(
        f"Meta STT request failed ({status}): {error.get('message') or response.text[:200]}",
        code=error.get("code"),
        error_type=error.get("type"),
        retryable=status == 429 or status >= 500,
        status_code=status,
    )


def read_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """``(pcm, sample_rate, channels, sample_width)`` of a WAV file.

    Raises:
        ValueError: If *data* is not a PCM WAV file.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            pcm = reader.readframes(reader.getnframes())
            return pcm, reader.getframerate(), reader.getnchannels(), reader.getsampwidth()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"not a PCM WAV file: {exc}") from exc


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Mono 16-bit PCM wrapped in a WAV container — the only upload format taken."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()
