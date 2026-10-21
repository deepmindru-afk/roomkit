"""Wire-level pieces of Mistral's Voxtral transcription API.

What the service sends and how it fails, kept apart from
:mod:`roomkit.voice.stt.voxtral`, which drives the connection: the language
codes, the realtime event mapping, the file endpoint's answer, and the error
shapes of the REST and WebSocket endpoints. The public import path stays
``roomkit.voice.stt.voxtral``.
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import ProviderError
from roomkit.voice.base import TranscriptionResult

REALTIME_PATH = "/v1/audio/transcriptions/realtime"

STREAMING_DELAYS_MS = "80 to 1200, or 2400"
"""The streaming delays the realtime model was trained for: any other value is
refused when the session is configured (measured 2026-10-21)."""

_DELTA = "transcription.text.delta"
_DONE = "transcription.done"

# 1011 is a server fault and 1013 a rate limit: both are worth a new stream.
_RETRYABLE_CLOSE_CODES = frozenset({1011, 1013})


class VoxtralSTTError(ProviderError):
    """A request Mistral's transcription API refused or failed.

    Attributes:
        code: Mistral's error ``code`` (``"1500"`` for an unknown model,
            ``3051`` for a refused realtime setting), or the WebSocket close
            code as a string when the service closed without an error event.
        error_type: Mistral's error ``type`` (``"invalid_model"``), when it
            sent one.
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
        super().__init__(
            message, retryable=retryable, provider="VoxtralSTT", status_code=status_code
        )
        self.code = code
        self.error_type = error_type


def valid_streaming_delay(delay_ms: int) -> bool:
    """Whether the realtime model takes *delay_ms* (:data:`STREAMING_DELAYS_MS`)."""
    return 80 <= delay_ms <= 1200 or delay_ms == 2400


def language_code(language: str | None) -> str | None:
    """The code the service takes for a BCP-47 *language*: ``"fr-CA"`` -> ``"fr"``.

    Voxtral takes ISO 639-1 codes only and refuses a region (``"fr-CA"``,
    measured 2026-10-21), so the region is dropped. ``None`` keeps automatic
    detection.

    Raises:
        ValueError: The language is not a two-letter code once its region is
            dropped.
    """
    if language is None:
        return None
    code = language.strip().replace("_", "-").split("-")[0].lower()
    if len(code) != 2 or not code.isascii() or not code.isalpha():
        raise ValueError(
            f"Voxtral takes a two-letter ISO 639-1 language code such as 'fr', got {language!r}"
        )
    return code


class Transcript:
    """The text of one realtime session, as RoomKit results.

    Each ``transcription.text.delta`` adds text after the text already sent,
    final as it arrives and carrying its own spaces, so a partial is the text
    so far. Without turn detection the session is one utterance, and
    ``transcription.done`` gives its text of record.

    With turn detection the text is cut into turns. ``turn.start`` opens one
    (a speech start) and ``turn.end`` sets :attr:`closing`; the caller then
    calls :meth:`close` once the turn's last words have had time to arrive.
    The turn events come at once while the text runs a streaming delay
    behind the audio, so a turn's last words can follow its ``turn.end`` by
    nearly that delay, and even the next ``turn.start`` (measured 2026-10-21:
    "Italy?" arrived after the next turn had started). The text that comes
    before :meth:`close` is the ending turn's; a ``turn.start`` meanwhile is
    a speech start, its turn opening at the close. Text that arrives after
    its turn's final belongs to a turn already given: it is counted in
    :attr:`late_chars`, never sent as a turn of its own.
    """

    def __init__(self, *, turns: bool) -> None:
        self._turns = turns
        self._text = ""
        self._given = False
        self._next_open = False
        self._next_ended = False
        self.closing = False
        self.late_chars = 0

    def feed(self, event: dict[str, Any]) -> list[TranscriptionResult]:
        """The results one server event gives, in order.

        Raises:
            VoxtralSTTError: The event reports an error.
        """
        kind = event.get("type")
        if kind == "error":
            raise event_error(event)
        if kind == _DELTA:
            return self._delta(str(event.get("text") or ""))
        if not self._turns:
            if kind == _DONE:
                return self._final(str(event.get("text") or self._text))
            return []
        if kind == _DONE:
            return self.close()
        return self._turn_event(kind)

    def close(self) -> list[TranscriptionResult]:
        """Give the ending turn as a final, ``[]`` when it holds no words.

        A turn that started while this one was closing is open from here, and
        :attr:`closing` again if it has ended too.
        """
        text, self._text = self._text, ""
        self._given = not self._next_open
        self.closing = self._next_open and self._next_ended
        self._next_open = self._next_ended = False
        return self._final(text)

    def _turn_event(self, kind: Any) -> list[TranscriptionResult]:
        if kind == "turn.start":
            if self.closing:
                # The latest turn to start is the one open after the close.
                self._next_open, self._next_ended = True, False
            else:
                self._given = False
            return [TranscriptionResult(text="", is_final=False, is_speech_start=True)]
        if kind == "turn.end":
            if self._next_open:
                self._next_ended = True
            else:
                self.closing = True
        return []

    def _delta(self, text: str) -> list[TranscriptionResult]:
        if self._turns and self._given:
            self.late_chars += len(text.strip())
            return []
        self._text += text
        so_far = self._text.strip()
        return [TranscriptionResult(text=so_far, is_final=False)] if so_far else []

    @staticmethod
    def _final(text: str) -> list[TranscriptionResult]:
        text = text.strip()
        return [TranscriptionResult(text=text, is_final=True)] if text else []


def is_done(event: dict[str, Any]) -> bool:
    """Whether *event* ends the session: the service then drops the connection."""
    return event.get("type") == _DONE


def rest_result(body: dict[str, Any]) -> TranscriptionResult:
    """The file endpoint's answer ``{text, language, segments, usage}``.

    ``language`` is ``null`` in every answer measured, pinned or detected
    (2026-10-21); it is passed on as the service reports it.
    """
    language = body.get("language")
    return TranscriptionResult(
        text=str(body.get("text") or "").strip(),
        language=language if isinstance(language, str) and language else None,
    )


def event_error(event: dict[str, Any]) -> VoxtralSTTError:
    """A realtime ``error`` event, ``{"error": {"message", "code"}}``.

    The service sends one for a setting it refuses, then keeps the session
    open: none of these is worth retrying unchanged.
    """
    raw = event.get("error")
    error: dict[str, Any] = raw if isinstance(raw, dict) else {}
    code = error.get("code")
    return VoxtralSTTError(
        f"VoxtralSTT error: {error.get('message') or code or 'unknown error'}",
        code=str(code) if code is not None else None,
    )


def stream_error(exc: Exception) -> VoxtralSTTError:
    """A failed connection, handshake or close, as an exception.

    Retryable unless the service refused: a server fault (1011), a rate limit
    (1013, or HTTP 429 at the upgrade), a 5xx at the upgrade, a timeout, a
    network error, and a connection dropped without a close frame may all work
    on a new stream. A refusal is any other close code or a 4xx at the upgrade
    (401 for a wrong key, 403 with no body for a model the key cannot use).
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
    if status == 403:
        reason += " (is the realtime model id right, and open to this key?)"
    return VoxtralSTTError(
        f"VoxtralSTT stream failed ({label}): {reason}",
        code=str(code) if code is not None else None,
        retryable=retryable,
        status_code=status,
    )


def http_error(response: Any) -> VoxtralSTTError:
    """A REST error body as an exception.

    Mistral answers ``{"object": "error", "message", "type", "code"}``, a
    rejected key ``{"detail": "Invalid API Key"}``, and a refused field
    ``{"detail": [{"msg", "loc"}]}``.
    """
    try:
        body = response.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    detail = body.get("detail")
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg")) for item in detail if isinstance(item, dict))
    message = body.get("message") or detail or response.text[:200]
    code = body.get("code")
    status = response.status_code
    return VoxtralSTTError(
        f"VoxtralSTT request failed ({status}): {message}",
        code=str(code) if code is not None else None,
        error_type=body.get("type"),
        retryable=status == 429 or status >= 500,
        status_code=status,
    )
