"""Mistral Voxtral text-to-speech provider (Voxtral Speech 3).

``POST /v1/audio/speech`` renders one complete text per request, streamed back
as server-sent events while it renders. Mistral's realtime speech model, which
takes text while it is still being written, was not served at launch
(October 2026), so the channel's sentence splitting feeds this provider.

Audio is 24 kHz mono. The ``pcm`` format the stream asks for is 32-bit float
(measured 2026-10-21, not documented); it is converted to the 16-bit PCM
RoomKit's audio path carries.

Voxtral 3 voices take inline emotion and pace tags in the text
(``[emotion=happy]``, ``[pace=slow]``); a :class:`~roomkit.voice.tts.filters.StripBrackets`
filter in front of this provider drops them unless told to keep them.
Install with ``pip install roomkit[mistral-voice]``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import sys
from array import array
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from roomkit.providers.ai.base import ProviderError
from roomkit.providers.utils import http_timeout
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.audio_utils import collect_wav_content
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.voices import VoiceInfo, filter_voices

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.voxtral")

DEFAULT_MODEL = "voxtral-mini-tts-3"
DEFAULT_VOICE = "voice_478"
SAMPLE_RATE = 24000

_FLOAT_BYTES = 4

# The Voxtral 3 presets the voices endpoint listed on 2026-10-21, as
# (slug, name, language, gender). Voxtral 2's presets (``gb_jane_neutral``…)
# are still accepted and listed by ``list_voices``, not curated here.
_PRESETS: tuple[tuple[str, str, str, str], ...] = (
    ("voice_478", "Sheila", "en-US", "female"),
    ("voice_447", "Margaret", "en-US", "female"),
    ("voice_3150", "James", "en-US", "male"),
    ("voice_485", "Rob", "en-US", "male"),
    ("voice_590", "Harriet", "en-GB", "female"),
    ("voice_1787", "Mary", "en-GB", "female"),
    ("voice_1786", "Sky", "en-GB", "female"),
    ("voice_3149", "Hugo", "en-GB", "male"),
    ("voice_3147", "Howard", "en-GB", "male"),
    ("voice_591", "Fred", "en-GB", "male"),
    ("voice_1788", "David", "en-GB", "male"),
    ("voice_592", "Marie", "fr", "female"),
    ("voice_3137", "Manon", "fr", "female"),
    ("voice_589", "Lea", "fr", "female"),
    ("voice_3141", "Camille", "fr", "female"),
    ("voice_479", "Maryam", "ar", "female"),
    ("voice_3143", "Jordi", "ca", "male"),
    ("voice_481", "Heidi", "de", "female"),
    ("voice_483", "Elena", "es", "female"),
    ("voice_3153", "Himanshu", "hi", "male"),
    ("voice_3139", "Ananya", "hi", "female"),
    ("voice_3146", "Olivia", "hu", "female"),
    ("voice_482", "Lorenzo", "it", "male"),
    ("voice_3133", "Haruto", "ja", "male"),
    ("voice_3136", "Sung", "ko", "male"),
    ("voice_3135", "Jin", "ko", "male"),
    ("voice_3145", "Lotte", "nl", "female"),
    ("voice_3134", "Lars", "no", "male"),
    ("voice_3154", "Leon", "pl", "male"),
    ("voice_480", "Gabriela", "pt", "female"),
    ("voice_3155", "Irina", "ro", "female"),
    ("voice_3132", "Tatiana", "ru", "female"),
    ("voice_3138", "Oran", "tr", "male"),
    ("voice_3140", "Taras", "uk", "male"),
    ("voice_3144", "Huong", "vi", "female"),
    ("voice_3142", "Mei", "zh", "female"),
)


class VoxtralTTSError(ProviderError):
    """A speech request Mistral refused or failed.

    Attributes:
        code: Mistral's error ``code`` (``"1902"`` for an unknown voice).
        error_type: Mistral's error ``type`` (``"invalid_voice"``).
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
            message, retryable=retryable, provider="VoxtralTTS", status_code=status_code
        )
        self.code = code
        self.error_type = error_type


@dataclass
class VoxtralTTSConfig:
    """Configuration for :class:`VoxtralTTSProvider`.

    Attributes:
        api_key: Mistral API key (sent as ``Bearer``).
        model: Speech model.
        voice: A preset's slug (``voice_478``, ``fr_marie_neutral``) or the id
            of a voice saved in the workspace.
        expressiveness: How strongly the speech follows the voice, the text
            and its tags, from 1.0; higher trades voice similarity for a
            stronger delivery. ``None`` keeps the service's default.
        base_url: API root, without a version.
        timeout: Read budget of a render, in seconds.
        connect_timeout: TCP connect budget, in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    expressiveness: float | None = None
    base_url: str = "https://api.mistral.ai"
    timeout: float = 60.0
    connect_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("VoxtralTTSConfig.api_key is required")
        if not self.api_key.isascii():
            # Refused here, not as a UnicodeEncodeError from the HTTP client.
            raise ValueError(
                "VoxtralTTSConfig.api_key holds a non-ASCII character (a placeholder such as "
                "'…' copied as is?): an HTTP Authorization header cannot carry it"
            )
        if self.expressiveness is not None and self.expressiveness < 1.0:
            raise ValueError("expressiveness must be at least 1.0")


class VoxtralTTSProvider(TTSProvider):
    """Speech on Mistral's Voxtral Speech 3, each text rendered whole and streamed."""

    def __init__(self, config: VoxtralTTSConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._voices: list[VoiceInfo] | None = None

    @property
    def name(self) -> str:
        return "VoxtralTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """The Voxtral 3 presets, as the voices endpoint listed them on 2026-10-21."""
        return [
            VoiceInfo(id=slug, name=name, language=language, gender=gender)
            for slug, name, language, gender in _PRESETS
        ]

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """Every voice the workspace can use: the presets of both generations, then
        its saved voices, whose ``id`` is the UUID ``voice`` takes.

        The list is read once, every page of it, and kept.
        """
        if self._voices is None:
            self._voices = [_voice_info(entry) for entry in await self._voice_entries()]
        return filter_voices(self._voices, language=language, gender=gender, query=query)

    async def _voice_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        params: dict[str, Any] = {"limit": 100}
        while True:
            response = await self._get_client().get("/v2/audio/voices", params=params)
            if response.status_code >= 400:
                raise _http_error(response)
            page = response.json()
            entries += page.get("data") or []
            token = page.get("next_page_token")
            if not token:
                return entries
            params["page_token"] = token

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                timeout=http_timeout(self._config),
            )
        return self._client

    def _body(self, text: str, voice: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._config.model,
            "input": text,
            "voice_id": voice or self._config.voice,
            "response_format": "pcm",
            "stream": True,
        }
        if self._config.expressiveness is not None:
            body["expressiveness"] = self._config.expressiveness
        return body

    @contextlib.asynccontextmanager
    async def _render(self, text: str, voice: str | None) -> AsyncGenerator[httpx.Response, None]:
        """Open the streamed render of *text*; a refusal raises before any audio."""
        async with self._get_client().stream(
            "POST", "/v1/audio/speech", json=self._body(text, voice)
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise _http_error(response)
            yield response

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream *text* as 24 kHz 16-bit PCM; *context* is ignored (no context level).

        Raises:
            VoxtralTTSError: The service refused the request (an unknown voice
                is a 404), failed while rendering, or ended the render before
                its end (retryable).
        """
        carry = b""
        done = False
        async with (
            self._render(text, voice) as response,
            contextlib.aclosing(_events(response)) as events,
        ):
            async for event in events:
                kind = event.get("type")
                if kind == "speech.audio.done":
                    done = True
                    break
                if kind != "speech.audio.delta":
                    continue
                data = carry + base64.b64decode(event.get("audio_data") or "")
                cut = len(data) - len(data) % _FLOAT_BYTES
                data, carry = data[:cut], data[cut:]
                if data:
                    yield AudioChunk(data=pcm16(data), sample_rate=SAMPLE_RATE)
        if not done:
            raise VoxtralTTSError(
                "VoxtralTTS render ended before speech.audio.done: its audio is cut short",
                retryable=True,
            )
        yield AudioChunk(data=b"", sample_rate=SAMPLE_RATE, is_final=True)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Render *text* whole, as a 16-bit WAV data URL."""
        stream = self.synthesize_stream(text, voice=voice)
        return await collect_wav_content(stream, text=text, sample_rate=SAMPLE_RATE)

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def pcm16(float32: bytes) -> bytes:
    """32-bit float little-endian samples as 16-bit PCM, clipped to [-1, 1]."""
    samples = array("f")
    samples.frombytes(float32)
    if sys.byteorder == "big":
        samples.byteswap()
    out = array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in samples))
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


async def _events(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """The JSON payload of each ``data:`` line of a server-sent event stream.

    An event that reports an error raises: the stream has nothing more to give.
    """
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        event = json.loads(line[5:])
        if event.get("type") == "error" or event.get("object") == "error":
            raise VoxtralTTSError(
                f"VoxtralTTS render failed: {event.get('message') or 'unknown error'}",
                code=_code(event),
                error_type=event.get("type"),
            )
        yield event


def _voice_info(entry: dict[str, Any]) -> VoiceInfo:
    """One voices-endpoint entry: a preset by its slug, a saved voice by its id."""
    languages = [_bcp47(tag) for tag in entry.get("languages") or []]
    attributes = {key: str(entry[key]) for key in ("type", "kind", "age") if entry.get(key)}
    return VoiceInfo(
        id=entry.get("slug") or entry["id"],
        name=entry.get("name"),
        language=languages[0] if len(languages) == 1 else "multilingual" if languages else None,
        gender=entry.get("gender"),
        description=entry.get("description") or None,
        attributes=attributes,
    )


def _bcp47(tag: str) -> str:
    """Mistral's ``en_us`` / ``en_GB`` / ``fr`` as BCP-47: ``en-US``, ``en-GB``, ``fr``."""
    language, _, region = tag.replace("-", "_").partition("_")
    return f"{language.lower()}-{region.upper()}" if region else language.lower()


def _code(body: dict[str, Any]) -> str | None:
    code = body.get("code")
    return str(code) if code is not None else None


def _http_error(response: httpx.Response) -> VoxtralTTSError:
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
    status = response.status_code
    message = body.get("message") or detail or response.text[:200]
    return VoxtralTTSError(
        f"VoxtralTTS request failed ({status}): {message}",
        code=_code(body),
        error_type=body.get("type"),
        retryable=status == 429 or status >= 500,
        status_code=status,
    )
