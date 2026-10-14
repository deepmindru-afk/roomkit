"""Fish Audio text-to-speech provider: Drama 3 and the S2 family.

Fish Audio renders text posted to ``POST /v1/tts`` and streams text sent over
``wss://api.fish.audio/v1/tts/live``; a ``model`` header picks the model.
Drama 3 (``drama-3``, in preview since 2026-10-13) is directed in the text
itself: ``[calm, then annoyed]`` before the words it colours, sounds such as
``[sigh]``, paired tags such as ``<whisper>…</whisper>``. Those are performed
only in expressive mode (:attr:`FishAudioTTSConfig.expressive`); otherwise the
text is spoken as written. Control tokens (``<|speaker:1|>``) in a text are
always neutralised: the provider writes the speaker tokens of a dialogue
itself.

Audio is mono 16-bit PCM at the configured rate. Install with
``pip install roomkit[fish-audio]``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from roomkit.providers.ai.base import ProviderError
from roomkit.providers.utils import http_timeout
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts._fish_audio_text import dialogue_text, prepare, says_words
from roomkit.voice.tts._fish_audio_ws import FishAudioSocketError, stream_audio
from roomkit.voice.tts.audio_utils import collect_wav_content, whole_samples
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.voices import (
    DialogueTurn,
    VoiceInfo,
    check_dialogue,
    dialogue_transcript,
    filter_voices,
)

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.fish_audio")

MODELS = ("drama-3", "s2.1-pro", "s2.1-pro-free", "s2-pro")
"""The models the provider asks for. The API bills any other name as
``s2.1-pro`` and says nothing (measured by the balance, 2026-10-14), so an
unknown name is refused here rather than sent."""

SAMPLE_RATES = (8000, 16000, 24000, 32000, 44100)
"""The PCM rates the API renders; it answers 400 to any other."""

LATENCIES = ("low", "balanced", "normal")
"""``normal`` holds every byte until the whole clip is generated."""

MAX_DIALOGUE_SPEAKERS = 4
"""Measured 2026-10-14: four voices render on both families; Drama 3 runs away
at five (41.7 s for five two-word lines) and six are refused, the limit being
the total length of the voices' reference audio."""

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_PAGE_SIZE = 100


class FishAudioTTSError(ProviderError):
    """A render or a listing Fish Audio refused or failed."""

    def __init__(self, message: str, *, status_code: int | None) -> None:
        super().__init__(
            message,
            retryable=status_code is not None and (status_code == 429 or status_code >= 500),
            provider="FishAudioTTS",
            status_code=status_code,
        )


@dataclass
class FishAudioTTSConfig:
    """Configuration for :class:`FishAudioTTSProvider`.

    Attributes:
        api_key: A Fish Audio API key. API credit is separate from the
            platform's: an account without it is refused (402).
        model: One of :data:`MODELS`.
        voice: A voice model id (``reference_id``) from :meth:`list_voices`
            or the Fish Audio site. ``None`` lets Fish pick its default voice.
        expressive: Perform the directions and tags written in the text
            (``[whispering]``, ``<whisper>…</whisper>``). Off, the text is
            spoken as written: brackets are read as parentheses and markup is
            read out. Do not combine it with
            :class:`~roomkit.voice.tts.filters.StripBrackets`, which removes
            the directions before they arrive.
        sample_rate: Output rate, one of :data:`SAMPLE_RATES`.
        latency: One of :data:`LATENCIES`.
        speed: Speaking rate, 0.5 to 2.0 (``prosody.speed``). ``None`` keeps
            the model's own pace; Drama 3 speaks 40 to 60 % slower than
            S2.1-Pro on the same text (measured 2026-10-14).
        base_url: The HTTP API.
        ws_url: The streaming-input socket.
        timeout: Read budget of a render, in seconds.
        connect_timeout: TCP connect budget, in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    model: str = "drama-3"
    voice: str | None = None
    expressive: bool = False
    sample_rate: int = 24000
    latency: str = "balanced"
    speed: float | None = None
    base_url: str = "https://api.fish.audio"
    ws_url: str = "wss://api.fish.audio/v1/tts/live"
    timeout: float = 60.0
    connect_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("FishAudioTTSConfig.api_key is required")
        if not self.api_key.isascii():
            raise ValueError(
                "FishAudioTTSConfig.api_key holds a non-ASCII character (a placeholder "
                "such as '…' copied as is?): an HTTP header cannot carry it"
            )
        if self.model not in MODELS:
            raise ValueError(
                f"model must be one of {', '.join(MODELS)}, got {self.model!r}: Fish Audio "
                "would render any other name with s2.1-pro without saying so"
            )
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {self.sample_rate}")
        if self.latency not in LATENCIES:
            raise ValueError(f"latency must be one of {LATENCIES}, got {self.latency!r}")
        if self.speed is not None and not 0.5 <= self.speed <= 2.0:
            raise ValueError(f"speed must be between 0.5 and 2.0, got {self.speed}")
        _check_secure(self.base_url, "base_url", secure="https", plain="http")
        _check_secure(self.ws_url, "ws_url", secure="wss", plain="ws")


def _check_secure(url: str, name: str, *, secure: str, plain: str) -> None:
    parsed = httpx.URL(url)
    if parsed.scheme != secure and not (parsed.scheme == plain and parsed.host in _LOOPBACK_HOSTS):
        raise ValueError(
            f"{name} must use {secure}: a plain connection would send the key in clear"
        )


class FishAudioTTSProvider(TTSProvider):
    """Fish Audio voices: rendered whole, streamed, fed sentence by sentence, or in dialogue."""

    def __init__(self, config: FishAudioTTSConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "FishAudioTTS"

    @property
    def default_voice(self) -> str | None:
        return self._config.voice

    @property
    def max_dialogue_speakers(self) -> int:
        return MAX_DIALOGUE_SPEAKERS

    @property
    def supports_streaming_input(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "model": self._config.model,
            "User-Agent": "roomkit",
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=self._headers(),
                timeout=http_timeout(self._config),
            )
        return self._client

    def _request(self, text: str, reference_id: str | list[str] | None) -> dict[str, Any]:
        """The render settings of one request, the text and voices given."""
        body: dict[str, Any] = {
            "text": text,
            "format": "pcm",
            "sample_rate": self._config.sample_rate,
            "latency": self._config.latency,
        }
        if reference_id is not None:
            body["reference_id"] = reference_id
        if self._config.speed is not None:
            body["prosody"] = {"speed": self._config.speed}
        return body

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream *text* as PCM at the configured rate; *context* is ignored.

        A text with no word to say (blank, punctuation, or directions alone)
        makes no request: Fish invents speech for one.

        Raises:
            FishAudioTTSError: Fish Audio refused or failed the render.
        """
        rate = self._config.sample_rate
        prepared = prepare(text, expressive=self._config.expressive)
        if says_words(prepared):
            body = self._request(prepared, voice or self._config.voice)
            # Closed with this generator: a listener that stops early ends the request.
            async with contextlib.aclosing(self._render(body)) as render:
                async for pcm in render:
                    yield AudioChunk(data=pcm, sample_rate=rate)
        yield AudioChunk(data=b"", sample_rate=rate, is_final=True)

    async def _render(self, body: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """Whole 16-bit samples of one render, as they arrive (the odd byte waits)."""
        carry = b""
        async with self._get_client().stream("POST", "/v1/tts", json=body) as response:
            if response.status_code >= 400:
                raise _refusal("render", response, await response.aread())
            async for data in response.aiter_bytes():
                pcm, carry = whole_samples(carry + data)
                if pcm:
                    yield pcm

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Render *text* whole, as a WAV data URL."""
        stream = self.synthesize_stream(text, voice=voice)
        return await collect_wav_content(stream, text=text, sample_rate=self._config.sample_rate)

    async def synthesize_dialogue(
        self, turns: Sequence[DialogueTurn], voices: Mapping[str, str]
    ) -> AudioContent:
        """Voice a scripted exchange in one render (RFC §12.2).

        Each turn opens with its speaker's token and, when it has one, its
        ``style`` as a direction (``[style]``); a token written in a turn's
        text is neutralised, so a turn cannot speak in another voice.

        Raises:
            ValueError: A turn names an unmapped speaker, has no word to say,
                or the turns name more than :data:`MAX_DIALOGUE_SPEAKERS`
                speakers.
            FishAudioTTSError: Fish Audio refused or failed the render (400
                ``Reference audio too long`` when the voices' references are
                too long together).
        """
        speakers = check_dialogue(
            turns, voices, max_speakers=self.max_dialogue_speakers, provider=self.name
        )
        rate = self._config.sample_rate
        text = dialogue_text(turns, speakers, expressive=self._config.expressive)
        body = self._request(text, [voices[speaker] for speaker in speakers])
        async with contextlib.aclosing(self._render(body)) as render:
            chunks = (AudioChunk(data=pcm, sample_rate=rate) async for pcm in render)
            return await collect_wav_content(
                chunks, text=dialogue_transcript(turns), sample_rate=rate
            )

    async def synthesize_stream_input(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Speak sentences as they come, over one socket; *context* is ignored.

        Each item of *text_stream* is sent with a flush, as a Voice Channel
        hands in whole sentences. An item with no word to say waits for the
        next one and goes with it; one left at the end is not sent.

        Raises:
            FishAudioTTSError: The socket was refused (402 without API credit),
                the audio finished with an error, or the socket closed before
                the end of the audio.
        """
        rate = self._config.sample_rate
        audio = stream_audio(
            self._config.ws_url,
            headers=self._headers(),
            request=self._request("", voice or self._config.voice),
            text_stream=text_stream,
            expressive=self._config.expressive,
        )
        carry = b""
        try:
            async for data in audio:
                pcm, carry = whole_samples(carry + data)
                if pcm:
                    yield AudioChunk(data=pcm, sample_rate=rate)
        except FishAudioSocketError as exc:
            raise FishAudioTTSError(str(exc), status_code=exc.status_code) from exc
        finally:
            await audio.aclose()
        yield AudioChunk(data=b"", sample_rate=rate, is_final=True)

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """The account's own voices, then the public ones, best rated first.

        The public catalog holds thousands of voices: one page of
        :data:`_PAGE_SIZE` is read, narrowed server side by the language's
        primary subtag and by *query* on the voice name (a public voice whose
        description alone matches is not found). Fish tags voices with a
        language and no region, so ``fr-CA`` matches none of them while
        ``fr`` does.
        """
        own = await self._voice_models({"self": "true", "page_size": _PAGE_SIZE})
        params: dict[str, Any] = {"page_size": _PAGE_SIZE, "sort_by": "score"}
        if language:
            params["language"] = language.split("-")[0].lower()
        if query:
            params["title"] = query
        listed = {entry["_id"] for entry in own}
        public = [e for e in await self._voice_models(params) if e["_id"] not in listed]
        voices = [_voice_info(entry) for entry in own + public]
        return filter_voices(voices, language=language, gender=gender, query=query)

    async def _voice_models(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        response = await self._get_client().get("/model", params=params)
        if response.status_code >= 400:
            raise _refusal("voice listing", response, response.content)
        return list(response.json()["items"])

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _voice_info(entry: dict[str, Any]) -> VoiceInfo:
    """A Fish Audio voice model as a :class:`VoiceInfo`."""
    languages = entry.get("languages") or []
    tags = [str(tag) for tag in entry.get("tags") or []]
    lowered = {tag.lower() for tag in tags}
    gender = "female" if "female" in lowered else "male" if "male" in lowered else None
    attributes = {"visibility": str(entry.get("visibility", "")), "tags": ", ".join(tags)}
    author = (entry.get("author") or {}).get("nickname")
    if author:
        attributes["author"] = str(author)
    return VoiceInfo(
        id=entry["_id"],
        name=entry.get("title") or entry["_id"],
        language=languages[0] if len(languages) == 1 else "multilingual" if languages else None,
        gender=gender,
        description=entry.get("description") or None,
        attributes=attributes,
    )


def _refusal(what: str, response: httpx.Response, body: bytes) -> FishAudioTTSError:
    """The error of a refused call, with Fish's error code and message."""
    detail = body.decode(errors="replace")[:200] or response.reason_phrase
    code = response.headers.get("x-fish-error-code")
    return FishAudioTTSError(
        f"Fish Audio {what} failed ({response.status_code}"
        f"{f', {code}' if code else ''}): {detail}",
        status_code=response.status_code,
    )
