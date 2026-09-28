"""Custom Gemini voices: designed from a description, or replicated with consent.

Behaviour measured against the live API on 2026-09-27:

* A designed voice must be stored: ``store=False`` is answered
  ``Prompted voice creation requires store=true``, so this library refuses it
  before the call (RFC §12.2.4). Creation takes about 15 s and returns a WAV
  preview; the voice expires after a year.
* A stored voice's ``voice_…`` id is accepted as ``voice`` by
  :class:`~roomkit.voice.tts.gemini.GeminiTTSProvider`, and comes first in its
  ``list_voices()``, typed ``prompted`` or ``replicated``.
* ``get`` and ``delete`` answer 404 for an id Google does not hold.

A replicated voice is Google's to verify: the consent recording travels with
the sample, and Google checks that the consenting speaker is the one in the
sample and that they read its consent statement word for word
(:data:`CONSENT_STATEMENTS`). Per Google's documentation both recordings are
16-bit, 24 kHz mono WAV, the sample 10 to 30 s of one adult speaker, and an
unstored replicated voice is a ``voicekey_…`` good for seven days. A refused
consent comes back as a 500 wrapping Google's ``Consent flow failed`` (seen
2026-09-27); it is raised as :class:`VoiceConsentError` with Google's reason.
The recordings are sent for the call and nothing of them is kept or logged
(RFC §17.6).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roomkit.models.event import AudioContent
from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.providers.gemini.voices import voice_info_from_catalog
from roomkit.voice.tts.library import CustomVoice, VoiceAudio, VoiceConsentError, VoiceLibrary

logger = logging.getLogger("roomkit.voice.tts.gemini_library")

REPLICATION_SAMPLE_RATE = 24000
"""The only rate Google accepts for replication audio."""

CONSENT_STATEMENTS: dict[str, str] = {
    "fr-CA": (
        "Je suis le propriétaire de cette voix et j'autorise Google à utiliser cette voix "
        "pour créer un modèle de voix synthétique."
    ),
    "fr-FR": (
        "Je suis le propriétaire de cette voix et j'autorise Google à utiliser cette voix "
        "pour créer un modèle de voix synthétique."
    ),
    "en-US": (
        "I am the owner of this voice and I consent to Google using this voice to create "
        "a synthetic voice model."
    ),
}
"""What the voice's owner reads, word for word, in the consent recording.

Copied from Google's voice-replication guide on 2026-09-27, which lists the
statement in 30 locales; these are the three RoomKit's users asked for. A
recording that departs from the text is refused."""

_CONSENT_REFUSED = "Consent flow failed"


@dataclass
class GeminiVoiceLibraryConfig:
    """Configuration for :class:`GeminiVoiceLibrary`.

    Args:
        api_key: Gemini API key (``GEMINI_API_KEY``).
        timeout: Per-request timeout in seconds. Designing a voice took about
            15 s on 2026-09-27.
        connect_timeout: TCP connect timeout in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    timeout: float = 120.0
    connect_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(self.connect_timeout) or self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be a positive finite number")


class GeminiVoiceLibrary(VoiceLibrary):
    """Custom voices on the Gemini API's voice store (RFC §12.2.4)."""

    def __init__(self, config: GeminiVoiceLibraryConfig) -> None:
        self._config = config
        self._client: Any = None
        self._http: Any = None

    @property
    def name(self) -> str:
        return "GeminiVoiceLibrary"

    @property
    def supports_design(self) -> bool:
        return True

    @property
    def supports_replication(self) -> bool:
        return True

    def _get_client(self) -> Any:
        if self._client is None:
            built = build_genai_client(
                self._config, provider="GeminiVoiceLibrary", api_key=self._config.api_key
            )
            self._client, self._http = built.client, built.http
        return self._client

    async def design_voice(
        self, description: str, *, store: bool = True, name: str | None = None
    ) -> CustomVoice:
        """A voice written from *description*, stored for the account.

        Raises:
            ValueError: A blank description, or ``store=False``: Google stores
                every designed voice, and returning a stored one to a caller
                who asked for none would leave a voice they do not know of.
        """
        if not description.strip():
            raise ValueError("description must not be empty")
        if not store:
            raise ValueError(
                "Google stores every designed voice: design_voice needs store=True "
                "(delete it with delete_voice when done)"
            )
        voice: dict[str, Any] = {"type": "prompted", "prompted": {"input": description}}
        if name:
            voice["display_name"] = name
        created = await self._get_client().aio.voices.create(voice=voice, store=True)
        logger.info("Designed voice %s created (stored)", created.id)
        return _custom_voice(created)

    async def replicate_voice(
        self,
        sample: VoiceAudio,
        consent: VoiceAudio,
        *,
        store: bool = True,
        name: str | None = None,
    ) -> CustomVoice:
        """A person's voice from *sample*, with their recorded *consent*.

        Both recordings are 16-bit, 24 kHz mono WAV. Google checks that the
        consent is spoken by the person in the sample and refuses the voice
        otherwise; that refusal is raised, and no voice is returned.

        Raises:
            ValueError: A recording that is not 16-bit 24 kHz mono WAV.
        """
        source = await _replication_wav(sample, "sample")
        consent_wav = await _replication_wav(consent, "consent")
        voice: dict[str, Any] = {
            "type": "replicated",
            "replicated": {
                "source_audio": {"data": _b64(source), "mime_type": "audio/wav"},
                "consent_audio": {"data": _b64(consent_wav), "mime_type": "audio/wav"},
            },
        }
        if name:
            voice["display_name"] = name
        try:
            created = await self._get_client().aio.voices.create(voice=voice, store=store)
        except Exception as exc:
            text = str(exc)
            if _CONSENT_REFUSED not in text:
                raise
            raise VoiceConsentError(
                f"Google refused the consent: {_consent_reason(text)}"
            ) from exc
        # The evidence an integrator keeps (RFC §17.6): what was made, and that
        # Google verified the consent. Never the recordings.
        logger.info(
            "Replicated voice %s created (stored=%s); consent verified by Google",
            created.id or "(unstored key)",
            store,
        )
        return _custom_voice(created)

    async def get_voice(self, voice_id: str) -> CustomVoice | None:
        try:
            voice = await self._get_client().aio.voices.get(voice_id)
        except Exception as exc:
            if _status(exc) == 404:
                return None
            raise
        return _custom_voice(voice)

    async def delete_voice(self, voice_id: str) -> None:
        try:
            await self._get_client().aio.voices.delete(voice_id)
        except Exception as exc:
            if _status(exc) != 404:
                raise
            return
        logger.info("Custom voice %s deleted", voice_id)

    async def close(self) -> None:
        """Close the genai client's connection pool and drop the reference."""
        client, self._client = self._client, None
        http, self._http = self._http, None
        await close_genai_client(client, http)


def _custom_voice(voice: Any) -> CustomVoice:
    info = voice_info_from_catalog(voice)
    return CustomVoice(
        voice=info,
        stored=bool(getattr(voice, "id", None)),
        expires_at=getattr(voice, "expire_time", None),
        sample=_sample(getattr(voice, "sample_audio", None)),
    )


def _sample(audio: Any) -> AudioContent | None:
    data = getattr(audio, "data", None)
    if not data:
        return None
    raw = data if isinstance(data, bytes | bytearray) else base64.b64decode(data)
    mime = getattr(audio, "mime_type", None) or "audio/wav"
    return AudioContent(url=f"data:{mime};base64,{base64.b64encode(raw).decode()}", mime_type=mime)


def _consent_reason(text: str) -> str:
    """Google's own sentence on why the consent failed, out of its error dump."""
    start = text.find(_CONSENT_REFUSED)
    reason = text[start:].split("\\n", 2)
    detail = reason[1] if len(reason) > 1 else reason[0]
    # The dump is a repr: its quotes arrive escaped.
    detail = detail.split("[type.googleapis", 1)[0].split('\\"', 1)[0].replace("\\'", "'")
    return detail.strip().rstrip(".") + "."


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def _replication_wav(audio: VoiceAudio, label: str) -> bytes:
    """The WAV bytes of *audio*, checked against what Google replicates from."""
    if isinstance(audio, Path):
        wav = await asyncio.to_thread(audio.read_bytes)
    elif isinstance(audio, bytes):
        wav = audio
    else:
        url = audio.url
        if not url.startswith("data:"):
            raise ValueError(f"{label} must carry its audio as a data: URL, not {url[:30]!r}")
        wav = base64.b64decode(url.split(",", 1)[1])
    try:
        with wave.open(io.BytesIO(wav), "rb") as reader:
            shape = (reader.getframerate(), reader.getnchannels(), reader.getsampwidth())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{label} is not a readable WAV file: {exc}") from exc
    if shape != (REPLICATION_SAMPLE_RATE, 1, 2):
        rate, channels, width = shape
        raise ValueError(
            f"{label} must be 16-bit {REPLICATION_SAMPLE_RATE} Hz mono WAV, "
            f"got {width * 8}-bit {rate} Hz with {channels} channel(s)"
        )
    return wav
