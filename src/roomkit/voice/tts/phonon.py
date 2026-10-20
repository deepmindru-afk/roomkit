"""Phonon text-to-speech provider: Gradium's small local models, on the CPU.

`Phonon <https://huggingface.co/collections/gradium/phonon>`_ (Gradium, built on
Pocket TTS and Kyutai's Mimi codec) streams 24 kHz speech from 31M- and
40M-parameter models through its own Rust runtime, ``ptts``: no torch, no GPU.
Measured on a desktop CPU with two threads, the q8 weights run about ten times
faster than real time and start speaking ~50 ms after they get a sentence.
Install with ``pip install roomkit[phonon]``.

A model speaks one language (``phonon-40m-fr``) or five
(``phonon-40m-multilingual``: English, French, German, Spanish, Portuguese).
Each model page on Hugging Face states its licence; the 40m models are gated:
accept their terms there, then log in (``hf auth login``) or set ``HF_TOKEN``.
``ptts.set_num_threads`` sizes the runtime's thread pool for the whole process,
before the first model loads.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts._thread_stream import iterate_in_thread
from roomkit.voice.tts.audio_utils import collect_wav_content
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.voices import VoiceInfo, filter_voices

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.phonon")

SAMPLE_RATE = 24000


@dataclass
class PhononTTSConfig:
    """Configuration for :class:`PhononTTSProvider`.

    Attributes:
        model: A Hugging Face repo as ``hf://owner/name[@revision]``
            (``hf://gradium/phonon-40m-fr``) or a local checkpoint directory;
            ``None`` loads the runtime's default, Phonon 31m English, pinned
            to one revision.
        language: How the text is spelled out before it is spoken (numbers,
            dates, symbols): ``en``, ``fr``, ``de``, ``es``, ``pt``, or
            ``none`` to speak it as written. Match the model's language.
        voice: Default voice: a name from the voice catalog
            (``gradium/phonon-voices``, any case, accents optional), its voice
            id, or a key of ``voices``; ``None`` uses the checkpoint's own.
        voices: Voice files (``.safetensors``, such as those Gradium Studio
            exports) by name, registered when the model loads.
        quant: Weight format; ``q8`` is the fastest on a CPU, ``None`` loads
            the full-precision weights.
        temperature: Sampling temperature; ``None`` keeps the runtime's (0.3).
        duration_delta: Speaking rate: negative speaks faster, positive
            slower (``-0.3``); ``None`` keeps the model's.
    """

    model: str | None = None
    language: str = "en"
    voice: str | None = None
    voices: dict[str, str] = field(default_factory=dict)
    quant: str | None = "q8"
    temperature: float | None = None
    duration_delta: float | None = None


class PhononTTSProvider(TTSProvider):
    """Gradium Phonon, streaming from a model held in this process."""

    def __init__(self, config: PhononTTSConfig | None = None) -> None:
        self._config = config or PhononTTSConfig()
        self._model: _PhononModel | None = None
        # Guards the load only: the runtime streams concurrent replies itself.
        self._load_lock = asyncio.Lock()
        self._catalog: list[VoiceInfo] | None = None

    @property
    def name(self) -> str:
        return "PhononTTS"

    @property
    def default_voice(self) -> str | None:
        return self._config.voice

    async def warmup(self) -> None:
        """Load the model, its voice files and the default voice."""
        await self._load()

    async def _load(self) -> _PhononModel:
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(_PhononModel, self._config)
            return self._model

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream *text* as 16-bit PCM chunks at 24 kHz, then a final marker.

        Raises:
            LookupError: *voice* is neither a voice file nor in the catalog.
        """
        if text.strip():
            model = await self._load()
            cancel = threading.Event()
            frames = model.generate(text, voice or self._config.voice, cancel)
            # aclosing: a barge-in closing this stream stops the runtime's workers.
            async with contextlib.aclosing(iterate_in_thread(frames, cancel)) as pcms:
                async for pcm in pcms:
                    yield AudioChunk(data=pcm, sample_rate=SAMPLE_RATE)
        yield AudioChunk(data=b"", sample_rate=SAMPLE_RATE, is_final=True)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize *text* as a WAV data URL."""
        stream = self.synthesize_stream(text, voice=voice)
        return await collect_wav_content(stream, text=text, sample_rate=SAMPLE_RATE)

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """The configured voice files, then the voice catalog (RFC §12.2).

        The catalog is fetched once and kept. It reports a name, an id, a
        language and a description for each voice, and no gender.
        """
        if self._catalog is None:
            self._catalog = await asyncio.to_thread(_catalog_voices)
        own = [VoiceInfo(id=name, name=name) for name in self._config.voices]
        return filter_voices(own + self._catalog, language=language, gender=gender, query=query)

    async def close(self) -> None:
        async with self._load_lock:
            self._model = None


def _import_ptts() -> Any:
    try:
        import ptts
    except ImportError as exc:
        raise ImportError(
            "ptts is required for PhononTTSProvider. Install it with: pip install roomkit[phonon]"
        ) from exc
    return ptts


def _catalog_voices() -> list[VoiceInfo]:
    return [
        VoiceInfo(
            id=voice["voice_id"],
            name=voice.get("name"),
            language=voice.get("language"),
            description=voice.get("description"),
        )
        for voice in _import_ptts().catalog_voices()
    ]


class _PhononModel:
    """The loaded runtime, with its voice files and the default voice fetched."""

    def __init__(self, config: PhononTTSConfig) -> None:
        ptts = _import_ptts()
        self._temperature = config.temperature
        logger.info("Loading Phonon %s (%s)", config.model or ptts.DEFAULT_MODEL, config.language)
        conditions = None
        if config.duration_delta is not None:
            conditions = {"duration_delta": config.duration_delta}
        self._tts = ptts.TTS(
            config.model, quant=config.quant, lang=config.language, conditions=conditions
        )
        if self._tts.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Phonon returned {self._tts.sample_rate} Hz, not 24000")
        for name, path in config.voices.items():
            self._tts.add_voice(name, path)
        if config.voice is not None and config.voice not in self._tts.voices:
            # A catalog voice downloads on first use: opening a stream fetches it.
            self._tts.stream("a", voice=config.voice).close()

    def generate(self, text: str, voice: str | None, cancel: threading.Event) -> Generator[bytes]:
        # Leaving the with block stops the runtime's workers (~70 ms).
        with self._tts.stream(text, voice=voice, temperature=self._temperature) as audio:
            for chunk in audio:
                if cancel.is_set():
                    return
                yield (chunk.clip(-1.0, 1.0) * 32767).astype("<i2").tobytes()
