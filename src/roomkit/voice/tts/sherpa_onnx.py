"""sherpa-onnx text-to-speech provider: VITS/Piper and Kokoro models."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import struct
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.audio_utils import wrap_wav as _wrap_wav
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger(__name__)


_MODEL_TYPES = ("vits", "kokoro")


@dataclass
class SherpaOnnxTTSConfig:
    """Configuration for the sherpa-onnx TTS provider.

    Attributes:
        model: Path to the ``.onnx`` model.
        tokens: Path to ``tokens.txt``.
        data_dir: Path to espeak-ng data directory (Piper and Kokoro models).
        lexicon: Path to optional lexicon file. A multi-lang Kokoro takes
            several, comma-separated (``lexicon-us-en.txt,lexicon-zh.txt``).
        speaker_id: Speaker ID for multi-speaker models; for Kokoro, the
            index of a voice in ``voices.bin`` (``ff_siwis`` is 30 in v1.0).
        speed: Speech speed multiplier (1.0 = normal).
        sample_rate: The rate a caller sets its output up for (Kokoro: 24000).
            Audio always carries the model's own rate.
        num_threads: Number of CPU threads for inference.
        provider: ONNX execution provider (``"cpu"`` or ``"cuda"``).
        model_type: ``"vits"`` (VITS/Piper) or ``"kokoro"``.
        voices: Path to Kokoro's ``voices.bin``, required with ``"kokoro"``.
        lang: Language espeak-ng reads the text in for a multi-lang Kokoro
            (``"fr"``, ``"es"``, ``"it"``...). Empty, any text that is not
            Chinese is read as English: French comes out mispronounced.
    """

    model: str = ""
    tokens: str = ""
    data_dir: str = ""
    lexicon: str = ""
    speaker_id: int = 0
    speed: float = 1.0
    sample_rate: int = 22050
    num_threads: int = 2
    provider: str = "cpu"
    model_type: str = "vits"
    voices: str = ""
    lang: str = ""

    def __post_init__(self) -> None:
        if self.model_type not in _MODEL_TYPES:
            raise ValueError(
                f"Unknown sherpa-onnx TTS model_type {self.model_type!r}; expected one of "
                f"{_MODEL_TYPES}"
            )
        if self.model_type == "kokoro" and not self.voices:
            raise ValueError("A Kokoro model needs its voices.bin in SherpaOnnxTTSConfig.voices")


def _float32_to_pcm_s16le(samples: list[float]) -> bytes:
    """Convert float32 samples in [-1, 1] to PCM signed 16-bit LE bytes."""
    clamped = [max(-1.0, min(1.0, s)) for s in samples]
    int_samples = [int(s * 32767) for s in clamped]
    return struct.pack(f"<{len(int_samples)}h", *int_samples)


_PARAGRAPH_RE = re.compile(r"\n\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_text(text: str, max_chars: int = 300) -> list[str]:
    """Split text into sentence-sized chunks for chunked TTS generation.

    Splits on paragraph breaks (\\n\\n) first, then on sentence boundaries
    (.!?) within each paragraph. Merges short sentence fragments to avoid
    unnatural pauses. Breaks long chunks that exceed *max_chars* on the
    nearest line break or whitespace.
    """
    text = text.strip()
    if not text:
        return []

    # Split on paragraph breaks.
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]

    # Split each paragraph on sentence boundaries, merging short bits.
    chunks: list[str] = []
    for para in paragraphs:
        sentences = [s.strip() for s in _SENTENCE_RE.split(para) if s.strip()]
        # Merge short consecutive fragments (< 40 chars)
        merged: list[str] = []
        for sent in sentences:
            if merged and len(merged[-1]) < 40:
                merged[-1] = merged[-1] + " " + sent
            else:
                merged.append(sent)
        chunks.extend(merged)

    # Break any chunk that still exceeds max_chars.
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
            continue
        # Try splitting on line breaks first
        lines = chunk.split("\n")
        buf = ""
        for line in lines:
            candidate = (buf + "\n" + line).strip() if buf else line
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                # If a single line still exceeds max_chars, split on whitespace
                if len(line) > max_chars:
                    words = line.split()
                    buf = ""
                    for w in words:
                        candidate = (buf + " " + w) if buf else w
                        if len(candidate) <= max_chars:
                            buf = candidate
                        else:
                            if buf:
                                result.append(buf)
                            buf = w
                else:
                    buf = line
        if buf:
            result.append(buf)

    return result


class SherpaOnnxTTSProvider(TTSProvider):
    """Text-to-speech provider using sherpa-onnx with VITS/Piper or Kokoro models."""

    def __init__(self, config: SherpaOnnxTTSConfig) -> None:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is required for SherpaOnnxTTSProvider. "
                "Install it with: pip install roomkit[sherpa-onnx]"
            ) from exc
        self._config = config
        self._sherpa = __import__("sherpa_onnx")
        self._tts: Any = None
        # Two first calls at once must not each load the model.
        self._load_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "SherpaOnnxTTS"

    @property
    def default_voice(self) -> str:
        return str(self._config.speaker_id)

    def _get_tts(self) -> Any:
        """Lazily create the OfflineTts engine, once. Blocking: call it off the loop."""
        with self._load_lock:
            if self._tts is None:
                self._tts = self._load_tts()
        return self._tts

    async def _tts_engine(self) -> Any:
        """The engine, loaded in a worker thread on first use.

        Loading reads the model from disk (Kokoro is ~126 MB): on the event
        loop it would stall every session until it is done.
        """
        if self._tts is not None:
            return self._tts
        return await asyncio.to_thread(self._get_tts)

    def _load_tts(self) -> Any:
        """Create the OfflineTts engine from the config."""
        cfg = self._config
        logger.info(
            "Loading TTS model: type=%s, model=%s, tokens=%s, data_dir=%s, "
            "lexicon=%s, provider=%s, num_threads=%d",
            cfg.model_type,
            cfg.model,
            cfg.tokens,
            cfg.data_dir,
            cfg.lexicon,
            cfg.provider,
            cfg.num_threads,
        )
        model_config = self._sherpa.OfflineTtsModelConfig(
            **self._model_kwargs(),
            num_threads=cfg.num_threads,
            provider=cfg.provider,
        )
        tts_config = self._sherpa.OfflineTtsConfig(model=model_config)
        tts = self._sherpa.OfflineTts(tts_config)
        logger.info(
            "TTS model loaded: sample_rate=%d, num_speakers=%d",
            tts.sample_rate,
            tts.num_speakers,
        )
        return tts

    def _model_kwargs(self) -> dict[str, Any]:
        """The model's own section of sherpa-onnx's ``OfflineTtsModelConfig``."""
        cfg = self._config
        if cfg.model_type == "kokoro":
            kokoro = self._sherpa.OfflineTtsKokoroModelConfig(
                model=cfg.model,
                voices=cfg.voices,
                tokens=cfg.tokens,
                data_dir=cfg.data_dir,
                lexicon=cfg.lexicon,
                lang=cfg.lang,
            )
            return {"kokoro": kokoro}
        vits = self._sherpa.OfflineTtsVitsModelConfig(
            model=cfg.model,
            tokens=cfg.tokens,
            data_dir=cfg.data_dir,
            lexicon=cfg.lexicon,
        )
        return {"vits": vits}

    async def warmup(self) -> None:
        """Pre-load the TTS model (CUDA init can be slow)."""
        await self._tts_engine()

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Speaker ID as string (uses default if not specified).

        Returns:
            AudioContent with a ``data:`` URL containing WAV audio.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        tts = await self._tts_engine()
        sid = int(voice) if voice is not None else self._config.speaker_id
        speed = self._config.speed
        chunks = _split_text(text)

        def _run() -> list[float]:
            all_samples: list[float] = []
            for chunk in chunks:
                audio = tts.generate(chunk, sid=sid, speed=speed)
                all_samples.extend(audio.samples)
            return all_samples

        t0 = time.monotonic()
        samples = await asyncio.to_thread(_run)
        synthesis_ms = (time.monotonic() - t0) * 1000
        sample_rate = tts.sample_rate

        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.tts.synthesis_ms",
            synthesis_ms,
            unit="ms",
            attributes={"provider": "sherpa_onnx"},
        )

        pcm_data = _float32_to_pcm_s16le(samples)
        wav_data = _wrap_wav(pcm_data, sample_rate)
        data_url = f"data:audio/wav;base64,{base64.b64encode(wav_data).decode()}"

        duration = len(samples) / sample_rate if sample_rate else None

        return AudioContentModel(
            url=data_url,
            mime_type="audio/wav",
            transcript=text,
            duration_seconds=duration,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks using a callback bridge.

        Args:
            text: Text to synthesize.
            voice: Speaker ID as string (uses default if not specified).

        Yields:
            AudioChunk with PCM S16LE audio data.

        Closing the stream early (a barge-in) stops the synthesis at the
        sentence under way; a synthesis error is raised to the reader.
        """
        tts = await self._tts_engine()
        sid = int(voice) if voice is not None else self._config.speaker_id
        speed = self._config.speed
        # The rate the model speaks at, which config.sample_rate may not match
        # (Kokoro speaks 24 kHz, a Piper model 16, 22.05 or 44.1 kHz).
        sample_rate = tts.sample_rate
        text_chunks = _split_text(text)

        queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # Set once the reader is gone: nothing more is synthesized for it.
        stopped = threading.Event()

        def _callback(samples: list[float], progress: float) -> int:
            """Called by sherpa-onnx from the executor thread."""
            if stopped.is_set():
                return 0  # sherpa-onnx: 0 stops the generation, 1 goes on
            pcm = _float32_to_pcm_s16le(samples)
            chunk = AudioChunk(
                data=pcm,
                sample_rate=sample_rate,
                format="pcm_s16le",
                is_final=False,
            )
            loop.call_soon_threadsafe(queue.put_nowait, chunk)
            return 1  # continue generation

        def _run() -> None:
            try:
                for tc in text_chunks:
                    if stopped.is_set():
                        return
                    tts.generate(tc, sid=sid, speed=speed, callback=_callback)
            finally:
                # Also on a failed generate(): the reader wakes, and the
                # error reaches it through ``await task``.
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        task = loop.run_in_executor(None, _run)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            stopped.set()
            await task

        # Yield final marker
        yield AudioChunk(
            data=b"",
            sample_rate=sample_rate,
            format="pcm_s16le",
            is_final=True,
        )
