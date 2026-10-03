"""Vui Nano text-to-speech provider: replies decoded inside the conversation.

`Vui Nano <https://huggingface.co/fluxions/vui>`_ (fluxions.ai, Apache 2.0) is
a 305M-parameter TTS over the Qwen3-TTS-12Hz codec that generates each reply
inside the dialogue: the conversation so far, the user's audio included, lives
in its KV cache. The provider declares :attr:`TTSContextLevel.AUDIO` and keeps
that cache in step with the voice session's :class:`TTSContext` (RFC §12.2.2).

Constraints of the model and of ``vui-tts``: English only, Python 3.12, a CUDA
GPU for real-time streaming, and one active conversation per provider (a
single KV cache). Install with ``pip install roomkit[vui]``.

It needs ``vui-tts`` 1.2: ``Row.truncate`` cuts the cache back to what the
user heard after a barge-in, and ``Row.prefill`` takes a preset's speaker token
and conditioning bias, as Vui's own server applies them.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import math
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts._thread_stream import iterate_in_thread
from roomkit.voice.tts._vui_session import FRAME_MS, VuiConversation
from roomkit.voice.tts.audio_utils import collect_wav_content
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.context import TTSContextLevel

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.vui")

SAMPLE_RATE = 24000
PRESET_VOICES = ("maeve", "abraham", "rhian", "harry")
# The tags Vui renders as sounds, as its prompting guide lists them
# (``docs/prompting.md``); any other bracketed word is read or garbled.
VUI_TAGS = ("breath", "laugh", "sigh", "gasp", "cough", "hesitate")

_SENTENCE_END = ".!?…"
_CLAUSE_END = ",;:—–"


def _as_prose(text: str) -> str:
    """*text* with its lines joined into running sentences, the only text Vui reads cleanly.

    Vui invents syllables at a line break and after a text that ends on a
    comma (measured on a 12-line poem: 15 % of the words wrong with its line
    breaks, 4 % joined; a text ending on a comma ran on in 3 takes of 3, one
    ending bare or on a full stop in none). A line without closing
    punctuation takes a comma, or a full stop at the end of its paragraph; a
    text ending on a clause mark ends on a full stop instead. The hosted Vui
    of fluxions.ai read the same poem cleanly with its line breaks.
    """
    lines: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        rows = [" ".join(row.split()) for row in paragraph.splitlines() if row.strip()]
        for i, row in enumerate(rows):
            if row[-1] not in _SENTENCE_END + _CLAUSE_END:
                row += "." if i == len(rows) - 1 else ","
            lines.append(row)
    prose = " ".join(lines)
    if prose and prose[-1] in _CLAUSE_END:
        prose = prose[:-1].rstrip()
        prose = prose + "." if prose else ""
    return prose


@dataclass(frozen=True)
class VuiVoice:
    """A voice prompt: a preset from the Hub, or a clip and its transcript.

    Attributes:
        preset: One of ``maeve``, ``abraham``, ``rhian``, ``harry``.
        ref_audio: Path to a clean speech clip (under ~15 s) to clone.
        ref_text: Exact transcript of ``ref_audio``.
    """

    preset: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None

    def __post_init__(self) -> None:
        if (self.preset is None) == (self.ref_audio is None):
            raise ValueError("a VuiVoice is either a preset or a ref_audio clip")
        if self.ref_audio is not None and not self.ref_text:
            raise ValueError("ref_audio needs its exact transcript in ref_text")
        if self.preset is not None and self.preset not in PRESET_VOICES:
            raise ValueError(f"Unknown Vui preset '{self.preset}'. Presets: {PRESET_VOICES}")


@dataclass
class VuiTTSConfig:
    """Configuration for :class:`VuiTTSProvider`.

    Attributes:
        voices: Named voices; the first one is the default.
        checkpoint: Vui checkpoint name or local path.
        temperature: Sampling temperature.
        max_secs: Longest reply, in seconds; a longer text is cut off there,
            with a warning. A minute holds most spoken replies, a short poem
            included.
    """

    voices: dict[str, VuiVoice] = field(default_factory=lambda: {"maeve": VuiVoice("maeve")})
    checkpoint: str = "vui-nano-1.1"
    temperature: float = 0.7
    max_secs: float = 60.0


class VuiTTSProvider(TTSProvider):
    """Vui Nano: each reply is generated inside the conversation it answers.

    Call :meth:`close` when done: it closes Vui on the provider's own thread
    and stops that thread. A provider dropped without it leaves torch's
    inference mode on in whichever thread collects it.
    """

    def __init__(self, config: VuiTTSConfig | None = None) -> None:
        self._config = config or VuiTTSConfig()
        if not self._config.voices:
            raise ValueError("VuiTTSConfig.voices needs at least one voice")
        self._cache: _VuiRow | None = None
        self._conversation: VuiConversation | None = None
        # Every vui-tts call runs on one thread, started with the engine and
        # joined by close(). Vui's codec keeps torch.inference_mode() entered
        # between calls, a thread-local state that must reach neither the event
        # loop nor the shared default executor, where an engine built later
        # fails on its first reply.
        self._thread: ThreadPoolExecutor | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "VuiTTS"

    @property
    def default_voice(self) -> str:
        return next(iter(self._config.voices))

    @property
    def context_level(self) -> TTSContextLevel:
        return TTSContextLevel.AUDIO

    def release_context(self, context_id: str) -> None:
        """Empty the cache of *context_id*'s dialogue, on the Vui thread, after any reply."""
        if self._thread is not None and self._conversation is not None:
            future = self._thread.submit(self._release, context_id)
            future.add_done_callback(functools.partial(_log_release_failure, context_id))

    def _release(self, context_id: str) -> None:
        if self._conversation is not None:
            self._conversation.release(context_id)

    async def warmup(self) -> None:
        """Load the model, the codec and every voice prompt."""
        async with self._lock:
            await self._on_vui_thread(self._load)

    def _vui_thread(self) -> ThreadPoolExecutor:
        if self._thread is None:
            self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="roomkit-vui")
        return self._thread

    async def _on_vui_thread(self, call: Callable[[], VuiConversation]) -> VuiConversation:
        return await asyncio.get_running_loop().run_in_executor(self._vui_thread(), call)

    def _load(self) -> VuiConversation:
        if self._conversation is None:
            self._cache = _VuiRow(self._config)
            self._conversation = VuiConversation(self._cache)
        return self._conversation

    def _unload(self) -> None:
        """Close the row and drop every Vui object, on the thread that made them."""
        cache, self._cache, self._conversation = self._cache, None, None
        if cache is not None:
            cache.close()

    def _voice_name(self, voice: str | None) -> str:
        name = voice or self.default_voice
        if name not in self._config.voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._config.voices)}")
        return name

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Speak *text* as the next turn of *context*'s conversation."""
        name = self._voice_name(voice)
        async with self._lock:
            conversation = await self._on_vui_thread(self._load)
            cancel = threading.Event()
            frames = conversation.speak(context, name, _as_prose(text), cancel)
            # aclosing: a barge-in closing this stream must stop the GPU thread
            # before the lock is released, not whenever the iterator is collected.
            pcms = iterate_in_thread(frames, cancel, executor=self._vui_thread())
            async with contextlib.aclosing(pcms):
                async for pcm in pcms:
                    yield AudioChunk(data=pcm, sample_rate=SAMPLE_RATE)
        yield AudioChunk(data=b"", sample_rate=SAMPLE_RATE, is_final=True)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize *text* on its own (no conversation) as a WAV data URL.

        The provider has one cache: this empties it, and the conversation it
        held restarts from the prompt at its next call.
        """
        stream = self.synthesize_stream(text, voice=voice)
        return await collect_wav_content(stream, text=text, sample_rate=SAMPLE_RATE)

    async def close(self) -> None:
        async with self._lock:
            thread, self._thread = self._thread, None
            if thread is None:
                return
            unload = asyncio.get_running_loop().run_in_executor(thread, self._unload)
            try:
                # Shielded: a cancelled close() still unloads before the thread stops.
                await asyncio.shield(unload)
            finally:
                await asyncio.to_thread(thread.shutdown)  # runs what is queued, then joins


class _VuiRow:
    """The real Vui cache: one ``Engine(max_rows=1)`` row and its codec, on CUDA."""

    def __init__(self, config: VuiTTSConfig) -> None:
        try:
            import soundfile
            import torch
            from julius.resample import resample_frac
            from vui.engine import Engine, GenConfig, Segment
            from vui.prompt_files import load_official_prompt
            from vui.qwen_codec import QwenCodecEncoder
            from vui.qwen_spk_enc import QwenSpeakerEncoder
        except ImportError as exc:
            raise ImportError(
                "vui-tts is required for VuiTTSProvider (Python 3.12). "
                "Install it with: pip install roomkit[vui]"
            ) from exc
        self._torch = torch
        self._soundfile = soundfile
        self._segment = Segment
        self._resample = resample_frac
        self._official_prompt = load_official_prompt
        self._speaker_encoder = QwenSpeakerEncoder
        self._engine = Engine(config.checkpoint, max_rows=1)  # places everything on CUDA
        self._row = self._engine.new_row()
        self._encoder = QwenCodecEncoder.from_pretrained().cuda().float().eval()
        self._gen = GenConfig(temperature=config.temperature, max_secs=config.max_secs)
        self._reply_positions = math.ceil(config.max_secs * 1000 / FRAME_MS)
        # The longest audio one training sequence held (360 s for vui-nano-1.1).
        trained_secs = float(self._engine.model.config.data.max_secs)
        self._audio_capacity = math.floor(trained_secs * 1000 / FRAME_MS)
        self._prompt_frames = 0
        self._prompts = {name: self._load_prompt(voice) for name, voice in config.voices.items()}

    def _load_prompt(self, voice: VuiVoice) -> _Prompt:
        """A voice's prompt: a preset baked for this checkpoint, or a clip encoded here."""
        if voice.preset is not None:
            # The speaker token and the bias fit only the checkpoint they were baked for.
            text, codes, token, bias = self._official_prompt(
                voice.preset, checkpoint=self._engine.checkpoint
            )
            return _Prompt(text=text, codes=codes, spk_emb=token, cond_bias=bias)
        data, rate = self._soundfile.read(voice.ref_audio, dtype="int16", always_2d=True)
        mono = data.mean(axis=1).astype("int16")
        spk_emb = None
        if getattr(self._engine.model, "spk_proj", None) is not None:
            pcm = self._torch.from_numpy(mono).float() / 32768.0
            wav24 = self._resample(pcm.unsqueeze(0), rate, SAMPLE_RATE).squeeze(0)
            spk_emb = self._speaker_encoder.from_pretrained().embed(wav24[: 30 * SAMPLE_RATE])
        frame = AudioFrame(data=mono.tobytes(), sample_rate=rate, sample_width=2)
        return _Prompt(text=voice.ref_text or "", codes=self.encode(frame), spk_emb=spk_emb)

    @property
    def offset(self) -> int:
        return int(self._row.offset)

    @property
    def capacity(self) -> int:
        return int(self._engine.max_seq)

    @property
    def reply_positions(self) -> int:
        return self._reply_positions

    @property
    def audio_capacity(self) -> int:
        return self._audio_capacity

    @property
    def prompt_frames(self) -> int:
        return self._prompt_frames

    def restart(self, voice: str) -> None:
        prompt = self._prompts[voice]
        self._prompt_frames = int(prompt.codes.shape[0])
        self._row.reset()
        if prompt.cond_bias is None:
            # The bias is engine-wide, and a prefill without one keeps the last voice's.
            self._engine.set_conditioning()
        self._row.prefill(
            [self._segment(prompt.text, prompt.codes)],
            spk_emb=prompt.spk_emb,
            cond_bias=prompt.cond_bias,
        )

    def reset(self) -> None:
        # Offset 0: the KV cache and the codec's rolling context are both emptied.
        self._row.reset()

    def truncate(self, offset: int) -> None:
        self._row.truncate(offset)

    def add_user(self, text: str, audio: AudioFrame | None) -> None:
        codes = self.encode(audio) if audio is not None else None
        self._row.add_user(text=text, codes=codes)

    def generate(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
        torch = self._torch
        for frame in self._row.stream(text, self._gen, cancel, final_turn=True):
            # The yielded tensor is a reused graph buffer: convert it now.
            samples = frame.detach().float().reshape(-1).clamp(-1.0, 1.0)
            yield (samples * 32767).to(torch.int16).cpu().numpy().tobytes()

    def encode(self, audio: AudioFrame) -> Any:
        """16-bit PCM at any rate -> codec codes (T, Q) on the GPU."""
        torch = self._torch
        pcm = torch.frombuffer(bytearray(audio.data), dtype=torch.int16).float() / 32768.0
        if audio.channels > 1:
            pcm = pcm.reshape(-1, audio.channels).mean(dim=1)
        wav = self._resample(pcm.unsqueeze(0), audio.sample_rate, SAMPLE_RATE)
        with torch.inference_mode():
            codes = self._encoder.encode(wav.reshape(1, 1, -1).cuda())
        return codes[0, : self._engine.Q].T.long()

    def close(self) -> None:
        self._row.close()


def _log_release_failure(context_id: str, future: Future[None]) -> None:
    if not future.cancelled() and (exc := future.exception()) is not None:
        logger.error("Vui: releasing context %s failed", context_id, exc_info=exc)


@dataclass
class _Prompt:
    text: str
    codes: Any
    spk_emb: Any = None  # a clip's speaker embedding, or a preset's projected token
    cond_bias: Any = None  # a preset's baked conditioning bias
