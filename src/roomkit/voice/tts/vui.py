"""Vui Nano text-to-speech provider: replies decoded inside the conversation.

`Vui Nano <https://huggingface.co/fluxions/vui>`_ (fluxions.ai, Apache 2.0) is
a 305M-parameter TTS over the Qwen3-TTS-12Hz codec that generates each reply
inside the dialogue: the conversation so far, the user's audio included, lives
in its KV cache. The provider declares :attr:`TTSContextLevel.AUDIO` and keeps
that cache in step with the voice session's :class:`TTSContext` (RFC §12.2.2).

Constraints of the model and of ``vui-tts``: English only, Python 3.12, a CUDA
GPU for real-time streaming, and one active conversation per provider (a
single KV cache). Install with ``pip install roomkit[vui]``.

Two operations use private ``vui-tts`` attributes, the ones Vui's own server
uses: cutting the cache back to the middle of a turn after a barge-in, and
setting a preset voice's speaker token. The dependency is pinned
(``vui-tts>=1.1.4,<1.2``) until Vui exposes them.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import math
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts._vui_session import FRAME_MS, VuiConversation
from roomkit.voice.tts.audio_utils import wrap_wav
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.context import TTSContextLevel

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.vui")

SAMPLE_RATE = 24000
PRESET_VOICES = ("maeve", "abraham", "rhian", "harry")


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
        max_secs: Longest reply, in seconds; a longer text is cut off there.
    """

    voices: dict[str, VuiVoice] = field(default_factory=lambda: {"maeve": VuiVoice("maeve")})
    checkpoint: str = "vui-nano-1.1"
    temperature: float = 0.7
    max_secs: float = 30.0


class VuiTTSProvider(TTSProvider):
    """Vui Nano: each reply is generated inside the conversation it answers."""

    def __init__(self, config: VuiTTSConfig | None = None) -> None:
        self._config = config or VuiTTSConfig()
        if not self._config.voices:
            raise ValueError("VuiTTSConfig.voices needs at least one voice")
        self._cache: _VuiRow | None = None
        self._conversation: VuiConversation | None = None
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
        if self._conversation is not None:
            self._conversation.release(context_id)

    async def warmup(self) -> None:
        """Load the model, the codec and every voice prompt."""
        await asyncio.to_thread(self._load)

    def _load(self) -> VuiConversation:
        if self._conversation is None:
            self._cache = _VuiRow(self._config)
            self._conversation = VuiConversation(self._cache)
        return self._conversation

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
            conversation = await asyncio.to_thread(self._load)
            cancel = threading.Event()
            frames = conversation.speak(context, name, text, cancel)
            # aclosing: a barge-in closing this stream must stop the GPU thread
            # before the lock is released, not whenever the iterator is collected.
            async with contextlib.aclosing(_iterate_in_thread(frames, cancel)) as pcms:
                async for pcm in pcms:
                    yield AudioChunk(data=pcm, sample_rate=SAMPLE_RATE)
        yield AudioChunk(data=b"", sample_rate=SAMPLE_RATE, is_final=True)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize *text* on its own (no conversation) as a WAV data URL.

        The provider has one cache: this empties it, and the conversation it
        held restarts from the prompt at its next call.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        pcm = b"".join([chunk.data async for chunk in self.synthesize_stream(text, voice=voice)])
        wav = wrap_wav(pcm, SAMPLE_RATE)
        return AudioContentModel(
            url=f"data:audio/wav;base64,{base64.b64encode(wav).decode()}",
            mime_type="audio/wav",
            transcript=text,
            duration_seconds=len(pcm) / 2 / SAMPLE_RATE,
        )

    async def close(self) -> None:
        async with self._lock:
            if self._cache is not None:
                self._cache.close()
            self._cache = None
            self._conversation = None


async def _iterate_in_thread(
    frames: Generator[bytes, None, None], cancel: threading.Event
) -> AsyncGenerator[bytes, None]:
    """Drive a blocking GPU generator from a worker thread.

    Closing this iterator (a barge-in) sets *cancel* and waits for the thread
    to stop, even if the waiting task is cancelled again meanwhile, so the
    lock is never released while the GPU is still busy.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    done = object()

    def run() -> None:
        # The generator stops on *cancel* itself (Vui checks it every frame),
        # so it ends through its own cleanup rather than suspended mid-frame.
        try:
            for pcm in frames:
                loop.call_soon_threadsafe(queue.put_nowait, pcm)
        except BaseException as exc:  # handed to the consumer, raised there
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            frames.close()
            loop.call_soon_threadsafe(queue.put_nowait, done)

    worker = loop.run_in_executor(None, run)
    try:
        while (item := await queue.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()
        cancelled = False
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError


class _VuiRow:
    """The real Vui cache: one ``Engine(max_rows=1)`` row and its codec, on CUDA."""

    def __init__(self, config: VuiTTSConfig) -> None:
        try:
            import soundfile
            import torch
            from julius.resample import resample_frac
            from safetensors import safe_open
            from vui.engine import Engine, GenConfig, Segment
            from vui.prompt_files import hub_prompt, hub_prompt_transcript
            from vui.qwen_codec import QwenCodecEncoder
            from vui.qwen_spk_enc import QwenSpeakerEncoder
        except ImportError as exc:
            raise ImportError(
                "vui-tts is required for VuiTTSProvider (Python 3.12). "
                "Install it with: pip install roomkit[vui]"
            ) from exc
        self._torch = torch
        self._segment = Segment
        self._resample = resample_frac
        self._engine = Engine(config.checkpoint, max_rows=1)  # places everything on CUDA
        self._row = self._engine.new_row()
        self._encoder = QwenCodecEncoder.from_pretrained().cuda().float().eval()
        self._gen = GenConfig(temperature=config.temperature, max_secs=config.max_secs)
        self._reply_positions = math.ceil(config.max_secs * 1000 / FRAME_MS)
        self._prompts: dict[str, _Prompt] = {}
        for name, voice in config.voices.items():
            if voice.preset is not None:
                path = hub_prompt(voice.preset, config.checkpoint)
                with safe_open(path, "pt") as f:
                    names = set(f.keys())  # noqa: SIM118 (safe_open has no __contains__)
                    codes = f.get_tensor("codes")
                    token = f.get_tensor("spk_token_emb") if "spk_token_emb" in names else None
                self._prompts[name] = _Prompt(
                    text=hub_prompt_transcript(voice.preset, path),
                    codes=codes[:, : self._engine.Q].long().cuda(),
                    spk_token=token.cuda().to(self._engine.dtype) if token is not None else None,
                )
            else:
                data, rate = soundfile.read(voice.ref_audio, dtype="int16", always_2d=True)
                mono = data.mean(axis=1).astype("int16")
                spk_emb = None
                if getattr(self._engine.model, "spk_proj", None) is not None:
                    pcm = torch.from_numpy(mono).float() / 32768.0
                    wav24 = resample_frac(pcm.unsqueeze(0), rate, SAMPLE_RATE).squeeze(0)
                    spk_emb = QwenSpeakerEncoder.from_pretrained().embed(wav24[: 30 * SAMPLE_RATE])
                frame = AudioFrame(data=mono.tobytes(), sample_rate=rate, sample_width=2)
                self._prompts[name] = _Prompt(
                    text=voice.ref_text or "", codes=self.encode(frame), spk_emb=spk_emb
                )

    @property
    def offset(self) -> int:
        return int(self._row.offset)

    @property
    def capacity(self) -> int:
        return int(self._engine.max_seq)

    @property
    def reply_positions(self) -> int:
        return self._reply_positions

    def restart(self, voice: str) -> None:
        prompt = self._prompts[voice]
        self._row.reset()
        self._row.prefill([self._segment(prompt.text, prompt.codes)], spk_emb=prompt.spk_emb)
        if prompt.spk_token is not None:
            _set_speaker_token(self._row, prompt.spk_token)

    def reset(self) -> None:
        # Offset 0: the KV cache and the codec's rolling context are both emptied.
        self._row.reset()

    def truncate(self, offset: int) -> None:
        _truncate_kv(self._engine, self._row, offset)

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


@dataclass
class _Prompt:
    text: str
    codes: Any
    spk_emb: Any = None
    spk_token: Any = None


# The two private vui-tts accesses (see the module docstring, and RMK-197).


def _truncate_kv(engine: Any, row: Any, offset: int) -> None:
    """Move *row*'s KV position back to *offset*, mid-turn included."""
    engine._rewind_row(row, offset)


def _set_speaker_token(row: Any, token: Any) -> None:
    """Condition *row*'s agent turns on a preset's projected speaker token."""
    row._spk_token = token
