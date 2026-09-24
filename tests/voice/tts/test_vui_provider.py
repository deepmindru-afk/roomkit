"""VuiTTSProvider around a fake Vui row: threading, cancellation, voices."""

from __future__ import annotations

import importlib.util
import threading
import time
import types
from collections.abc import Iterator

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.tts import vui as vui_module
from roomkit.voice.tts.context import TTSContext, TTSContextLevel
from roomkit.voice.tts.vui import VuiTTSConfig, VuiTTSProvider, VuiVoice


class _SlowRow:
    """A stand-in for the GPU row: 20 frames, 10 ms each."""

    instances: list[_SlowRow] = []

    capacity = 100_000
    reply_positions = 375

    def __init__(self, config: VuiTTSConfig) -> None:
        self.offset = 0
        self.stopped_at: int | None = None
        self.closed = False
        _SlowRow.instances.append(self)

    def reset(self) -> None:
        self.offset = 0

    def restart(self, voice: str) -> None:
        self.offset = 100

    def truncate(self, offset: int) -> None:
        self.offset = offset

    def add_user(self, text: str, audio: AudioFrame | None) -> None:
        self.offset += 5

    def generate(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
        for i in range(20):
            if cancel.is_set():
                self.stopped_at = i
                return
            time.sleep(0.01)
            self.offset += 1
            yield b"\x01\x00" * 1920

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> VuiTTSProvider:
    _SlowRow.instances.clear()
    monkeypatch.setattr(vui_module, "_VuiRow", _SlowRow)
    return VuiTTSProvider(VuiTTSConfig(voices={"maeve": VuiVoice("maeve")}))


def _context() -> TTSContext:
    return TTSContext(context_id="s1", turns=(), next_turn_id="t1")


async def test_streams_every_frame_then_a_final_marker(provider: VuiTTSProvider) -> None:
    chunks = [c async for c in provider.synthesize_stream("hello", context=_context())]

    assert len(chunks) == 21
    assert chunks[0].sample_rate == 24000
    assert chunks[-1].is_final and chunks[-1].data == b""


async def test_closing_the_stream_stops_the_gpu_thread(provider: VuiTTSProvider) -> None:
    stream = provider.synthesize_stream("hello", context=_context())
    await anext(stream)
    await stream.aclose()  # barge-in

    [row] = _SlowRow.instances
    assert row.stopped_at is not None and row.stopped_at < 20


async def test_synthesize_returns_a_wav(provider: VuiTTSProvider) -> None:
    content = await provider.synthesize("hello")

    assert content.mime_type == "audio/wav"
    assert content.duration_seconds == pytest.approx(20 * 1920 / 24000)


async def test_a_second_cancel_while_waiting_keeps_the_lock_until_the_thread_stops(
    provider: VuiTTSProvider,
) -> None:
    import asyncio
    import contextlib

    async def consume() -> None:
        stream = provider.synthesize_stream("hello", context=_context())
        async with contextlib.aclosing(stream) as chunks:
            async for _ in chunks:
                await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.03)
    task.cancel()  # the barge-in: the stream closes, waiting for the GPU thread
    await asyncio.sleep(0.002)
    task.cancel()  # cancelled again while it waits
    with pytest.raises(asyncio.CancelledError):
        await task

    [row] = _SlowRow.instances
    assert row.stopped_at is not None  # the thread had stopped before the task ended
    assert not provider._lock.locked()


async def test_an_unknown_voice_is_refused(provider: VuiTTSProvider) -> None:
    with pytest.raises(ValueError, match="not found"):
        await anext(provider.synthesize_stream("hi", voice="nobody"))


async def test_close_closes_the_row(provider: VuiTTSProvider) -> None:
    await provider.warmup()
    await provider.close()

    assert _SlowRow.instances[0].closed


def test_level_is_audio() -> None:
    assert VuiTTSProvider().context_level == TTSContextLevel.AUDIO


def test_a_voice_is_a_preset_or_a_clip() -> None:
    with pytest.raises(ValueError):
        VuiVoice()
    with pytest.raises(ValueError):
        VuiVoice(preset="maeve", ref_audio="x.wav", ref_text="x")
    with pytest.raises(ValueError):
        VuiVoice(ref_audio="x.wav")
    with pytest.raises(ValueError, match="Unknown Vui preset"):
        VuiVoice(preset="nobody")


@pytest.mark.skipif(importlib.util.find_spec("vui") is not None, reason="vui-tts installed")
async def test_without_vui_tts_the_error_says_how_to_install() -> None:
    with pytest.raises(ImportError, match=r"roomkit\[vui\]"):
        await VuiTTSProvider().warmup()


def test_lazy_getters() -> None:
    from roomkit.voice import get_vui_tts_config, get_vui_tts_provider, get_vui_voice

    assert get_vui_tts_provider() is VuiTTSProvider
    assert get_vui_tts_config() is VuiTTSConfig
    assert get_vui_voice() is VuiVoice


def test_each_reply_reseeds_the_audio_decoder_from_a_short_tail() -> None:
    """RMK-199: before streaming, the decoder is seeded from the last second only."""
    order: list[object] = []

    class _Ctx:
        _buf: object = None

        def prefill(self, n_codebooks: int = 0, device: str = "cuda") -> None:
            order.append(("prefill", self._buf))
            assert n_codebooks == 0

    class _Buf:
        shape = (1, 16, 300)

        def __getitem__(self, key: object) -> str:
            return f"tail{key[2].start}"

    ctx = _Ctx()
    ctx._buf = _Buf()

    class _Row:
        _codec_ctx = ctx

        def stream(self, text, cfg, cancel, final_turn):
            order.append("stream")
            yield from ()

    row = object.__new__(vui_module._VuiRow)
    row._row = _Row()
    row._gen = types.SimpleNamespace(n_codebooks=0)
    row._torch = types.SimpleNamespace()

    list(row.generate("hi", threading.Event()))

    assert order == [("prefill", "tail-12"), "stream"]
    assert isinstance(ctx._buf, _Buf)  # the full buffer is back


def test_an_empty_codec_buffer_is_left_to_vui() -> None:
    """Right after a reset there is nothing to seed from: Vui starts cold itself."""
    seeded: list[bool] = []
    ctx = types.SimpleNamespace(_buf=None, prefill=lambda **kw: seeded.append(True))

    vui_module._reseed_decoder(types.SimpleNamespace(_codec_ctx=ctx), 12, 0)

    assert seeded == []
