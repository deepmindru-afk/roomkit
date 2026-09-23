"""VuiTTSProvider around a fake Vui row: threading, cancellation, voices."""

from __future__ import annotations

import importlib.util
import threading
import time
from collections.abc import Iterator

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.tts import vui as vui_module
from roomkit.voice.tts.context import TTSContext, TTSContextLevel
from roomkit.voice.tts.vui import VuiTTSConfig, VuiTTSProvider, VuiVoice


class _SlowRow:
    """A stand-in for the GPU row: 20 frames, 10 ms each."""

    instances: list[_SlowRow] = []

    def __init__(self, config: VuiTTSConfig) -> None:
        self.offset = 0
        self.stopped_at: int | None = None
        self.closed = False
        _SlowRow.instances.append(self)

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


@pytest.mark.skipif(importlib.util.find_spec("vui") is not None, reason="vui-tts installed")
async def test_without_vui_tts_the_error_says_how_to_install() -> None:
    with pytest.raises(ImportError, match=r"roomkit\[vui\]"):
        await VuiTTSProvider().warmup()


def test_lazy_getters() -> None:
    from roomkit.voice import get_vui_tts_config, get_vui_tts_provider, get_vui_voice

    assert get_vui_tts_provider() is VuiTTSProvider
    assert get_vui_tts_config() is VuiTTSConfig
    assert get_vui_voice() is VuiVoice
