"""PhononTTSProvider around a fake ptts runtime: config, streaming, barge-in, voices."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest

from roomkit.voice.tts.phonon import PhononTTSConfig, PhononTTSProvider

CHUNKS = 20
CHUNK_SAMPLES = 1920


class _FakeStream:
    """A ptts AudioStream: CHUNKS float32 chunks of 80 ms, ``pace`` seconds
    each to generate (none unless a test needs a stream that lasts)."""

    pace = 0.0

    def __init__(self, tts: _FakeTTS, samples: np.ndarray) -> None:
        self._tts = tts
        self._samples = samples
        self.sent = 0
        self.closed = False

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def __iter__(self) -> _FakeStream:
        return self

    def __next__(self) -> np.ndarray:
        if self.closed or self.sent == CHUNKS:
            raise StopIteration
        time.sleep(self.pace)
        self.sent += 1
        return self._samples

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with self._tts.lock:
                self._tts.running -= 1


class _FakeTTS:
    instances: list[_FakeTTS] = []
    samples = np.full(CHUNK_SAMPLES, 0.5, dtype=np.float32)

    def __init__(self, model: str | None, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        self.sample_rate = 24000
        self.voices = ["default"]
        self.added: dict[str, str] = {}
        self.streams: list[tuple[str, str | None, float | None, _FakeStream]] = []
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        _FakeTTS.instances.append(self)

    def add_voice(self, name: str, path: str) -> None:
        self.added[name] = path
        self.voices.append(name)

    def stream(
        self, text: str, *, voice: str | None = None, temperature: float | None = None
    ) -> _FakeStream:
        if voice is not None and voice not in self.voices + ["Noémie"]:
            raise LookupError(f"no voice '{voice}' is registered")
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        stream = _FakeStream(self, self.samples)
        self.streams.append((text, voice, temperature, stream))
        return stream


_CATALOG = [
    {"voice_id": "fr1", "name": "Noémie", "language": "fr", "description": "Warm"},
    {"voice_id": "en1", "name": "Zoey", "language": "en", "description": "Upbeat"},
]


@pytest.fixture
def fake_ptts(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _FakeTTS.instances.clear()
    module = types.ModuleType("ptts")
    module.TTS = _FakeTTS  # type: ignore[attr-defined]
    module.DEFAULT_MODEL = "hf://gradium/phonon-31m-en@rev"  # type: ignore[attr-defined]
    module.catalog_calls = 0  # type: ignore[attr-defined]

    def catalog_voices() -> list[dict[str, str | None]]:
        module.catalog_calls += 1  # type: ignore[attr-defined]
        return _CATALOG

    module.catalog_voices = catalog_voices  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ptts", module)
    return module


@pytest.fixture
def provider(fake_ptts: types.ModuleType) -> PhononTTSProvider:
    return PhononTTSProvider(PhononTTSConfig(model="hf://gradium/phonon-40m-fr", language="fr"))


async def test_streams_every_chunk_as_pcm16_then_a_final_marker(
    provider: PhononTTSProvider,
) -> None:
    chunks = [c async for c in provider.synthesize_stream("Bonjour")]

    assert len(chunks) == CHUNKS + 1
    assert all(c.sample_rate == 24000 for c in chunks)
    assert chunks[0].data == np.full(CHUNK_SAMPLES, 16383, dtype="<i2").tobytes()
    assert chunks[-1].is_final and chunks[-1].data == b""
    [tts] = _FakeTTS.instances
    assert [(text, voice) for text, voice, _, _ in tts.streams] == [("Bonjour", None)]


async def test_samples_beyond_full_scale_are_clipped(
    provider: PhononTTSProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_FakeTTS, "samples", np.array([2.0, -2.0, 0.0], dtype=np.float32))

    chunk = await anext(provider.synthesize_stream("Bonjour"))

    assert np.frombuffer(chunk.data, dtype="<i2").tolist() == [32767, -32767, 0]


async def test_the_config_reaches_the_runtime(fake_ptts: types.ModuleType) -> None:
    config = PhononTTSConfig(
        model="hf://gradium/phonon-40m-fr",
        language="fr",
        voice="Noémie",
        voices={"brand": "brand.safetensors"},
        quant="q4k",
        temperature=0.5,
        duration_delta=-0.3,
    )
    provider = PhononTTSProvider(config)

    await provider.warmup()
    [_ async for _ in provider.synthesize_stream("Bonjour")]

    [tts] = _FakeTTS.instances
    assert tts.model == "hf://gradium/phonon-40m-fr"
    assert tts.kwargs == {"quant": "q4k", "lang": "fr", "conditions": {"duration_delta": -0.3}}
    assert tts.added == {"brand": "brand.safetensors"}
    prefetch, reply = tts.streams
    # The default voice was fetched at load, by a stream closed at once.
    assert prefetch[1] == "Noémie" and prefetch[3].closed and prefetch[3].sent == 0
    assert reply[:3] == ("Bonjour", "Noémie", 0.5)


async def test_defaults_load_the_runtime_default_in_q8(fake_ptts: types.ModuleType) -> None:
    await PhononTTSProvider().warmup()

    [tts] = _FakeTTS.instances
    assert tts.model is None
    assert tts.kwargs == {"quant": "q8", "lang": "en", "conditions": None}
    assert tts.streams == []  # no default voice to fetch


async def test_a_default_voice_file_is_not_fetched(fake_ptts: types.ModuleType) -> None:
    config = PhononTTSConfig(voice="brand", voices={"brand": "brand.safetensors"})

    await PhononTTSProvider(config).warmup()

    assert _FakeTTS.instances[0].streams == []  # registered from the file already


async def test_the_voice_argument_overrides_the_default(fake_ptts: types.ModuleType) -> None:
    config = PhononTTSConfig(voice="Noémie", voices={"brand": "brand.safetensors"})
    provider = PhononTTSProvider(config)

    [_ async for _ in provider.synthesize_stream("Hello", voice="brand")]

    assert _FakeTTS.instances[0].streams[-1][1] == "brand"
    assert provider.default_voice == "Noémie"


async def test_closing_the_stream_stops_the_runtime_stream(
    provider: PhononTTSProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_FakeStream, "pace", 0.005)  # still generating at the barge-in
    stream = provider.synthesize_stream("Bonjour")
    await anext(stream)
    await stream.aclose()  # barge-in

    [tts] = _FakeTTS.instances
    [(_, _, _, runtime)] = tts.streams
    assert runtime.closed and runtime.sent < CHUNKS
    assert tts.running == 0


async def test_concurrent_replies_share_one_loaded_model(
    provider: PhononTTSProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_FakeStream, "pace", 0.005)  # long enough to overlap

    async def speak(text: str) -> int:
        return len([c async for c in provider.synthesize_stream(text)])

    counts = await asyncio.gather(speak("un"), speak("deux"))

    assert counts == [CHUNKS + 1, CHUNKS + 1]
    [tts] = _FakeTTS.instances
    assert tts.max_running == 2  # streamed side by side, not one after the other


async def test_an_unknown_voice_raises_to_the_consumer(provider: PhononTTSProvider) -> None:
    with pytest.raises(LookupError, match="nobody"):
        await anext(provider.synthesize_stream("Bonjour", voice="nobody"))


async def test_blank_text_yields_only_the_final_marker(provider: PhononTTSProvider) -> None:
    chunks = [c async for c in provider.synthesize_stream("  \n")]

    assert [c.is_final for c in chunks] == [True]
    assert _FakeTTS.instances == []  # the model is not even loaded


async def test_synthesize_returns_a_wav(provider: PhononTTSProvider) -> None:
    content = await provider.synthesize("Bonjour")

    assert content.mime_type == "audio/wav"
    assert content.url.startswith("data:audio/wav;base64,")
    assert content.duration_seconds == pytest.approx(CHUNKS * CHUNK_SAMPLES / 24000)


async def test_list_voices_gives_the_voice_files_then_the_catalog(
    fake_ptts: types.ModuleType,
) -> None:
    provider = PhononTTSProvider(PhononTTSConfig(voices={"brand": "brand.safetensors"}))

    voices = await provider.list_voices()
    french = await provider.list_voices(language="fr")

    assert [v.id for v in voices] == ["brand", "fr1", "en1"]
    assert [(v.id, v.name, v.description) for v in french] == [("fr1", "Noémie", "Warm")]
    assert fake_ptts.catalog_calls == 1  # fetched once, then kept
    assert _FakeTTS.instances == []  # listing loads no model


async def test_a_model_at_another_rate_is_refused(
    provider: PhononTTSProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _FakeTTS.__init__

    def at_16k(self: _FakeTTS, model: str | None, **kwargs: Any) -> None:
        original(self, model, **kwargs)
        self.sample_rate = 16000

    monkeypatch.setattr(_FakeTTS, "__init__", at_16k)

    with pytest.raises(ValueError, match="16000 Hz"):
        await provider.warmup()


async def test_warmup_loads_once_and_close_drops_the_model(provider: PhononTTSProvider) -> None:
    await provider.warmup()
    await provider.warmup()
    assert len(_FakeTTS.instances) == 1

    await provider.close()
    [_ async for _ in provider.synthesize_stream("Bonjour")]
    assert len(_FakeTTS.instances) == 2


@pytest.mark.skipif(importlib.util.find_spec("ptts") is not None, reason="installed")
async def test_without_ptts_the_error_says_how_to_install() -> None:
    with pytest.raises(ImportError, match=r"roomkit\[phonon\]"):
        await PhononTTSProvider().warmup()


def test_lazy_getters() -> None:
    from roomkit.voice import get_phonon_tts_config, get_phonon_tts_provider

    assert get_phonon_tts_provider() is PhononTTSProvider
    assert get_phonon_tts_config() is PhononTTSConfig
