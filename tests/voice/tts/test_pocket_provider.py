"""PocketTTSProvider around a fake model: threading, cancellation, voices, config."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
import threading
import time
import types
from collections.abc import Generator
from typing import Any

import pytest

from roomkit.voice.tts import pocket as pocket_module
from roomkit.voice.tts.pocket import PocketTTSConfig, PocketTTSProvider


class _SlowModel:
    """A stand-in for the loaded model: 20 chunks of 80 ms, 10 ms each."""

    instances: list[_SlowModel] = []

    def __init__(self, config: PocketTTSConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, str]] = []
        self.stopped_at: int | None = None
        self.running = 0
        self.max_running = 0
        _SlowModel.instances.append(self)

    def generate(self, text: str, voice: str, cancel: threading.Event) -> Generator[bytes]:
        self.calls.append((text, voice))
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            for i in range(20):
                if cancel.is_set():
                    self.stopped_at = i
                    return
                time.sleep(0.01)
                yield b"\x01\x00" * 1920
        finally:
            self.running -= 1


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> PocketTTSProvider:
    _SlowModel.instances.clear()
    monkeypatch.setattr(pocket_module, "_PocketModel", _SlowModel)
    config = PocketTTSConfig(language="french", voices={"estelle": "estelle", "moi": "moi.wav"})
    return PocketTTSProvider(config)


async def test_streams_every_chunk_then_a_final_marker(provider: PocketTTSProvider) -> None:
    chunks = [c async for c in provider.synthesize_stream("Bonjour")]

    assert len(chunks) == 21
    assert all(c.sample_rate == 24000 for c in chunks)
    assert chunks[-1].is_final and chunks[-1].data == b""
    assert _SlowModel.instances[0].calls == [("Bonjour", "estelle")]


async def test_the_voice_argument_picks_a_configured_voice(provider: PocketTTSProvider) -> None:
    [_ async for _ in provider.synthesize_stream("Salut", voice="moi")]

    assert _SlowModel.instances[0].calls == [("Salut", "moi")]


async def test_closing_the_stream_stops_the_model_thread(provider: PocketTTSProvider) -> None:
    stream = provider.synthesize_stream("Bonjour")
    await anext(stream)
    await stream.aclose()  # barge-in

    [model] = _SlowModel.instances
    assert model.stopped_at is not None and model.stopped_at < 20
    assert not provider._lock.locked()


async def test_concurrent_replies_never_share_the_model(provider: PocketTTSProvider) -> None:
    async def speak(text: str) -> int:
        return len([c async for c in provider.synthesize_stream(text)])

    counts = await asyncio.gather(speak("un"), speak("deux"))

    assert counts == [21, 21]
    assert _SlowModel.instances[0].max_running == 1  # one model, loaded once


async def test_a_cancelled_consumer_waits_for_the_thread_before_unlocking(
    provider: PocketTTSProvider,
) -> None:
    async def consume() -> None:
        async with contextlib.aclosing(provider.synthesize_stream("Bonjour")) as chunks:
            async for _ in chunks:
                await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    [model] = _SlowModel.instances
    assert model.stopped_at is not None
    assert not provider._lock.locked()


async def test_blank_text_yields_only_the_final_marker(provider: PocketTTSProvider) -> None:
    chunks = [c async for c in provider.synthesize_stream("  \n")]

    assert [c.is_final for c in chunks] == [True]
    assert _SlowModel.instances == []  # the model is not even loaded


async def test_synthesize_returns_a_wav(provider: PocketTTSProvider) -> None:
    content = await provider.synthesize("Bonjour")

    assert content.mime_type == "audio/wav"
    assert content.url.startswith("data:audio/wav;base64,")
    assert content.duration_seconds == pytest.approx(20 * 1920 / 24000)


async def test_an_unknown_voice_is_refused(provider: PocketTTSProvider) -> None:
    with pytest.raises(ValueError, match="not found"):
        await anext(provider.synthesize_stream("Bonjour", voice="nobody"))


async def test_warmup_loads_once_and_close_drops_the_model(provider: PocketTTSProvider) -> None:
    await provider.warmup()
    await provider.warmup()
    assert len(_SlowModel.instances) == 1

    await provider.close()
    [_ async for _ in provider.synthesize_stream("Bonjour")]
    assert len(_SlowModel.instances) == 2


def test_default_voice_is_the_first_one(provider: PocketTTSProvider) -> None:
    assert provider.default_voice == "estelle"
    assert PocketTTSProvider().default_voice == "alba"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"voices": {}}, "at least one voice"),
        ({"device": "mps"}, "CUDA device"),
        ({"device": "cuda", "quantize": True}, "only works on CPU"),
    ],
)
def test_invalid_configs_are_refused(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PocketTTSConfig(**kwargs)


def test_a_gpu_config_is_accepted() -> None:
    assert PocketTTSConfig(device="cuda:1").device == "cuda:1"


class _FakeTTSModel:
    loaded: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.sample_rate = 24000
        self.device = "cpu"
        self.prompts: list[str] = []

    @classmethod
    def load_model(cls, **kwargs: Any) -> _FakeTTSModel:
        cls.loaded.append(kwargs)
        return cls()

    def to(self, device: str) -> _FakeTTSModel:
        self.device = device
        return self

    def get_state_for_audio_prompt(self, source: str) -> str:
        self.prompts.append(source)
        return f"state:{source}"


@pytest.fixture
def fake_pocket_tts(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTTSModel]:
    _FakeTTSModel.loaded.clear()
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    module = types.ModuleType("pocket_tts")
    module.TTSModel = _FakeTTSModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pocket_tts", module)
    return _FakeTTSModel


def test_the_model_loads_the_language_on_the_device(fake_pocket_tts: type[_FakeTTSModel]) -> None:
    config = PocketTTSConfig(
        language="french", voices={"e": "estelle", "m": "moi.wav"}, device="cuda", temperature=0.5
    )

    model = pocket_module._PocketModel(config)

    assert fake_pocket_tts.loaded == [{"language": "french", "temp": 0.5, "quantize": False}]
    assert model._model.device == "cuda"
    assert model._states == {"e": "state:estelle", "m": "state:moi.wav"}


def test_the_cpu_model_stays_on_cpu(fake_pocket_tts: type[_FakeTTSModel]) -> None:
    model = pocket_module._PocketModel(PocketTTSConfig(quantize=True))

    assert fake_pocket_tts.loaded[0]["quantize"] is True
    assert model._model.device == "cpu"


@pytest.mark.skipif(importlib.util.find_spec("pocket_tts") is not None, reason="installed")
async def test_without_pocket_tts_the_error_says_how_to_install() -> None:
    with pytest.raises(ImportError, match=r"roomkit\[pocket-tts\]"):
        await PocketTTSProvider().warmup()


def test_lazy_getters() -> None:
    from roomkit.voice import get_pocket_tts_config, get_pocket_tts_provider

    assert get_pocket_tts_provider() is PocketTTSProvider
    assert get_pocket_tts_config() is PocketTTSConfig
