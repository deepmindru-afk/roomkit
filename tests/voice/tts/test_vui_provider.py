"""VuiTTSProvider around a fake Vui row: threading, cancellation, voices."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
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
    audio_capacity = 4500
    prompt_frames = 0

    def __init__(self, config: VuiTTSConfig) -> None:
        self.offset = 0
        self.stopped_at: int | None = None
        self.closed = False
        self.fail_close = False
        self.reset_gate: threading.Event | None = None  # reset() waits on it when set
        self.threads: list[tuple[str, threading.Thread]] = []  # (call, thread it ran on)
        self.texts: list[str] = []  # what each reply was asked to say
        self._ran("load")
        _SlowRow.instances.append(self)

    def _ran(self, call: str) -> None:
        self.threads.append((call, threading.current_thread()))

    def reset(self) -> None:
        self._ran("reset")
        if self.reset_gate is not None:
            self.reset_gate.wait(5)
        self.offset = 0

    def restart(self, voice: str) -> None:
        self._ran("restart")
        self.offset = 100

    def truncate(self, offset: int) -> None:
        self._ran("truncate")
        self.offset = offset

    def add_user(self, text: str, audio: AudioFrame | None) -> None:
        self._ran("add_user")
        self.offset += 5

    def generate(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
        self._ran("generate")
        self.texts.append(text)
        for i in range(20):
            if cancel.is_set():
                self.stopped_at = i
                return
            time.sleep(0.01)
            self.offset += 1
            yield b"\x01\x00" * 1920

    def close(self) -> None:
        self._ran("close")
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")


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


class TestVuiThread:
    """Every vui-tts call runs on one thread of its own, per loaded engine (RMK-371)."""

    async def test_every_call_runs_on_the_providers_own_thread(
        self, provider: VuiTTSProvider
    ) -> None:
        await provider.warmup()
        async for _ in provider.synthesize_stream("hi", context=_context()):
            pass
        provider.release_context("s1")
        await provider.close()

        calls = _SlowRow.instances[0].threads
        assert [call for call, _ in calls] == ["load", "restart", "generate", "reset", "close"]
        [thread] = {thread for _, thread in calls}
        assert thread.name.startswith("roomkit-vui")
        assert thread is not threading.main_thread()
        assert not thread.is_alive()  # stopped with the engine

    async def test_a_reloaded_engine_starts_on_a_fresh_thread(
        self, provider: VuiTTSProvider
    ) -> None:
        await provider.warmup()
        await provider.close()
        await provider.warmup()

        first, second = (row.threads[0][1] for row in _SlowRow.instances)
        assert second is not first
        assert second.is_alive()
        await provider.close()

    async def test_a_failing_close_still_stops_the_thread_and_unloads(
        self, provider: VuiTTSProvider
    ) -> None:
        await provider.warmup()
        row = _SlowRow.instances[0]
        row.fail_close = True

        with pytest.raises(RuntimeError, match="close failed"):
            await provider.close()

        assert not row.threads[0][1].is_alive()
        await provider.warmup()
        assert len(_SlowRow.instances) == 2  # loaded again, not the failed engine
        await provider.close()

    async def test_a_cancelled_close_still_unloads_before_the_thread_stops(
        self, provider: VuiTTSProvider
    ) -> None:
        async for _ in provider.synthesize_stream("hi", context=_context()):
            pass
        row = _SlowRow.instances[0]
        row.reset_gate = threading.Event()
        provider.release_context("s1")  # holds the Vui thread until the gate opens
        closing = asyncio.create_task(provider.close())
        await asyncio.sleep(0.05)

        closing.cancel()
        row.reset_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await closing

        assert row.closed
        assert not row.threads[0][1].is_alive()

    def test_a_release_before_any_load_does_nothing(self, provider: VuiTTSProvider) -> None:
        provider.release_context("s1")

        assert _SlowRow.instances == []


async def test_a_reply_reaches_vui_as_prose(provider: VuiTTSProvider) -> None:
    # Vui invents syllables at a line break (RMK-400).
    poem = "Roses are red,\nViolets are blue\n\nSugar is sweet"
    async for _ in provider.synthesize_stream(poem, context=_context()):
        pass

    assert _SlowRow.instances[0].texts == ["Roses are red, Violets are blue. Sugar is sweet."]


class TestAsProse:
    """The text Vui is given: no line breaks, no trailing clause mark (RMK-400)."""

    def test_the_session_poem_reads_as_running_sentences(self) -> None:
        poem = (
            "Here's a little poem about Quebec: \n\n"
            "Beneath the snow-draped peaks of Quebec,  \n"
            "Where rivers hum and stories creep,  \n"
            "The old stone buildings stand in grace.  \n\n"
            "Hope you like it!"
        )
        assert vui_module._as_prose(poem) == (  # noqa: SLF001
            "Here's a little poem about Quebec: Beneath the snow-draped peaks of Quebec, "
            "Where rivers hum and stories creep, The old stone buildings stand in grace. "
            "Hope you like it!"
        )

    def test_a_line_without_punctuation_takes_a_comma_or_ends_its_paragraph(self) -> None:
        text = "First line\nsecond line\n\nNew paragraph"
        assert vui_module._as_prose(text) == (  # noqa: SLF001
            "First line, second line. New paragraph."
        )

    def test_a_text_ending_on_a_comma_ends_on_a_full_stop(self) -> None:
        # Ending on a comma, Vui ran on in 3 takes of 3.
        assert vui_module._as_prose("the weather in Montreal today,") == (  # noqa: SLF001
            "the weather in Montreal today."
        )

    def test_one_sentence_is_left_as_it_is(self) -> None:
        text = "Glad to hear that! [laugh] What have you been up to lately?"
        assert vui_module._as_prose(text) == text  # noqa: SLF001

    def test_nothing_to_say_stays_nothing(self) -> None:
        assert vui_module._as_prose("") == ""  # noqa: SLF001
        assert vui_module._as_prose(" \n — \n") == ""  # noqa: SLF001


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


class _FakeRow:
    """The public ``vui.engine.Row`` calls, recorded; no ``_codec_ctx`` to reach into."""

    def __init__(self, log: list[tuple[object, ...]]) -> None:
        self.log = log

    def reset(self) -> None:
        self.log.append(("reset",))

    def prefill(self, segments: list[object], spk_emb: object = None, *, cond_bias=None) -> None:
        self.log.append(("prefill", segments, spk_emb, cond_bias))

    def truncate(self, offset: int) -> None:
        self.log.append(("truncate", offset))

    def stream(self, text: str, cfg: object, cancel: threading.Event, final_turn: bool):
        self.log.append(("stream", text))
        yield from ()


class _FakeEngine:
    checkpoint = "vui-nano-1.1.safetensors"

    def __init__(self, log: list[tuple[object, ...]]) -> None:
        self.log = log

    def set_conditioning(self) -> None:
        self.log.append(("set_conditioning",))


_CODES = types.SimpleNamespace(shape=(50, 16))


def _vui_row(**prompts: vui_module._Prompt) -> tuple[vui_module._VuiRow, list[tuple]]:
    """A ``_VuiRow`` over a fake engine and row, without loading anything."""
    log: list[tuple[object, ...]] = []
    row = object.__new__(vui_module._VuiRow)
    row._engine = _FakeEngine(log)
    row._row = _FakeRow(log)
    row._segment = lambda text, codes: (text, codes)
    row._gen = types.SimpleNamespace()
    row._torch = types.SimpleNamespace()
    row._prompts = prompts
    return row, log


class TestVuiRow:
    def test_a_preset_prompt_is_the_one_baked_for_the_engine_checkpoint(self) -> None:
        row, _ = _vui_row()
        calls: list[tuple[str, object]] = []

        def official_prompt(voice: str, checkpoint: object = None) -> tuple:
            calls.append((voice, checkpoint))
            return "transcript", _CODES, "token", "bias"

        row._official_prompt = official_prompt

        prompt = row._load_prompt(VuiVoice("rhian"))

        assert calls == [("rhian", "vui-nano-1.1.safetensors")]
        assert prompt == vui_module._Prompt("transcript", _CODES, "token", "bias")

    def test_a_preset_is_prefilled_with_its_speaker_token_and_bias(self) -> None:
        row, log = _vui_row(maeve=vui_module._Prompt("hi", _CODES, "token", "bias"))

        row.restart("maeve")

        assert log == [("reset",), ("prefill", [("hi", _CODES)], "token", "bias")]
        assert row.prompt_frames == 50

    def test_a_cloned_voice_after_a_preset_drops_the_preset_bias(self) -> None:
        row, log = _vui_row(
            maeve=vui_module._Prompt("hi", _CODES, "token", "bias"),
            clone=vui_module._Prompt("me", _CODES, "embedding"),
        )
        row.restart("maeve")
        log.clear()

        row.restart("clone")

        assert log == [
            ("reset",),
            ("set_conditioning",),
            ("prefill", [("me", _CODES)], "embedding", None),
        ]

    def test_a_cut_reply_goes_through_row_truncate(self) -> None:
        row, log = _vui_row()

        row.truncate(120)

        assert log == [("truncate", 120)]

    def test_a_reply_leaves_the_audio_decoder_to_vui(self) -> None:
        """vui-tts 1.2 keeps the decoder on its 10 s grid (RMK-199): no re-seed."""
        row, log = _vui_row()

        list(row.generate("hi", threading.Event()))

        assert log == [("stream", "hi")]


class TestVuiPublicApi:
    def test_the_public_api_the_provider_calls_exists(self) -> None:
        """Fails when vui-tts drops or reshapes what the provider calls (RMK-197)."""
        engine_mod = pytest.importorskip("vui.engine")
        prompts_mod = pytest.importorskip("vui.prompt_files")

        prefill = inspect.signature(engine_mod.Row.prefill).parameters
        assert "spk_emb" in prefill
        assert prefill["cond_bias"].kind is inspect.Parameter.KEYWORD_ONLY
        assert "offset" in inspect.signature(engine_mod.Row.truncate).parameters
        assert {"text", "codes"} <= inspect.signature(engine_mod.Row.add_user).parameters.keys()
        stream = list(inspect.signature(engine_mod.Row.stream).parameters)
        assert stream[1:4] == ["text", "cfg", "cancel"]
        assert "final_turn" in stream
        assert callable(engine_mod.Engine.set_conditioning)
        assert isinstance(engine_mod.Engine.checkpoint, property)
        assert "checkpoint" in inspect.signature(prompts_mod.load_official_prompt).parameters
