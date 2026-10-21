"""Tests for VoxtralSTTProvider (Mistral Voxtral Transcribe 3).

The streaming tests run the provider against a real WebSocket server on
localhost, scripted with the frames the live service sent on 2026-10-21, so
the close codes and the frame types go through the actual protocol stack.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import email
import email.policy
import io
import json
import logging
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from websockets.asyncio.server import ServerConnection, serve

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.event import AudioContent
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.voxtral import (
    VoxtralSTTConfig,
    VoxtralSTTError,
    VoxtralSTTProvider,
    VoxtralTurnDetection,
)
from roomkit.voice.stt.voxtral_protocol import language_code
from roomkit.voice.stt.voxtral_transcript import Transcript

_CREATED = {
    "type": "session.created",
    "session": {
        "request_id": "ws-test",
        "model": "voxtral-mini-transcribe-realtime-3-0-4",
        "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
        "target_streaming_delay_ms": None,
        "turn_detection": None,
    },
}


def _delta(text: str) -> dict[str, Any]:
    return {"type": "transcription.text.delta", "text": text}


def _done(text: str) -> dict[str, Any]:
    return {
        "type": "transcription.done",
        "model": "voxtral-mini-transcribe-realtime-3-0-4",
        "text": text,
        "language": None,
        "segments": [],
        "usage": {"prompt_audio_seconds": 7},
    }


_START = {"type": "turn.start", "time_s": 0.0}
_EAGER = {"type": "turn.eager_end", "time_s": 4.88}
_RESUME = {"type": "turn.resume", "time_s": 5.2}
_END = {"type": "turn.end", "time_s": 5.84, "end_reason": "threshold"}

# One turn as the live service sent it (2026-10-21), eager end and resume included.
_TURN = [
    _START,
    _delta("Hi"),
    _delta(","),
    _delta(" I"),
    _delta(" would"),
    _delta(" like"),
    _delta(" a"),
    _delta(" table"),
    _EAGER,
    _RESUME,
    _delta("."),
    _END,
]


def _replayed(capture: list[tuple[float, dict[str, Any]]], scale: float) -> list[Any]:
    """A capture's events, ``(seconds, event)`` as received, with its pauses scaled."""
    script: list[Any] = []
    previous = capture[0][0]
    for at, event in capture:
        script += [(at - previous) * scale, event]
        previous = at
    return script


# The live session's events after its first audio (2026-10-21, aggressive
# preset, 480 ms delay), eager ends left out.
_ITALY = _replayed(
    [
        (0.798, _START),
        (1.086, _delta("What")),
        (1.191, _delta(" is")),
        (1.283, _delta(" the")),
        (1.856, _delta(" capital")),
        (1.947, {"type": "turn.end", "time_s": 1.2, "end_reason": "threshold"}),
        (1.947, _delta(" of")),
        (2.272, {"type": "turn.start", "time_s": 1.44}),
        (2.272, _delta(" Italy")),
        (2.363, _delta("?")),
        (3.266, _delta(" Answer")),
        (3.363, _delta(" in")),
        (3.611, _delta(" one")),
        (4.022, _delta(" word")),
        (4.088, _delta(".")),
        (4.526, {"type": "turn.end", "time_s": 3.68, "end_reason": "threshold"}),
        (6.583, _done("What is the capital of Italy? Answer in one word.")),
    ],
    scale=0.2,
)

Handler = Callable[[ServerConnection], Awaitable[None]]
Script = list[dict[str, Any] | float]
"""Events to send, a float being a pause in seconds."""


class _Record:
    """What the scripted server received."""

    def __init__(self) -> None:
        self.update: dict[str, Any] = {}
        self.appends: list[str] = []
        self.controls: list[dict[str, Any]] = []
        self.path = ""
        self.authorization: str | None = None
        self.connections = 0

    @property
    def audio(self) -> bytes:
        return b"".join(base64.b64decode(piece) for piece in self.appends)


async def _play(ws: ServerConnection, script: Script) -> None:
    for item in script:
        if isinstance(item, float):
            await asyncio.sleep(item)
        else:
            await ws.send(json.dumps(item))


def _scripted(
    record: _Record,
    *,
    on_audio: Script = (),  # type: ignore[assignment]
    on_end: Script = (),  # type: ignore[assignment]
    updated: dict[str, Any] | None = None,
) -> Handler:
    """Open the session, take its settings, then play ``on_audio`` after the first
    append and ``on_end`` once the audio ends, then drop the line as the service
    does after ``transcription.done``."""

    async def handler(ws: ServerConnection) -> None:
        record.connections += 1
        record.path = ws.request.path if ws.request else ""
        record.authorization = ws.request.headers.get("Authorization") if ws.request else None
        await ws.send(json.dumps(_CREATED))
        record.update = json.loads(await ws.recv())
        await ws.send(json.dumps(updated or {"type": "session.updated", "session": {}}))
        played = False
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "input_audio.append":
                record.appends.append(event["audio"])
                if not played:
                    played = True
                    await _play(ws, list(on_audio))
                continue
            record.controls.append(event)
            if event["type"] == "input_audio.end":
                break
        await _play(ws, list(on_end))
        ws.transport.abort()

    return handler


@contextlib.asynccontextmanager
async def _server(handler: Handler, **kwargs: Any) -> AsyncIterator[str]:
    async with serve(handler, "127.0.0.1", 0, **kwargs) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


def _provider(url: str, **config: Any) -> VoxtralSTTProvider:
    config.setdefault("final_wait_ms", 50)
    return VoxtralSTTProvider(VoxtralSTTConfig(api_key="test-key", realtime_url=url, **config))


async def _chunks(*chunks: AudioChunk) -> AsyncIterator[AudioChunk]:
    for chunk in chunks:
        yield chunk


async def _drain(stream: AsyncIterator[TranscriptionResult]) -> list[TranscriptionResult]:
    """Every result of a stream, failing rather than hanging if it never ends."""

    async def read() -> list[TranscriptionResult]:
        return [result async for result in stream]

    return await asyncio.wait_for(read(), timeout=5)


async def _collect(
    provider: VoxtralSTTProvider, *chunks: AudioChunk, language: str | None = None
) -> list[TranscriptionResult]:
    return await _drain(provider.transcribe_stream(_chunks(*chunks), language=language))


def _finals(results: list[TranscriptionResult]) -> list[str]:
    return [r.text for r in results if r.is_final]


_PCM_16K = b"\x01\x00" * 1600  # 100 ms of 16 kHz mono
_SPEECH_START = TranscriptionResult(text="", is_final=False, is_speech_start=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_are_the_services(self) -> None:
        config = VoxtralSTTConfig(api_key="k")
        assert config.model == "voxtral-mini-transcribe-3"
        assert config.realtime_model == "voxtral-mini-transcribe-realtime-3"
        assert config.streaming_delay_ms == 480
        assert config.turn_detection == VoxtralTurnDetection(0.5, 0.5, 0.33, 500, 3000)
        assert config.stream_url == (
            "wss://api.mistral.ai/v1/audio/transcriptions/realtime"
            "?model=voxtral-mini-transcribe-realtime-3"
        )
        assert "api_key" not in repr(VoxtralSTTConfig(api_key="sk-secret"))

    @pytest.mark.parametrize("delay", [80, 480, 1200, 2400])
    def test_trained_delays_are_accepted(self, delay: int) -> None:
        assert VoxtralSTTConfig(api_key="k", streaming_delay_ms=delay).streaming_delay_ms == delay

    @pytest.mark.parametrize("delay", [79, 1300, 2399, 3000])
    def test_other_delays_are_refused(self, delay: int) -> None:
        with pytest.raises(ValueError, match="80 to 1200, or 2400"):
            VoxtralSTTConfig(api_key="k", streaming_delay_ms=delay)

    @pytest.mark.parametrize(
        "thresholds",
        [
            {"start_threshold": 0.5, "eager_end_threshold": 0.3, "end_threshold": 0.4},
            {"start_threshold": 0.4, "eager_end_threshold": 0.5},
            {"eager_end_threshold": 0.33},
            {"start_threshold": 1.2},
        ],
    )
    def test_thresholds_out_of_order_are_refused(self, thresholds: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="order"):
            VoxtralTurnDetection(**thresholds)

    def test_end_timeout_shorter_than_the_silence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="end_timeout_ms"):
            VoxtralTurnDetection(min_silence_ms=800, end_timeout_ms=500)

    def test_aggressive_is_mistrals_preset(self) -> None:
        assert VoxtralTurnDetection.aggressive() == VoxtralTurnDetection(0.5, 0.4, 0.33, 300, 2500)

    @pytest.mark.parametrize(
        ("config", "wait_s"),
        [
            ({}, 0.58),
            ({"turn_detection": VoxtralTurnDetection.aggressive()}, 0.58),
            ({"streaming_delay_ms": 1200}, 1.3),
            ({"streaming_delay_ms": 240}, 0.34),
            ({"final_wait_ms": 650}, 0.65),
        ],
    )
    def test_final_wait_covers_the_streaming_delay(
        self, config: dict[str, Any], wait_s: float
    ) -> None:
        assert VoxtralSTTConfig(api_key="k", **config).final_wait_s == pytest.approx(wait_s)

    def test_api_key_is_required(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            VoxtralSTTConfig(api_key="")

    def test_a_key_no_header_can_carry_is_refused_without_echoing_it(self) -> None:
        with pytest.raises(ValueError, match="non-ASCII") as info:
            VoxtralSTTConfig(api_key="sk-…secret")
        assert "secret" not in str(info.value)

    @pytest.mark.parametrize(
        ("language", "code"), [("fr-CA", "fr"), ("EN_us", "en"), (None, None)]
    )
    def test_a_region_is_dropped_from_the_language(
        self, language: str | None, code: str | None
    ) -> None:
        assert language_code(language) == code

    @pytest.mark.parametrize("language", ["french", "f", "123", "-fr"])
    def test_a_language_that_is_no_code_is_refused(self, language: str) -> None:
        with pytest.raises(ValueError, match="ISO 639-1"):
            VoxtralSTTConfig(api_key="k", language=language)

    def test_the_realtime_url_follows_the_base_url(self) -> None:
        config = VoxtralSTTConfig(api_key="k", base_url="https://eu.example/")
        assert config.stream_url.startswith("wss://eu.example/v1/audio/transcriptions/realtime?")

    @pytest.mark.parametrize(
        ("config", "url"),
        [
            (
                {"base_url": "http://localhost:8080"},
                "ws://localhost:8080/v1/audio/transcriptions/realtime"
                "?model=voxtral-mini-transcribe-realtime-3",
            ),
            (
                {"realtime_url": "wss://proxy.example/rt?region=eu", "realtime_model": "a b&c"},
                "wss://proxy.example/rt?region=eu&model=a+b%26c",
            ),
        ],
    )
    def test_the_realtime_url_is_built_with_its_query(
        self, config: dict[str, Any], url: str
    ) -> None:
        assert VoxtralSTTConfig(api_key="k", **config).stream_url == url

    def test_with_turn_detection_the_stream_carries_turn_after_turn(self) -> None:
        assert VoxtralSTTProvider(VoxtralSTTConfig(api_key="k")).keeps_stream_across_turns
        assert not VoxtralSTTProvider(
            VoxtralSTTConfig(api_key="k", turn_detection=None)
        ).keeps_stream_across_turns

    def test_language_override_is_claimed(self) -> None:
        provider = VoxtralSTTProvider(VoxtralSTTConfig(api_key="k"))
        assert provider.supports_language_override is True
        assert provider.supports_streaming is True
        assert provider.supports_diarization is False


# ---------------------------------------------------------------------------
# Event mapping
# ---------------------------------------------------------------------------


def _feed(transcript: Transcript, *events: dict[str, Any]) -> list[TranscriptionResult]:
    return [result for event in events for result in transcript.feed(event)]


class TestTranscript:
    def test_deltas_give_the_text_so_far(self) -> None:
        results = _feed(Transcript(turns=True), _delta(" Hi"), _delta(","), _delta(" there"))
        assert [r.text for r in results] == ["Hi", "Hi,", "Hi, there"]
        assert not any(r.is_final for r in results)

    def test_a_turn_starts_with_a_speech_start_and_ends_on_close(self) -> None:
        transcript = Transcript(turns=True)
        results = _feed(transcript, *_TURN)
        assert results[0] == _SPEECH_START
        assert results[-1] == TranscriptionResult(text="Hi, I would like a table.", is_final=False)
        assert transcript.closing
        assert transcript.close() == [TranscriptionResult(text="Hi, I would like a table.")]
        assert not transcript.closing

    def test_text_after_the_final_is_counted_not_given(self) -> None:
        transcript = Transcript(turns=True)
        _feed(transcript, *_TURN)
        transcript.close()
        assert _feed(transcript, _delta(" please"), _done("whole session")) == []
        assert transcript.late_chars == len("please")

    def test_text_after_the_next_turn_starts_still_ends_this_one(self) -> None:
        # The text runs a streaming delay behind the turn events: words of the
        # ending turn came after the next turn.start, live.
        transcript = Transcript(turns=True)
        _feed(transcript, *_TURN)
        assert _feed(transcript, _START, _delta(" please")) == [
            _SPEECH_START,
            TranscriptionResult(text="Hi, I would like a table. please", is_final=False),
        ]
        assert transcript.close() == [TranscriptionResult(text="Hi, I would like a table. please")]
        assert not transcript.closing
        # The turn that started meanwhile is open: its words are no late text.
        assert _feed(transcript, _delta(" For"), _delta(" two")) == [
            TranscriptionResult(text="For", is_final=False),
            TranscriptionResult(text="For two", is_final=False),
        ]
        assert transcript.late_chars == 0

    def test_a_turn_that_ends_during_the_wait_closes_next(self) -> None:
        transcript = Transcript(turns=True)
        _feed(transcript, *_TURN, _START, _END)
        assert transcript.close() == [TranscriptionResult(text="Hi, I would like a table.")]
        assert transcript.closing
        assert _feed(transcript, _delta(" Sure.")) == [
            TranscriptionResult(text="Sure.", is_final=False)
        ]
        assert transcript.close() == [TranscriptionResult(text="Sure.")]

    def test_the_latest_turn_to_start_during_the_wait_is_the_open_one(self) -> None:
        # Turn 2 starts and ends during turn 1's wait, then turn 3 starts: after
        # the close, turn 3 is open and its words are neither a final nor late.
        transcript = Transcript(turns=True)
        _feed(transcript, *_TURN, _START, _delta(" Yes."), _END, _START)
        assert transcript.close() == [TranscriptionResult(text="Hi, I would like a table. Yes.")]
        assert not transcript.closing
        _feed(transcript, _delta(" And a window seat"))
        assert transcript.late_chars == 0
        assert _feed(transcript, _END) == []
        assert transcript.close() == [TranscriptionResult(text="And a window seat")]

    def test_the_ending_turn_keeps_its_last_words_when_the_next_starts_quickly(self) -> None:
        # Its last words came up to 0.9 of the delay after its end (0.43 s at
        # 480 ms); the next turn started 0.05 s after the end. The close comes
        # 0.95 of the delay after the end, after those words and before the
        # next turn's own, which come a delay after it.
        transcript = Transcript(turns=True, final_wait_s=0.58, next_turn_wait_s=0.456)
        _feed(transcript, _START, _delta("What is the capital"), _END)
        assert transcript.deadline(1.0, None) == 1.58
        _feed(transcript, _START)
        assert transcript.deadline(1.05, 1.58) == pytest.approx(1.456)
        _feed(transcript, _delta(" of Italy?"))  # at 1.43, before the close
        assert transcript.close() == [TranscriptionResult(text="What is the capital of Italy?")]
        assert _feed(transcript, _delta(" Answer")) == [
            TranscriptionResult(text="Answer", is_final=False)
        ]

    def test_late_text_is_reported_once(self) -> None:
        transcript = Transcript(turns=True)
        _feed(transcript, *_TURN)
        transcript.close()
        _feed(transcript, _delta(" please"))
        assert transcript.take_late() == len("please")
        assert transcript.take_late() == 0

    def test_done_gives_the_turn_still_open(self) -> None:
        transcript = Transcript(turns=True)
        results = _feed(transcript, _START, _delta("Wait"), _done("Wait"))
        assert results[-1] == TranscriptionResult(text="Wait")

    def test_a_turn_without_words_gives_no_final(self) -> None:
        transcript = Transcript(turns=True)
        assert _feed(transcript, _START, _END) == [_SPEECH_START]
        assert transcript.close() == []

    def test_without_turns_done_is_the_final(self) -> None:
        transcript = Transcript(turns=False)
        results = _feed(transcript, _delta("Hi"), _delta(" there"), _done(" Hi there. "))
        assert results[-1] == TranscriptionResult(text="Hi there.")
        assert _feed(Transcript(turns=False), _delta("Hi"), _done(""))[-1].text == "Hi"

    def test_without_turns_turn_events_are_ignored(self) -> None:
        assert _feed(Transcript(turns=False), _START, _END) == []

    def test_error_event_raises_with_the_services_code(self) -> None:
        event = {
            "type": "error",
            "error": {
                "message": "Diarization is not compatible with turn detection.",
                "code": 3051,
            },
        }
        with pytest.raises(VoxtralSTTError, match="not compatible") as info:
            Transcript(turns=True).feed(event)
        assert info.value.code == "3051"
        assert info.value.retryable is False


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_session_takes_every_setting_before_the_audio(self) -> None:
        record = _Record()
        async with _server(_scripted(record, on_end=[_done("")])) as url:
            provider = _provider(
                url,
                context_bias=["Northwind", "Alder Creek Design"],
                turn_detection=VoxtralTurnDetection.aggressive(),
                streaming_delay_ms=240,
            )
            await _collect(
                provider, AudioChunk(data=_PCM_16K, sample_rate=16000), language="fr-FR"
            )

        assert record.path == "/?model=voxtral-mini-transcribe-realtime-3"
        assert record.authorization == "Bearer test-key"
        assert record.update == {
            "type": "session.update",
            "session": {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "target_streaming_delay_ms": 240,
                "language": "fr",
                "context_bias": ["Northwind", "Alder Creek Design"],
                "turn_detection": {
                    "start_threshold": 0.5,
                    "eager_end_threshold": 0.4,
                    "end_threshold": 0.33,
                    "min_silence_ms": 300,
                    "end_timeout_ms": 2500,
                },
            },
        }
        assert record.audio == _PCM_16K
        assert record.controls == [{"type": "input_audio.end"}]

    async def test_without_turn_detection_none_is_sent(self) -> None:
        record = _Record()
        async with _server(_scripted(record, on_end=[_done("")])) as url:
            await _collect(
                _provider(url, turn_detection=None), AudioChunk(data=_PCM_16K, sample_rate=16000)
            )

        assert "turn_detection" not in record.update["session"]
        assert "language" not in record.update["session"]

    async def test_a_turn_is_final_once_its_last_words_had_time(self) -> None:
        # The text of the turn's end followed turn.end by 0.29 s live: within
        # the wait, it belongs to the turn.
        record = _Record()
        script: Script = [*_TURN, 0.02, _delta(" please"), 0.3]
        async with _server(_scripted(record, on_audio=script, on_end=[_done("")])) as url:
            stream = _provider(url, final_wait_ms=150).transcribe_stream(
                _chunks(AudioChunk(data=_PCM_16K, sample_rate=16000))
            )
            results = await _drain(stream)

        assert results[0] == _SPEECH_START
        assert _finals(results) == ["Hi, I would like a table. please"]

    async def test_words_later_than_the_wait_are_dropped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        record = _Record()
        script: Script = [*_TURN, 0.3, _delta(" please")]
        caplog.set_level(logging.WARNING, logger="roomkit.voice.stt.voxtral")
        async with _server(_scripted(record, on_audio=script, on_end=[_done("")])) as url:
            results = await _collect(
                _provider(url, final_wait_ms=50), AudioChunk(data=_PCM_16K, sample_rate=16000)
            )

        assert _finals(results) == ["Hi, I would like a table."]
        assert "6 characters of a turn arrived after its final" in caplog.text
        assert "please" not in caplog.text  # the person's words stay out of the log

    async def test_two_turns_as_the_live_service_cut_them(self) -> None:
        # Mistral's aggressive preset at a 480 ms delay, live on 2026-10-21, a
        # fifth of the time: the first turn ended after "capital", and " of",
        # " Italy" and "?" came after its end and after the next turn.start.
        record = _Record()
        async with _server(_scripted(record, on_audio=_ITALY, on_end=[0.4])) as url:
            results = await _collect(
                _provider(url, streaming_delay_ms=96, final_wait_ms=116),
                AudioChunk(data=_PCM_16K, sample_rate=16000),
            )

        assert _finals(results) == ["What is the capital of Italy?", "Answer in one word."]
        assert [r for r in results if r.is_speech_start] == [_SPEECH_START, _SPEECH_START]

    async def test_the_next_turns_own_words_are_never_the_last_ones(self) -> None:
        # Live with the defaults: the next turn started 0.05 s after turn.end,
        # and its first word came within the final wait. That word trails its
        # turn.start by a streaming delay; the ending turn's tail does not.
        record = _Record()
        script: Script = [
            *_TURN,
            0.02,
            _START,
            _delta(" please"),  # the ending turn's tail, at once
            0.15,
            _delta(" For"),  # the next turn's own word, a delay after its start
            _delta(" two."),
            _END,
            0.3,
        ]
        async with _server(_scripted(record, on_audio=script, on_end=[_done("")])) as url:
            results = await _collect(
                _provider(url, streaming_delay_ms=80, final_wait_ms=500),
                AudioChunk(data=_PCM_16K, sample_rate=16000),
            )

        assert _finals(results) == ["Hi, I would like a table. please", "For two."]

    async def test_per_utterance_final_is_the_done_text(self) -> None:
        record = _Record()
        on_end: Script = [_delta("Thanks"), _delta("."), _done("Thanks.")]
        async with _server(_scripted(record, on_end=on_end)) as url:
            results = await _collect(
                _provider(url, turn_detection=None),
                AudioChunk(data=_PCM_16K, sample_rate=16000),
                AudioChunk(data=_PCM_16K, sample_rate=16000, is_final=True),
            )

        assert record.audio == _PCM_16K * 2
        assert results == [
            TranscriptionResult(text="Thanks", is_final=False),
            TranscriptionResult(text="Thanks.", is_final=False),
            TranscriptionResult(text="Thanks."),
        ]

    async def test_a_backlog_leaves_in_appends_of_100_ms(self) -> None:
        record = _Record()
        async with _server(_scripted(record, on_end=[_done("")])) as url:
            await _collect(_provider(url), AudioChunk(data=_PCM_16K * 5, sample_rate=16000))

        assert [len(base64.b64decode(piece)) for piece in record.appends] == [3200] * 5

    async def test_documented_rates_go_through_untouched(self) -> None:
        record = _Record()
        pcm = b"\x02\x00" * 2400
        async with _server(_scripted(record, on_end=[_done("")])) as url:
            await _collect(_provider(url), AudioChunk(data=pcm, sample_rate=24000))

        assert record.update["session"]["audio_format"]["sample_rate"] == 24000
        assert record.audio == pcm

    async def test_other_rates_are_resampled_to_16khz(self) -> None:
        record = _Record()
        async with _server(_scripted(record, on_end=[_done("")])) as url:
            await _collect(_provider(url), AudioChunk(data=b"\x03\x00" * 1200, sample_rate=12000))

        assert record.update["session"]["audio_format"]["sample_rate"] == 16000
        assert len(record.audio) == 1600 * 2  # 100 ms at 16 kHz

    async def test_a_refused_setting_raises_the_services_error(self) -> None:
        record = _Record()
        refusal = {
            "type": "error",
            "error": {
                "message": "target_streaming_delay_ms must be one of the trained values",
                "code": 3051,
            },
        }
        async with _server(_scripted(record, updated=refusal)) as url:
            with pytest.raises(VoxtralSTTError, match="trained values") as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.code == "3051"
        assert info.value.retryable is False
        assert record.appends == []

    @pytest.mark.parametrize(
        ("status", "retryable", "hint"),
        [
            (HTTPStatus.UNAUTHORIZED, False, ""),
            (HTTPStatus.FORBIDDEN, False, "realtime model id"),
            (HTTPStatus.SERVICE_UNAVAILABLE, True, ""),
        ],
    )
    async def test_upgrade_refused_by_http_status(
        self, status: HTTPStatus, retryable: bool, hint: str
    ) -> None:
        async def handler(ws: ServerConnection) -> None:  # pragma: no cover - never reached
            await ws.close()

        def refuse(connection: ServerConnection, request: Any) -> Any:
            return connection.respond(status, "")

        async with _server(handler, process_request=refuse) as url:
            with pytest.raises(VoxtralSTTError, match=hint or None) as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.status_code == status
        assert info.value.retryable is retryable

    async def test_connection_dropped_mid_stream_is_retryable(self) -> None:
        async def handler(ws: ServerConnection) -> None:
            await ws.send(json.dumps(_CREATED))
            await ws.recv()
            await ws.send(json.dumps({"type": "session.updated", "session": {}}))
            await ws.recv()
            ws.transport.abort()

        async with _server(handler) as url:
            with pytest.raises(VoxtralSTTError) as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.retryable is True

    async def test_no_end_of_transcript_in_time_raises(self) -> None:
        async def silent(ws: ServerConnection) -> None:
            await ws.send(json.dumps(_CREATED))
            await ws.recv()
            await ws.send(json.dumps({"type": "session.updated", "session": {}}))
            await ws.wait_closed()  # never a transcription.done

        async with _server(silent) as url:
            with pytest.raises(VoxtralSTTError, match="no end of transcript") as info:
                await _collect(
                    _provider(url, final_timeout_s=0.2),
                    AudioChunk(data=_PCM_16K, sample_rate=16000),
                )

        assert info.value.retryable is True

    @pytest.mark.wallclock
    async def test_the_handshake_has_one_bound_in_all(self) -> None:
        # session.created inside the bound, session.updated never: a bound per
        # event would wait 0.25 s then 0.3 s more.
        async def slow(ws: ServerConnection) -> None:
            await asyncio.sleep(0.25)
            await ws.send(json.dumps(_CREATED))
            await ws.recv()
            await ws.wait_closed()  # never a session.updated

        loop = asyncio.get_running_loop()
        async with _server(slow) as url:
            start = loop.time()
            with pytest.raises(VoxtralSTTError) as info:
                await _collect(
                    _provider(url, handshake_timeout_s=0.3),
                    AudioChunk(data=_PCM_16K, sample_rate=16000),
                )
            elapsed = loop.time() - start

        assert info.value.retryable is True
        assert elapsed < 0.45

    @pytest.mark.wallclock
    async def test_an_abandoned_session_closes_without_waiting_for_the_service(
        self,
    ) -> None:
        # Mistral answers a close frame about 1.1 s late; this server never
        # does. A stream given up mid-session (the channel closing) must not
        # wait for it.
        released = asyncio.Event()

        async def deaf(ws: ServerConnection) -> None:
            await ws.send(json.dumps(_CREATED))
            await ws.recv()
            await ws.send(json.dumps({"type": "session.updated", "session": {}}))
            await ws.recv()  # the first append
            await ws.send(json.dumps(_delta("Hello")))
            ws.transport.pause_reading()  # the close frame is never read
            await released.wait()
            ws.transport.abort()

        async def endless() -> AsyncIterator[AudioChunk]:
            while True:
                yield AudioChunk(data=_PCM_16K, sample_rate=16000)
                await asyncio.sleep(0.1)

        loop = asyncio.get_running_loop()
        async with _server(deaf) as url:
            stream = _provider(url).transcribe_stream(endless())
            partial = await asyncio.wait_for(anext(stream), timeout=5)
            start = loop.time()
            await asyncio.wait_for(stream.aclose(), timeout=5)
            elapsed = loop.time() - start
            released.set()

        assert partial.text == "Hello"
        assert elapsed < 0.5

    async def test_no_audio_opens_no_connection(self) -> None:
        record = _Record()
        async with _server(_scripted(record)) as url:
            results = await _collect(
                _provider(url), AudioChunk(data=b"", sample_rate=16000, is_final=True)
            )

        assert results == []
        assert record.connections == 0

    async def test_failing_audio_source_still_ends_the_audio(self) -> None:
        record = _Record()

        async def broken() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=_PCM_16K, sample_rate=16000)
            raise RuntimeError("microphone unplugged")

        async with _server(_scripted(record, on_end=[_done("Hello.")])) as url:
            provider = _provider(url, turn_detection=None)
            results = await _drain(provider.transcribe_stream(broken()))

        assert record.controls == [{"type": "input_audio.end"}]
        assert _finals(results) == ["Hello."]


# ---------------------------------------------------------------------------
# File (REST)
# ---------------------------------------------------------------------------


def _multipart(request: Any) -> list[tuple[str, str | None, str | None, bytes]]:
    """``(part name, filename, content type, payload)`` of each part, in order."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = email.message_from_bytes(head + request.content, policy=email.policy.HTTP)
    return [
        (
            part.get_param("name", header="content-disposition"),
            part.get_filename(),
            part.get_content_type() if part.get("content-type") else None,
            part.get_payload(decode=True),
        )
        for part in message.iter_parts()
    ]


@contextlib.contextmanager
def _rest(respond: Callable[[Any], Any]) -> Any:
    real = httpx.AsyncClient
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return respond(request)

    with patch(
        "httpx.AsyncClient",
        side_effect=lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
    ):
        yield seen


_ANSWER = {
    "model": "voxtral-mini-transcribe-3",
    "text": " Hi, I would like to book a table. ",
    "language": None,
    "segments": [],
    "usage": {"prompt_audio_seconds": 5},
}


class TestFile:
    async def test_pcm_is_uploaded_as_a_wav_with_the_settings(self) -> None:
        with _rest(lambda _: httpx.Response(200, json=_ANSWER)) as seen:
            provider = VoxtralSTTProvider(
                VoxtralSTTConfig(
                    api_key="test-key", language="fr-CA", context_bias=["Northwind", "Quillara"]
                )
            )
            result = await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))
            await provider.close()

        assert result == TranscriptionResult(text="Hi, I would like to book a table.")
        request = seen[0]
        assert str(request.url) == "https://api.mistral.ai/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-key"
        parts = _multipart(request)
        fields = [(name, payload.decode()) for name, filename, _, payload in parts if not filename]
        assert fields == [
            ("model", "voxtral-mini-transcribe-3"),
            ("language", "fr"),
            ("context_bias", "Northwind"),
            ("context_bias", "Quillara"),
        ]
        (name, filename, mime, payload) = next(p for p in parts if p[1])
        assert (name, filename, mime) == ("file", "audio.wav", "audio/wav")
        with wave.open(io.BytesIO(payload)) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            assert wav.readframes(wav.getnframes()) == _PCM_16K

    async def test_a_call_language_overrides_the_configured_one(self) -> None:
        with _rest(lambda _: httpx.Response(200, json=_ANSWER)) as seen:
            provider = VoxtralSTTProvider(VoxtralSTTConfig(api_key="k", language="en"))
            await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000), language="de")
            await provider.close()

        assert ("language", b"de") in [(p[0], p[3]) for p in _multipart(seen[0])]

    async def test_a_frame_at_another_rate_is_resampled(self) -> None:
        frame = AudioFrame(data=b"\x03\x00" * 1200, sample_rate=12000)
        with _rest(lambda _: httpx.Response(200, json=_ANSWER)) as seen:
            provider = VoxtralSTTProvider(VoxtralSTTConfig(api_key="k"))
            await provider.transcribe(frame)
            await provider.close()

        payload = next(p[3] for p in _multipart(seen[0]) if p[1])
        with wave.open(io.BytesIO(payload)) as wav:
            assert (wav.getframerate(), wav.getnframes()) == (16000, 1600)

    async def test_a_data_uri_is_sent_as_the_file_it_carries(self) -> None:
        url = "data:audio/mpeg;base64," + base64.b64encode(b"ID3fake").decode()
        with _rest(lambda _: httpx.Response(200, json=_ANSWER)) as seen:
            provider = VoxtralSTTProvider(VoxtralSTTConfig(api_key="k"))
            await provider.transcribe(AudioContent(url=url, mime_type="audio/mpeg"))
            await provider.close()

        name, filename, mime, payload = next(p for p in _multipart(seen[0]) if p[1])
        assert (name, filename, mime, payload) == ("file", "audio.mp3", "audio/mpeg", b"ID3fake")

    async def test_an_http_url_is_fetched_by_mistral_not_roomkit(self) -> None:
        with _rest(lambda _: httpx.Response(200, json=_ANSWER)) as seen:
            provider = VoxtralSTTProvider(VoxtralSTTConfig(api_key="k"))
            await provider.transcribe(AudioContent(url="https://example.com/call.mp3"))
            await provider.close()

        assert len(seen) == 1  # the one request to Mistral, nothing fetched here
        parts = _multipart(seen[0])
        assert ("file_url", b"https://example.com/call.mp3") in [(p[0], p[3]) for p in parts]
        assert not any(p[1] for p in parts)

    @pytest.mark.parametrize(
        ("status", "body", "message", "code", "retryable"),
        [
            (
                400,
                {
                    "object": "error",
                    "message": "Invalid model: x",
                    "type": "invalid_model",
                    "code": "1500",
                },
                "Invalid model: x",
                "1500",
                False,
            ),
            (401, {"detail": "Invalid API Key"}, "Invalid API Key", None, False),
            (
                422,
                {"detail": [{"msg": "Input should be greater than or equal to 1"}]},
                "greater than or equal",
                None,
                False,
            ),
            (429, {"object": "error", "message": "Rate limit exceeded"}, "Rate limit", None, True),
            (503, {}, "", None, True),
        ],
    )
    async def test_error_bodies_are_mapped(
        self,
        status: int,
        body: dict[str, Any],
        message: str,
        code: str | None,
        retryable: bool,
    ) -> None:
        with _rest(lambda _: httpx.Response(status, json=body)):
            provider = VoxtralSTTProvider(VoxtralSTTConfig(api_key="k"))
            with pytest.raises(VoxtralSTTError, match=message or None) as info:
                await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))
            await provider.close()

        assert info.value.status_code == status
        assert info.value.code == code
        assert info.value.retryable is retryable


# ---------------------------------------------------------------------------
# VoiceChannel, continuous mode
# ---------------------------------------------------------------------------


async def _eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)


class TestVoiceChannelContinuous:
    async def test_model_turn_detection_drives_the_turn(self) -> None:
        # No pipeline VAD: the channel streams everything, and the model's
        # turn.start / turn.end are the turn's only boundaries.
        record = _Record()
        handler = _scripted(record, on_audio=list(_TURN), on_end=[_done("")])

        transcripts: list[str] = []
        speech_starts: list[Any] = []
        async with _server(handler) as url:
            stt = _provider(url)
            backend = MockVoiceBackend()
            channel = VoiceChannel(
                "voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig()
            )
            kit = RoomKit(stt=stt, voice=backend)
            kit.register_channel(channel)

            @kit.hook(HookTrigger.ON_TRANSCRIPTION)
            async def on_transcription(event: Any, ctx: Any) -> HookResult:
                transcripts.append(event.text)
                return HookResult.allow()

            @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
            async def on_speech_start(event: Any, ctx: Any) -> None:
                speech_starts.append(event)

            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice-1")
            session = await kit.join(room.id, "voice-1", participant_id="user-1")
            assert channel._continuous_stt

            await backend.simulate_audio_received(session, AudioFrame(data=_PCM_16K * 2))
            await _eventually(lambda: bool(transcripts))
            await channel.close()

        assert transcripts == ["Hi, I would like a table."]
        assert speech_starts
        assert record.update["session"]["turn_detection"]["min_silence_ms"] == 500

    async def test_a_turn_started_during_the_wait_stays_whole(self) -> None:
        # The live capture: the next turn starts before the first one's last
        # words. The server plays its events while the audio comes and stops
        # at input_audio.end, as the service does; a second session would hear
        # only what was said after the first one ended. The channel keeps the
        # session, so neither turn is cut.
        record = _Record()
        rest_of_turn_2: Script = [_START, _delta(" in one word."), _END]

        async def live(ws: ServerConnection) -> None:
            record.connections += 1
            script = _ITALY[:-2] if record.connections == 1 else rest_of_turn_2
            await ws.send(json.dumps(_CREATED))
            await ws.recv()
            await ws.send(json.dumps({"type": "session.updated", "session": {}}))
            player: asyncio.Task[None] | None = None
            async for message in ws:
                kind = json.loads(message)["type"]
                if kind == "input_audio.append" and player is None:
                    player = asyncio.create_task(_play(ws, list(script)))
                elif kind == "input_audio.end":
                    break
            if player is not None:
                player.cancel()
            await ws.send(json.dumps(_done("")))
            ws.transport.abort()

        transcripts: list[str] = []
        async with _server(live) as url:
            stt = _provider(url, streaming_delay_ms=96, final_wait_ms=116)
            backend = MockVoiceBackend()
            channel = VoiceChannel(
                "voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig()
            )
            kit = RoomKit(stt=stt, voice=backend)
            kit.register_channel(channel)

            @kit.hook(HookTrigger.ON_TRANSCRIPTION)
            async def on_transcription(event: Any, ctx: Any) -> HookResult:
                transcripts.append(event.text)
                return HookResult.allow()

            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice-1")
            session = await kit.join(room.id, "voice-1", participant_id="user-1")

            async def microphone() -> None:
                while True:
                    await backend.simulate_audio_received(session, AudioFrame(data=_PCM_16K))
                    await asyncio.sleep(0.1)

            mic = asyncio.create_task(microphone())
            with contextlib.suppress(AssertionError):
                await _eventually(lambda: len(transcripts) >= 2, timeout=3.0)
            mic.cancel()
            await channel.close()

        assert transcripts == ["What is the capital of Italy?", "Answer in one word."]
        assert record.connections == 1
