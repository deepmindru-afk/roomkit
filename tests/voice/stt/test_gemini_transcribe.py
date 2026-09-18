"""Tests for the Gemini streaming STT provider (``gemini-3.5-transcribe-live``).

The socket is faked: what these pin is the setup RoomKit sends, how the two
transcript fields are turned into results, and that the stream ends and
reports failures instead of going quiet.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from roomkit.voice.base import AudioChunk
from roomkit.voice.stt.gemini_transcribe import (
    GeminiTranscribeConfig,
    GeminiTranscribeProvider,
)


def _content(**fields: Any) -> SimpleNamespace:
    base = {
        "interim_input_transcription": None,
        "input_transcription": None,
        "turn_complete": False,
        "generation_complete": False,
    }
    base.update(fields)
    return SimpleNamespace(server_content=SimpleNamespace(**base))


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(text=value)


class _FakeSession:
    """A Live session that replays a scripted server stream."""

    def __init__(self, responses: list[Any], *, send_error: Exception | None = None) -> None:
        self._responses = responses
        self._send_error = send_error
        self.sent_audio: list[Any] = []
        self.stream_ended = False

    async def send_realtime_input(
        self, *, audio: Any = None, audio_stream_end: bool | None = None
    ) -> None:
        if self._send_error is not None:
            raise self._send_error
        if audio is not None:
            self.sent_audio.append(audio)
        if audio_stream_end:
            self.stream_ended = True

    async def receive(self) -> AsyncIterator[Any]:
        for response in self._responses:
            # Let the sender task run between messages, the way a socket does.
            await asyncio.sleep(0)
            yield response
        # A real socket stays open; the provider must break out on its own.
        while True:
            await asyncio.sleep(0.01)


class _FakeConnect:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.model: str | None = None
        self.config: Any = None

    def __call__(self, *, model: str, config: Any) -> _FakeConnect:
        self.model = model
        self.config = config
        return self

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _provider(session: _FakeSession, **kwargs: Any) -> tuple[Any, _FakeConnect]:
    config = GeminiTranscribeConfig(api_key="test-key", **kwargs)
    provider = GeminiTranscribeProvider(config)
    connect = _FakeConnect(session)
    provider._client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=connect)))
    return provider, connect


async def _chunks(*payloads: bytes, sample_rate: int = 16000) -> AsyncIterator[AudioChunk]:
    for payload in payloads:
        yield AudioChunk(data=payload, sample_rate=sample_rate)


# --- configuration --------------------------------------------------------


class TestConfig:
    def test_defaults_ask_the_model_to_detect_the_language(self) -> None:
        config = GeminiTranscribeConfig(api_key="k")
        assert config.model == "gemini-3.5-transcribe-live"
        assert config.language_codes == []
        assert config.mode == "VERBATIM"

    def test_mode_is_normalised(self) -> None:
        assert GeminiTranscribeConfig(api_key="k", mode="smart").mode == "SMART"

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            GeminiTranscribeConfig(api_key="k", mode="creative")

    def test_an_empty_api_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            GeminiTranscribeConfig(api_key="  ")

    def test_the_vocabulary_bound_is_enforced(self) -> None:
        with pytest.raises(ValueError, match="1000 terms"):
            GeminiTranscribeConfig(api_key="k", custom_vocabulary=[f"t{i}" for i in range(1001)])


class TestProviderShape:
    def test_it_declares_streaming_and_the_language_override(self) -> None:
        provider = GeminiTranscribeProvider(GeminiTranscribeConfig(api_key="k"))
        assert provider.supports_streaming is True
        assert provider.supports_language_override is True
        assert provider.name == "GeminiTranscribe"


# --- the setup RoomKit sends ---------------------------------------------


class TestSetup:
    async def test_setup_carries_the_transcription_config(self) -> None:
        session = _FakeSession([_content(turn_complete=True)])
        provider, connect = _provider(
            session,
            language_codes=["fr-FR"],
            custom_vocabulary=["RoomKit"],
            mode="SMART",
        )

        async for _ in provider.transcribe_stream(_chunks(b"\x00\x01")):
            pass

        assert connect.model == "gemini-3.5-transcribe-live"
        assert connect.config.response_modalities == ["TEXT"]
        transcription = connect.config.input_audio_transcription
        assert transcription.language_codes == ["fr-FR"]
        assert transcription.custom_vocabulary == ["RoomKit"]
        assert transcription.mode == "SMART"

    async def test_a_per_call_language_replaces_the_configured_hints(self) -> None:
        """Appending would dilute the hint the caller just gave."""
        session = _FakeSession([_content(turn_complete=True)])
        provider, connect = _provider(session, language_codes=["fr-FR", "de-DE"])

        async for _ in provider.transcribe_stream(_chunks(b"\x00"), language="en-US"):
            pass

        assert connect.config.input_audio_transcription.language_codes == ["en-US"]

    async def test_no_hints_means_automatic_detection(self) -> None:
        session = _FakeSession([_content(turn_complete=True)])
        provider, connect = _provider(session)

        async for _ in provider.transcribe_stream(_chunks(b"\x00")):
            pass

        assert connect.config.input_audio_transcription.language_codes is None


# --- the transcript stream ------------------------------------------------


class TestStream:
    async def test_interim_and_final_transcripts_are_distinguished(self) -> None:
        session = _FakeSession(
            [
                _content(interim_input_transcription=_text("bon")),
                _content(interim_input_transcription=_text("bonjour")),
                _content(input_transcription=_text("Bonjour."), turn_complete=True),
            ]
        )
        provider, _ = _provider(session)

        results = [r async for r in provider.transcribe_stream(_chunks(b"\x00", b"\x01"))]

        assert [(r.text, r.is_final) for r in results] == [
            ("bon", False),
            ("bonjour", False),
            ("Bonjour.", True),
        ]

    async def test_the_audio_is_sent_and_the_input_is_closed(self) -> None:
        session = _FakeSession([_content(turn_complete=True)])
        provider, _ = _provider(session)

        async for _ in provider.transcribe_stream(_chunks(b"\x00\x01", b"\x02\x03")):
            pass

        assert [blob.data for blob in session.sent_audio] == [b"\x00\x01", b"\x02\x03"]
        assert session.stream_ended is True

    async def test_empty_chunks_are_not_sent(self) -> None:
        session = _FakeSession([_content(turn_complete=True)])
        provider, _ = _provider(session)

        async for _ in provider.transcribe_stream(_chunks(b"", b"\x01")):
            pass

        assert [blob.data for blob in session.sent_audio] == [b"\x01"]

    async def test_the_sample_rate_rides_the_mime_type(self) -> None:
        session = _FakeSession([_content(turn_complete=True)])
        provider, _ = _provider(session)

        async for _ in provider.transcribe_stream(_chunks(b"\x01", sample_rate=8000)):
            pass

        assert session.sent_audio[0].mime_type == "audio/pcm;rate=8000"

    async def test_an_off_rate_stream_is_reported_once(self, caplog) -> None:
        session = _FakeSession([_content(turn_complete=True)])
        provider, _ = _provider(session)

        with caplog.at_level("WARNING"):
            async for _ in provider.transcribe_stream(_chunks(b"\x01", b"\x02", sample_rate=8000)):
                pass

        assert caplog.text.count("documents 16000 Hz input") == 1

    async def test_generation_complete_also_ends_the_stream(self) -> None:
        """Either boundary is the server saying it has nothing more to send."""
        session = _FakeSession([_content(generation_complete=True)])
        provider, _ = _provider(session)

        results = [r async for r in provider.transcribe_stream(_chunks(b"\x01"))]
        assert results == []

    async def test_a_send_failure_is_raised_not_swallowed(self) -> None:
        """Ending clean would tell the caller it has the whole transcript."""
        session = _FakeSession(
            [_content(turn_complete=True)], send_error=RuntimeError("socket gone")
        )
        provider, _ = _provider(session)

        with pytest.raises(RuntimeError, match="socket gone"):
            async for _ in provider.transcribe_stream(_chunks(b"\x01")):
                pass


# --- the batch convenience ------------------------------------------------


class TestTranscribe:
    async def test_it_joins_the_finals(self) -> None:
        session = _FakeSession(
            [
                _content(interim_input_transcription=_text("ignored partial")),
                _content(input_transcription=_text("Bonjour,")),
                _content(input_transcription=_text("ca va ?"), turn_complete=True),
            ]
        )
        provider, _ = _provider(session)

        result = await provider.transcribe(AudioChunk(data=b"\x01\x02", sample_rate=16000))

        assert result.text == "Bonjour, ca va ?"
        assert result.is_final is True

    async def test_a_url_backed_input_is_refused_with_the_right_pointer(self) -> None:
        provider, _ = _provider(_FakeSession([]))

        with pytest.raises(TypeError, match="GeminiSTTProvider"):
            await provider.transcribe(SimpleNamespace(url="https://example.test/a.wav"))
