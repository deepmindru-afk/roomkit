"""ElevenLabs streaming text input over WebSocket.

v4 and v4 Turbo stream over the Text to Dialogue socket, the v2 and v2.5
models over the Text to Speech socket, and v3 has none. A fake socket answers
each chunk with audio and the end of the input with the final flag.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import ServerConnection, serve

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts._elevenlabs_ws import (
    DialogueSocket,
    SpeechSocket,
    socket_for,
    stream_audio,
)
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

_CLOSED = object()


def _config(**kwargs: Any) -> ElevenLabsConfig:
    """A config with streaming input on, unless the test says otherwise."""
    kwargs.setdefault("stream_input", True)
    return ElevenLabsConfig(**kwargs)


class FakeSocket:
    """Answers each text chunk with one audio message, and the end of the input
    with the final flag. The audio is the chunk's text, or 16-bit silence as
    long as it with ``pcm``; ``error`` is answered to the first chunk instead."""

    def __init__(self, *, error: str | None = None, pcm: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._error = error
        self._pcm = pcm
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        text = (
            message.get("text") or "".join(i["text"] for i in message.get("inputs", []))
        ).strip()
        if text and self._error:
            self._inbox.put_nowait({"error": self._error})
        elif text:
            data = b"\x00\x00" * len(text) if self._pcm else text.encode()
            audio = base64.b64encode(data).decode()
            self._inbox.put_nowait({"audio": audio})
        elif message == {"text": ""}:
            self._inbox.put_nowait({"audio": None, "isFinal": True})
        elif message.get("close_socket"):
            self._inbox.put_nowait({"is_final": True, "is_final_audio_for_turn": True})

    async def close(self) -> None:
        self.closed = True
        self._inbox.put_nowait(_CLOSED)

    def __aiter__(self) -> FakeSocket:
        return self

    async def __anext__(self) -> str:
        item = await self._inbox.get()
        if item is _CLOSED:
            raise StopAsyncIteration
        return json.dumps(item)


def _connect(socket: FakeSocket) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    @contextlib.asynccontextmanager
    async def connect(url: str, **kwargs: Any) -> AsyncIterator[FakeSocket]:
        calls.append((url, kwargs))
        try:
            yield socket
        finally:
            socket.closed = True

    return connect, calls


async def _texts(*items: str, delay: float = 0.0) -> AsyncIterator[str]:
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


async def _run(
    provider: ElevenLabsTTSProvider, *texts: str, voice: str | None = None
) -> tuple[list[Any], FakeSocket, list[tuple[str, dict[str, Any]]]]:
    socket = FakeSocket()
    connect, calls = _connect(socket)
    with patch("websockets.connect", connect):
        chunks = [c async for c in provider.synthesize_stream_input(_texts(*texts), voice=voice)]
    return chunks, socket, calls


class TestSocketChoice:
    @pytest.mark.parametrize(
        ("model_id", "socket_type"),
        [
            ("eleven_v4_turbo", DialogueSocket),
            ("eleven_v4", DialogueSocket),
            ("eleven_multilingual_v2", SpeechSocket),
            ("eleven_flash_v2_5", SpeechSocket),
            ("eleven_turbo_v2_5", SpeechSocket),
        ],
    )
    def test_the_model_picks_its_socket(self, model_id: str, socket_type: type) -> None:
        assert isinstance(socket_for(model_id), socket_type)
        provider = ElevenLabsTTSProvider(_config(api_key="k", model_id=model_id))
        assert provider.supports_streaming_input is True

    @pytest.mark.parametrize("model_id", ["eleven_v4_turbo", "eleven_multilingual_v2"])
    def test_off_by_default(self, model_id: str) -> None:
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=model_id))
        assert provider.supports_streaming_input is False

    def test_expressive_mode_streams_input(self) -> None:
        provider = ElevenLabsTTSProvider(_config(api_key="k", expressive=True))
        assert provider.supports_streaming_input is True

    @pytest.mark.parametrize(
        "config",
        [{"model_id": "eleven_v3"}, {"model_id": "eleven_v4_turbo", "stream_input": False}],
    )
    async def test_no_socket_no_streaming_input(self, config: dict[str, Any]) -> None:
        provider = ElevenLabsTTSProvider(_config(api_key="k", **config))
        assert provider.supports_streaming_input is False
        with pytest.raises(NotImplementedError):
            [c async for c in provider.synthesize_stream_input(_texts("Hi."))]


class TestDialogueSocket:
    async def test_v4_turbo_sends_the_dialogue_messages(self) -> None:
        provider = ElevenLabsTTSProvider(
            _config(api_key="key", model_id="eleven_v4_turbo", output_format="pcm_16000")
        )

        chunks, socket, calls = await _run(provider, "Hello there. ", "Bye.")

        url, kwargs = calls[0]
        assert urlparse(url).path == "/v1/text-to-dialogue/stream-input"
        assert parse_qs(urlparse(url).query) == {
            "model_id": ["eleven_v4_turbo"],
            "output_format": ["pcm_16000"],
        }
        assert kwargs["additional_headers"] == {"xi-api-key": "key"}
        voice = "21m00Tcm4TlvDq8ikWAM"
        assert socket.sent == [
            {"voices": [voice]},
            {"inputs": [{"text": "Hello there. ", "voice_id": voice}]},
            {"flush": True},
            {"inputs": [{"text": "Bye.", "voice_id": voice}]},
            {"flush": True},
            {"close_socket": True},
        ]
        assert [c.data for c in chunks] == [b"Hello there.", b"Bye.", b""]
        assert chunks[0].sample_rate == 16000 and chunks[0].format == "pcm_s16le"
        assert chunks[-1].is_final is True
        assert socket.closed is True

    async def test_a_voice_override_is_the_registered_voice(self) -> None:
        provider = ElevenLabsTTSProvider(_config(api_key="k", model_id="eleven_v4"))

        _, socket, _ = await _run(provider, "Hi.", voice="other")

        assert socket.sent[0] == {"voices": ["other"]}
        assert socket.sent[1] == {"inputs": [{"text": "Hi.", "voice_id": "other"}]}

    def test_custom_voice_settings_warn_they_are_not_applied(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="roomkit.voice.tts.elevenlabs"):
            ElevenLabsTTSProvider(_config(api_key="k", model_id="eleven_v4_turbo", stability=0.8))
        assert "applies no voice settings" in caplog.text

    @pytest.mark.parametrize(
        "config",
        [
            {"model_id": "eleven_v4_turbo"},
            {"model_id": "eleven_v4_turbo", "stability": 0.8, "stream_input": False},
            {"model_id": "eleven_flash_v2_5", "stability": 0.8},
        ],
    )
    def test_no_warning_when_nothing_is_lost(
        self, config: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="roomkit.voice.tts.elevenlabs"):
            ElevenLabsTTSProvider(_config(api_key="k", **config))
        assert "applies no voice settings" not in caplog.text


class TestSpeechSocket:
    async def test_flash_sends_the_speech_messages(self) -> None:
        provider = ElevenLabsTTSProvider(
            _config(
                api_key="key",
                voice_id="v1",
                model_id="eleven_flash_v2_5",
                optimize_streaming_latency=3,
                stability=0.6,
            )
        )

        chunks, socket, calls = await _run(provider, "Hello there.", "Bye. ")

        url, _ = calls[0]
        assert urlparse(url).path == "/v1/text-to-speech/v1/stream-input"
        assert parse_qs(urlparse(url).query) == {
            "model_id": ["eleven_flash_v2_5"],
            "output_format": ["mp3_44100_128"],
            "optimize_streaming_latency": ["3"],
        }
        assert socket.sent == [
            {
                "text": " ",
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            },
            {"text": "Hello there. ", "flush": True},
            {"text": "Bye. ", "flush": True},
            {"text": ""},
        ]
        assert [c.data for c in chunks] == [b"Hello there.", b"Bye.", b""]


class TestStreamLifecycle:
    async def test_an_idle_text_stream_is_kept_alive(self) -> None:
        socket = FakeSocket()
        connect, _ = _connect(socket)
        with patch("websockets.connect", connect):
            audio = stream_audio(
                DialogueSocket(),
                api_key="k",
                voice_id="v",
                query={},
                voice_settings={},
                text_stream=_texts("Late.", delay=0.05),
                keep_alive_s=0.01,
            )
            [a async for a in audio]

        keep_alives = [m for m in socket.sent if m == {"keep_alive": True}]
        assert keep_alives
        # The chunk awaited across keep-alives still arrives, once.
        assert socket.sent.count({"inputs": [{"text": "Late.", "voice_id": "v"}]}) == 1

    async def test_a_server_error_raises(self) -> None:
        provider = ElevenLabsTTSProvider(_config(api_key="k", model_id="eleven_v4"))
        socket = FakeSocket(error="quota exceeded")
        connect, _ = _connect(socket)
        with patch("websockets.connect", connect), pytest.raises(RuntimeError, match="quota"):
            [c async for c in provider.synthesize_stream_input(_texts("Hi."))]

    async def test_an_error_of_the_text_stream_propagates(self) -> None:
        async def failing() -> AsyncIterator[str]:
            yield "One."
            raise ValueError("the LLM stream broke")

        provider = ElevenLabsTTSProvider(_config(api_key="k", model_id="eleven_v4"))
        socket = FakeSocket()
        connect, _ = _connect(socket)
        with (
            patch("websockets.connect", connect),
            pytest.raises(ValueError, match="LLM stream broke"),
        ):
            [c async for c in provider.synthesize_stream_input(failing())]
        assert socket.closed is True
        assert {"close_socket": True} not in socket.sent

    async def test_closing_early_stops_the_sender_and_the_socket(self) -> None:
        text_closed = asyncio.Event()

        async def endless() -> AsyncIterator[str]:
            try:
                yield "One."
                await asyncio.Event().wait()  # an LLM still generating
                yield "never"
            finally:
                text_closed.set()

        provider = ElevenLabsTTSProvider(_config(api_key="k", model_id="eleven_v4"))
        socket = FakeSocket()
        connect, _ = _connect(socket)
        with patch("websockets.connect", connect):
            stream = provider.synthesize_stream_input(endless())
            first = await anext(stream)
            await stream.aclose()  # a barge-in

        assert first.data == b"One."
        assert socket.closed is True
        await asyncio.wait_for(text_closed.wait(), 1)
        assert {"close_socket": True} not in socket.sent

    async def test_cancelling_the_consumer_is_not_swallowed(self) -> None:
        async def endless() -> AsyncIterator[str]:
            yield "One."
            await asyncio.Event().wait()
            yield "never"

        provider = ElevenLabsTTSProvider(_config(api_key="k", model_id="eleven_v4"))
        socket = FakeSocket()
        connect, _ = _connect(socket)
        received: list[Any] = []

        async def consume() -> None:
            async for chunk in provider.synthesize_stream_input(endless()):
                received.append(chunk)

        with patch("websockets.connect", connect):
            task = asyncio.create_task(consume())
            while not received:
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert socket.closed is True


@contextlib.asynccontextmanager
async def _server(handler: Any) -> AsyncIterator[None]:
    """A local websockets server the sockets' URLs point at."""
    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        with patch("roomkit.voice.tts._elevenlabs_ws.BASE_URL", f"ws://127.0.0.1:{port}"):
            yield


def _audio_message() -> str:
    return json.dumps({"audio": base64.b64encode(b"\x00\x00" * 800).decode()})


def _stream(text_stream: AsyncIterator[str]) -> Any:
    return stream_audio(
        DialogueSocket(),
        api_key="k",
        voice_id="v",
        query={},
        voice_settings={},
        text_stream=text_stream,
    )


async def _endless() -> AsyncIterator[str]:
    yield "One."
    await asyncio.Event().wait()
    yield "never"


class TestAgainstARealSocket:
    """What the fake cannot show: flow control and a close that yields."""

    async def test_a_barge_in_does_not_wait_behind_unread_audio(self) -> None:
        async def flood(ws: ServerConnection) -> None:
            await ws.recv()  # the open message
            for _ in range(200):  # faster than any transport plays it
                await ws.send(_audio_message())
            await ws.wait_closed()

        async with _server(flood):
            stream = _stream(_endless())
            await anext(stream)
            await asyncio.sleep(0.2)  # the audio piles up unread
            started = time.monotonic()
            await stream.aclose()

        assert time.monotonic() - started < 2.0

    async def test_an_error_of_the_text_stream_propagates(self) -> None:
        async def echo(ws: ServerConnection) -> None:
            async for raw in ws:
                if "inputs" in json.loads(raw):
                    await ws.send(_audio_message())

        async def failing() -> AsyncIterator[str]:
            yield "One."
            await asyncio.sleep(0.05)
            raise ValueError("the LLM stream broke")

        async with _server(echo):
            with pytest.raises(ValueError, match="LLM stream broke"):
                [a async for a in _stream(failing())]

    async def test_a_close_before_the_final_flag_raises(self) -> None:
        async def truncate(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.recv()
            await ws.send(_audio_message())
            await ws.close()  # a normal close, without is_final

        async with _server(truncate):
            with pytest.raises(RuntimeError, match="before the end of the audio"):
                [a async for a in _stream(_texts("One.", "Two."))]


class _StreamingAI(AIProvider):
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    @property
    def model_name(self) -> str:
        return "mock-streaming"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="".join(self._tokens))

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token


class TestThroughTheVoiceChannel:
    async def test_a_streaming_ai_response_goes_over_the_socket(self) -> None:
        tts = ElevenLabsTTSProvider(
            _config(api_key="k", expressive=True, output_format="pcm_16000")
        )
        backend = MockVoiceBackend()
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
                None,
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x00\x01" * 10),
            ]
        )
        stt = MockSTTProvider(transcripts=["Hello"])
        channel = VoiceChannel(
            "voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig(vad=vad)
        )
        kit = RoomKit(stt=stt, voice=backend)
        kit.register_channel(channel)
        kit.register_channel(
            AIChannel("ai", provider=_StreamingAI(["[laughs] Sure thing. ", "Anything else?"]))
        )
        assert channel.supports_streaming_delivery is True

        socket = FakeSocket(pcm=True)
        connect, calls = _connect(socket)
        with patch("websockets.connect", connect):
            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice")
            await kit.attach_channel(room.id, "ai")
            session = await kit.connect_voice(room.id, "user", "voice")
            for data in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
                await backend.simulate_audio_received(session, AudioFrame(data=data))
            for _ in range(200):
                if {"close_socket": True} in socket.sent and backend.sent_audio:
                    break
                await asyncio.sleep(0.01)
            await kit.close()

        assert urlparse(calls[0][0]).path == "/v1/text-to-dialogue/stream-input"
        spoken = [m["inputs"][0]["text"] for m in socket.sent if "inputs" in m]
        assert "".join(spoken).split() == ["[laughs]", "Sure", "thing.", "Anything", "else?"]
        assert backend.sent_audio
