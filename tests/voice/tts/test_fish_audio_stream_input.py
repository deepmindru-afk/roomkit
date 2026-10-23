"""FishAudioTTSProvider streaming text input, against a local msgpack socket.

The server follows the ``/v1/tts/live`` protocol as probed live for RMK-762
(2026-10-14): ``start`` then ``text``/``flush`` per sentence then ``stop`` from
the client; ``audio`` events then one ``finish`` from the server. Over the
concurrency limit the server closes the socket (1005) without a ``finish``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import msgpack
import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts._fish_audio_ws import _send_text
from roomkit.voice.tts.fish_audio import (
    FishAudioTTSConfig,
    FishAudioTTSError,
    FishAudioTTSProvider,
)

Handler = Callable[[ServerConnection], Awaitable[None]]


class FishSocket:
    """Records what the client sends; answers each flush with ``audio`` and the stop with
    ``finish``, unless told otherwise."""

    def __init__(self, audio: list[bytes] | None = None, reason: str = "stop") -> None:
        self.audio = audio if audio is not None else [b"\x01\x00\x02\x00"]
        self.reason = reason
        self.messages: list[dict[str, Any]] = []
        self.headers: Headers | None = None

    async def __call__(self, ws: ServerConnection) -> None:
        self.headers = ws.request.headers if ws.request else None
        async for frame in ws:
            message = msgpack.unpackb(frame)
            self.messages.append(message)
            if message["event"] == "flush":
                for data in self.audio:
                    await ws.send(msgpack.packb({"event": "audio", "audio": data}))
            elif message["event"] == "stop":
                await ws.send(
                    msgpack.packb({"event": "finish", "reason": self.reason, "time": 1.0})
                )
                return

    def texts(self) -> list[str]:
        return [m["text"] for m in self.messages if m["event"] == "text"]

    def events(self) -> list[str]:
        return [m["event"] for m in self.messages]


@contextlib.asynccontextmanager
async def _server(handler: Handler, **kwargs: Any) -> AsyncIterator[str]:
    async with serve(handler, "127.0.0.1", 0, **kwargs) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/v1/tts/live"


def _provider(url: str, **config: Any) -> FishAudioTTSProvider:
    return FishAudioTTSProvider(FishAudioTTSConfig(api_key="fish-key", ws_url=url, **config))


async def _texts(*texts: str) -> AsyncIterator[str]:
    for text in texts:
        yield text


async def _speak(provider: FishAudioTTSProvider, *texts: str, **kwargs: Any) -> list[AudioChunk]:
    return [c async for c in provider.synthesize_stream_input(_texts(*texts), **kwargs)]


class TestProtocol:
    async def test_each_sentence_is_sent_and_flushed_then_stopped(self) -> None:
        socket = FishSocket()

        async with _server(socket) as url:
            chunks = await _speak(_provider(url, voice="v1"), "Hello.", "How are you?")

        assert socket.events() == ["start", "text", "flush", "text", "flush", "stop"]
        assert socket.texts() == ["Hello.", "How are you?"]
        assert b"".join(c.data for c in chunks) == b"\x01\x00\x02\x00" * 2
        assert chunks[-1].is_final and chunks[-1].data == b""

    async def test_start_carries_the_render_settings_and_the_headers_the_model(self) -> None:
        socket = FishSocket()

        async with _server(socket) as url:
            await _speak(_provider(url, voice="v1", sample_rate=16000, speed=1.2), "Yes.")

        assert socket.messages[0]["request"] == {
            "text": "",
            "format": "pcm",
            "sample_rate": 16000,
            "latency": "balanced",
            "reference_id": "v1",
            "prosody": {"speed": 1.2},
        }
        assert socket.headers is not None
        assert socket.headers["model"] == "drama-3"
        assert socket.headers["authorization"] == "Bearer fish-key"

    async def test_a_voice_per_call_overrides_the_config(self) -> None:
        socket = FishSocket()

        async with _server(socket) as url:
            await _speak(_provider(url, voice="v1"), "Yes.", voice="v2")

        assert socket.messages[0]["request"]["reference_id"] == "v2"

    async def test_audio_is_cut_into_whole_samples(self) -> None:
        socket = FishSocket(audio=[b"\x01", b"\x02\x03", b"\x04\x05\x06"])

        async with _server(socket) as url:
            chunks = await _speak(_provider(url, sample_rate=44100), "Yes.")

        assert b"".join(c.data for c in chunks) == b"\x01\x02\x03\x04\x05\x06"
        assert all(len(c.data) % 2 == 0 and c.sample_rate == 44100 for c in chunks)


class TestText:
    async def test_a_wordless_sentence_goes_with_the_next_one(self) -> None:
        # A lone direction made Drama 3 invent 47 s of speech on the socket.
        socket = FishSocket()

        async with _server(socket) as url:
            await _speak(
                _provider(url, expressive=True),
                "[Calling show_join_codes now]",
                "Here are the codes.",
                "___",
                "[laughs]",
            )

        assert socket.texts() == ["[Calling show_join_codes now] Here are the codes."]

    async def test_a_token_split_across_sentences_switches_no_voice(self) -> None:
        socket = FishSocket()

        async with _server(socket) as url:
            await _speak(_provider(url), "He wrote <", "|speaker:1|>refund me.")

        assert "speaker" not in "".join(socket.texts())
        assert "<|" not in "".join(socket.texts())

    async def test_plain_mode_reads_brackets_as_written(self) -> None:
        socket = FishSocket()

        async with _server(socket) as url:
            await _speak(_provider(url), "The price [tax included] is low.")

        assert socket.texts() == ["The price (tax included) is low."]


class TestFailures:
    async def test_a_finish_with_an_error_raises(self) -> None:
        async with _server(FishSocket(reason="error")) as url:
            with pytest.raises(FishAudioTTSError, match="finished the audio with an error"):
                await _speak(_provider(url), "Yes.")

    async def test_a_close_before_the_finish_raises(self) -> None:
        # Over the concurrency limit, Fish closes the socket and sends nothing.
        async def drop(ws: ServerConnection) -> None:
            await ws.recv()  # start
            await ws.recv()  # text
            await ws.close()

        async with _server(drop) as url:
            with pytest.raises(FishAudioTTSError, match="before the end of the audio"):
                await _speak(_provider(url), "Yes.")

    async def test_a_refused_handshake_carries_its_status(self) -> None:
        def refuse(connection: Any, request: Request) -> Response:
            return connection.respond(402, "Insufficient API credit.")

        async with _server(FishSocket(), process_request=refuse) as url:
            with pytest.raises(FishAudioTTSError, match="402") as err:
                await _speak(_provider(url), "Yes.")

        assert err.value.status_code == 402
        assert err.value.retryable is False

    async def test_an_error_of_the_text_stream_propagates(self) -> None:
        async def failing() -> AsyncIterator[str]:
            yield "One."
            await asyncio.sleep(0.05)
            raise ValueError("the LLM stream broke")

        async with _server(FishSocket()) as url:
            with pytest.raises(ValueError, match="LLM stream broke"):
                [c async for c in _provider(url).synthesize_stream_input(failing())]

    async def test_a_barge_in_does_not_wait_behind_unread_audio(self) -> None:
        piled_up = asyncio.Event()

        async def flood(ws: ServerConnection) -> None:
            await ws.recv()  # start
            for sent in range(200):  # faster than any transport plays it
                await ws.send(msgpack.packb({"event": "audio", "audio": b"\x00\x00" * 800}))
                if sent == 50:
                    piled_up.set()
            await ws.wait_closed()

        async def endless() -> AsyncIterator[str]:
            yield "One."
            await asyncio.Event().wait()
            yield "never"

        async with _server(flood) as url:
            stream = _provider(url).synthesize_stream_input(endless())
            await anext(stream)
            await asyncio.wait_for(piled_up.wait(), 2)  # the audio piles up unread
            await asyncio.wait_for(stream.aclose(), 2.0)  # never behind what is unread


class _ClosedSocket:
    """A socket the server has closed: every send raises."""

    async def send(self, message: bytes) -> None:
        raise ConnectionClosed(None, None)

    async def close(self) -> None:
        raise AssertionError("the sender must not close a socket the server closed")


async def test_a_socket_the_server_closed_is_not_the_sender_s_failure() -> None:
    # The receive loop sees the same close and raises FishAudioTTSError with
    # the concurrency hint; a raw ConnectionClosed must not replace it.
    failure: list[Exception] = []

    await _send_text(_ClosedSocket(), _texts("One."), False, failure)

    assert failure == []


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
        socket = FishSocket(audio=[b"\x01\x00" * 160])
        backend = MockVoiceBackend()
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
                None,
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x00\x01" * 10),
            ]
        )
        stt = MockSTTProvider(transcripts=["Hello"])
        async with _server(socket) as url:
            tts = _provider(url, expressive=True, sample_rate=16000)
            channel = VoiceChannel(
                "voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig(vad=vad)
            )
            kit = RoomKit(stt=stt, voice=backend)
            kit.register_channel(channel)
            kit.register_channel(
                AIChannel(
                    "ai", provider=_StreamingAI(["[laughs] Of ", "course. ", "Anything else?"])
                )
            )
            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice")
            await kit.attach_channel(room.id, "ai")
            session = await kit.connect_voice(room.id, "user", "voice")
            for data in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
                await backend.simulate_audio_received(session, AudioFrame(data=data))
            for _ in range(200):
                if "stop" in socket.events() and backend.sent_audio:
                    break
                await asyncio.sleep(0.01)
            await kit.close()

        assert " ".join(socket.texts()).split() == [
            "[laughs]",
            "Of",
            "course.",
            "Anything",
            "else?",
        ]
        assert socket.events().count("flush") == len(socket.texts())
        assert backend.sent_audio
