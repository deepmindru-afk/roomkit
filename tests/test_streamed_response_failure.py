"""A response that fails mid-stream is not handed again to the channel that streamed it.

RFC §12.2 step 13s (RMK-467): the text the response produced is stored,
reaches the streaming channel through the stream, as a completed response's
last row does, then the failure; every other channel gets it as an ordinary
event and ON_ERROR fires. Only a failure of the streaming channel itself
sends the text to it again, since it may not have rendered it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit, VoiceChannel
from roomkit.channels.base import Channel
from roomkit.channels.cli import CLIChannel
from roomkit.channels.websocket import StreamMessage, WebSocketChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider
from tests.test_framework import SimpleChannel

SAID = "Your table is booked. "


class _FailsMidAnswer(AIProvider):
    """Streams one sentence, then its vendor goes down."""

    @property
    def model_name(self) -> str:
        return "mock-fails-mid-answer"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:  # pragma: no cover
        return AIResponse(content="unused")

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(text=SAID)
        raise RuntimeError("ai down")


class _CountingTTS(TTSProvider):
    """Records what each path synthesizes: streamed sentences and standard replays."""

    def __init__(self) -> None:
        self.streamed: list[str] = []
        self.standard: list[str] = []

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.standard.append(text)
        yield AudioChunk(data=b"\x00\x00", sample_rate=16000)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for sentence in text_stream:
            self.streamed.append(sentence)
            yield AudioChunk(data=b"\x00\x00", sample_rate=16000)


class _BufferingChannel(SimpleChannel):
    """A transport that takes a streamed answer through the default deliver_stream."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True


class _BrokenTransport(_BufferingChannel):
    """Takes the first delta, then its own transport drops."""

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for chunk in text_stream:
            if isinstance(chunk, str):
                raise ConnectionError("socket gone")
        return ChannelOutput.empty()  # pragma: no cover


class _SwallowingTransport(_BufferingChannel):
    """Renders the stream as it comes and swallows whatever it raises."""

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        with contextlib.suppress(Exception):
            async for _chunk in text_stream:
                pass
        return ChannelOutput.empty()


class _VoiceBench:
    def __init__(self) -> None:
        backend = MockVoiceBackend()
        self.tts = _CountingTTS()
        self.kit = RoomKit(voice=backend)
        self.channel: Channel = VoiceChannel("stream", tts=self.tts, backend=backend)

    async def ready(self, room_id: str) -> None:
        await self.kit.join(room_id, "stream", participant_id="user")


class _WebSocketBench:
    """One client speaking the streaming protocol, one taking whole events."""

    def __init__(self) -> None:
        self.kit = RoomKit()
        self.channel: Channel = WebSocketChannel("stream")
        self.live: list[tuple[str, str | None]] = []
        self.plain: list[str | None] = []

    async def ready(self, room_id: str) -> None:
        async def live_event(_conn: str, event: RoomEvent) -> None:
            self.live.append(("event", _body(event)))

        async def live_stream(_conn: str, message: StreamMessage) -> None:
            self.live.append((message.type, None))

        async def plain_event(_conn: str, event: RoomEvent) -> None:
            self.plain.append(_body(event))

        assert isinstance(self.channel, WebSocketChannel)
        self.channel.register_connection(
            "live", live_event, room_id=room_id, stream_send_fn=live_stream
        )
        self.channel.register_connection("plain", plain_event, room_id=room_id)


class _CLIBench:
    def __init__(self) -> None:
        self.kit = RoomKit()
        self.channel: Channel = CLIChannel("stream", use_color=False)

    async def ready(self, room_id: str) -> None:
        return None


class _BufferingBench:
    def __init__(self) -> None:
        self.kit = RoomKit()
        self.channel: Channel = _BufferingChannel("stream")

    async def ready(self, room_id: str) -> None:
        return None


Bench = _VoiceBench | _WebSocketBench | _CLIBench | _BufferingBench


@dataclass
class _Turn:
    result: InboundResult
    rows: list[RoomEvent]
    sms: list[str | None]
    errors: list[RoomEvent]


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


async def _run(kit: RoomKit, streaming: Channel, ready: Any = None) -> _Turn:
    """One turn the user starts on *streaming*, which also renders the answer."""
    sms = SimpleChannel("sms1")
    kit.register_channel(streaming)
    kit.register_channel(sms)
    kit.register_channel(AIChannel("ai-1", provider=_FailsMidAnswer()))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "stream")
    await kit.attach_channel(room.id, "sms1")
    await kit.attach_channel(room.id, "ai-1", category=ChannelCategory.INTELLIGENCE)
    if ready is not None:
        await ready(room.id)
    errors: list[RoomEvent] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="stream", sender_id="user", content=TextContent(body="Book a table")
        ),
        room_id=room.id,
    )
    await asyncio.sleep(0.05)  # ON_ERROR runs after the room lock is released
    events = await kit.store.list_events(room.id, offset=0, limit=50)
    await kit.close()
    return _Turn(
        result=result,
        rows=[e for e in events if e.source.channel_id == "ai-1"],
        sms=[_body(e) for e in sms.delivered if e.source.channel_id == "ai-1"],
        errors=errors,
    )


async def _bench_turn(bench: Bench) -> _Turn:
    return await _run(bench.kit, bench.channel, bench.ready)


class TestTheChannelThatStreamedIsNotHandedTheTextAgain:
    async def test_a_voice_channel_does_not_speak_it_again(self) -> None:
        bench = _VoiceBench()
        await _bench_turn(bench)

        assert bench.tts.streamed == [SAID.strip()]
        assert bench.tts.standard == []

    async def test_a_websocket_client_gets_the_row_inside_the_stream_and_never_after(
        self,
    ) -> None:
        bench = _WebSocketBench()
        await _bench_turn(bench)

        # As a completed response: the row inline, then the end of the stream.
        assert bench.live == [
            ("stream_start", None),
            ("stream_chunk", None),
            ("event", SAID),
            ("stream_error", None),
        ]
        assert bench.plain == [SAID]

    async def test_the_cli_does_not_print_it_again(self, capsys: pytest.CaptureFixture) -> None:
        await _bench_turn(_CLIBench())

        assert capsys.readouterr().out.count(SAID.strip()) == 1

    async def test_a_buffering_channel_delivers_what_it_buffered_once(self) -> None:
        bench = _BufferingBench()
        await _bench_turn(bench)

        assert isinstance(bench.channel, _BufferingChannel)
        assert [_body(e) for e in bench.channel.delivered if _body(e) == SAID] == [SAID]


@pytest.mark.parametrize(
    "bench_type",
    [_VoiceBench, _WebSocketBench, _CLIBench, _BufferingBench],
    ids=["voice", "websocket", "cli", "buffering"],
)
async def test_the_others_get_the_text_and_on_error_fires(bench_type: type[Bench]) -> None:
    turn = await _bench_turn(bench_type())

    assert turn.sms == [SAID]
    assert len(turn.errors) == 1
    assert str(turn.result.error) == "ai down"
    (row,) = turn.rows
    assert _body(row) == SAID
    assert "cancelled" not in row.metadata


async def test_a_streaming_channel_that_fails_itself_gets_the_text_again() -> None:
    """Its transport dropped while rendering: what it was handed may be unseen."""
    channel = _BrokenTransport("stream")
    turn = await _run(RoomKit(), channel)

    assert [_body(e) for e in channel.delivered if e.source.channel_id == "ai-1"] == [SAID]
    assert turn.sms == [SAID]
    assert isinstance(turn.result.error, ConnectionError)
    assert len(turn.errors) == 1


async def test_a_channel_that_swallows_the_failure_does_not_make_the_turn_a_success() -> None:
    channel = _SwallowingTransport("stream")
    turn = await _run(RoomKit(), channel)

    assert [e for e in channel.delivered if e.source.channel_id == "ai-1"] == []
    assert turn.sms == [SAID]
    assert str(turn.result.error) == "ai down"
    assert len(turn.errors) == 1
    (row,) = turn.rows
    assert "cancelled" not in row.metadata
