"""Tests for the inbound DSP offload — FIFO per stream, parallel across.

The unit tests exercise :class:`InboundFrameOffload` directly; the
integration tests prove a VoiceChannel or RealtimeVoiceChannel configured
with ``inbound_dsp_threads`` behaves as it does inline (the streaming STT,
the realtime provider's audio feed), the stages just running off the event
loop. Each of those runs on both paths: inline (``None``) and the pool.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from roomkit import HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider
from roomkit.voice.pipeline.offload import InboundFrameOffload
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

# Inline, then the DSP pool: a behaviour of the channel holds on both paths.
_PATHS = pytest.mark.parametrize("threads", [None, 2], ids=["inline", "dsp-pool"])

_FRAME = b"\x01\x00" * 320  # 20 ms at 16 kHz mono


async def _until(condition: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Whether *condition* holds within *timeout*, polled on the loop."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


class _StreamingSTT(MockSTTProvider):
    """A streaming STT that keeps what its streams heard apart from batch calls."""

    def __init__(self) -> None:
        super().__init__(transcripts=["batch"], streaming=True)
        self.streamed: list[bytes] = []
        self.chunk_heard = asyncio.Event()
        self.streams_ended = 0

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for chunk in audio_stream:
            self.streamed.append(chunk.data)
            self.chunk_heard.set()
        self.streams_ended += 1
        yield TranscriptionResult(text="streamed", is_final=True)


class _SlowDenoiser(MockDenoiserProvider):
    """A stage that takes *seconds* per frame, as a heavy model would."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        time.sleep(self._seconds)
        return super().process(frame, stream)


class TestInboundFrameOffload:
    def test_one_stream_is_fifo_whatever_the_pool_size(self) -> None:
        # Queue bound above the burst: this test is about ordering, not drops.
        offload = InboundFrameOffload(4, max_queued_frames=1000)
        seen: list[int] = []
        for i in range(200):
            offload.submit("s1", seen.append, i)
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()
        assert seen == list(range(200))

    def test_streams_run_in_parallel(self) -> None:
        """A blocked stream must not hold another stream's frames hostage."""
        offload = InboundFrameOffload(2)
        gate = threading.Event()
        s2_done = threading.Event()

        offload.submit("s1", gate.wait, 5.0)
        offload.submit("s2", s2_done.set)
        try:
            assert s2_done.wait(2.0), "s2 waited behind s1's blocked frame"
        finally:
            gate.set()
            offload.shutdown()

    def test_a_full_stream_queue_drops_its_oldest_frames(self) -> None:
        offload = InboundFrameOffload(1, max_queued_frames=4)
        gate = threading.Event()
        seen: list[int] = []

        offload.submit("s1", gate.wait, 5.0)  # occupies the drainer
        for i in range(1, 10):  # 9 more; queue keeps the newest 4
            offload.submit("s1", seen.append, i)
        gate.set()
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()

        assert seen == [6, 7, 8, 9]
        assert offload.dropped("s1") == 5

    def test_release_drops_a_gone_streams_queue(self) -> None:
        offload = InboundFrameOffload(1)
        gate = threading.Event()
        seen: list[int] = []

        offload.submit("s1", gate.wait, 5.0)
        offload.submit("s1", seen.append, 1)
        offload.release("s1")
        gate.set()
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()
        assert seen == []

    def test_a_failing_frame_does_not_stall_the_stream(self) -> None:
        offload = InboundFrameOffload(1)
        seen: list[int] = []

        def boom() -> None:
            raise RuntimeError("stage blew up")

        offload.submit("s1", boom)
        offload.submit("s1", seen.append, 1)
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()
        assert seen == [1]

    def test_submit_after_shutdown_is_a_noop(self) -> None:
        offload = InboundFrameOffload(1)
        offload.shutdown()
        offload.submit("s1", lambda: None)  # must not raise

    def test_shutdown_finishes_the_queue_and_refuses_what_comes_after(self) -> None:
        offload = InboundFrameOffload(1)
        gate = threading.Event()
        seen: list[int] = []
        offload.submit("s1", gate.wait, 5.0)
        offload.submit("s1", seen.append, 1)
        stopper = threading.Thread(target=offload.shutdown)
        stopper.start()
        time.sleep(0.05)  # shutdown is now waiting on the blocked frame
        offload.submit("s1", seen.append, 2)
        gate.set()
        stopper.join(timeout=5.0)

        assert seen == [1]


class TestVoiceChannelWithOffload:
    async def test_the_frame_to_stt_path_runs_through_the_pool(self) -> None:
        stt = MockSTTProvider(transcripts=["Hello"])
        backend = MockVoiceBackend()
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x01\x00" * 80),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad, inbound_dsp_threads=2)

        kit = RoomKit(voice=backend)
        channel = VoiceChannel(
            "voice-1", stt=stt, tts=MockTTSProvider(), backend=backend, pipeline=pipeline
        )
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.connect_voice(room.id, "user-1", "voice-1")

        assert channel._inbound_offload is not None

        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))

        # The DSP ran on the pool; wait for it, then for the scheduled
        # speech-end coroutine on the loop.
        assert await asyncio.to_thread(channel._inbound_offload.wait_idle, timeout=5.0)
        await asyncio.sleep(0.15)

        assert len(stt.calls) >= 1
        await kit.close()

    async def test_without_the_knob_processing_stays_inline(self) -> None:
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig(vad=MockVADProvider(events=[]))
        kit = RoomKit(voice=backend)
        channel = VoiceChannel(
            "voice-1",
            stt=MockSTTProvider(transcripts=[]),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=pipeline,
        )
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.connect_voice(room.id, "user-1", "voice-1")

        assert channel._inbound_offload is None
        await kit.close()


class TestStreamingSTTBehindThePool:
    """The channel's STT stream is loop code: the pool must not reach into it."""

    @_PATHS
    async def test_a_segment_is_transcribed_by_its_stream(self, threads: int | None) -> None:
        speech_frames = 6
        pre_roll = b"\x02\x00" * 320
        events: list[VADEvent | None] = [
            VADEvent(type=VADEventType.SPEECH_START, audio_bytes=pre_roll)
        ]
        events += [None] * (speech_frames - 1)
        events.append(VADEvent(type=VADEventType.SPEECH_END, audio_bytes=_FRAME * speech_frames))
        stt = _StreamingSTT()
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)
        channel = VoiceChannel(
            "voice-1",
            stt=stt,
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(
                vad=MockVADProvider(events=events), inbound_dsp_threads=threads
            ),
        )
        kit.register_channel(channel)
        texts: list[str] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event: Any, ctx: Any) -> None:
            texts.append(event.text)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.join(room.id, "voice-1", participant_id="user-1")
        for _ in range(speech_frames + 1):
            await backend.simulate_audio_received(session, AudioFrame(data=_FRAME))

        assert await _until(lambda: texts == ["streamed"]), texts
        # The pre-roll, then every speech frame (SPEECH_START's included,
        # SPEECH_END's not): the stream heard the whole utterance.
        assert b"".join(stt.streamed) == pre_roll + _FRAME * speech_frames
        assert stt.streams_ended == 1
        assert stt.calls == []  # no batch fallback
        await kit.close()

    @_PATHS
    async def test_continuous_stt_hears_a_chunk_as_it_lands(self, threads: int | None) -> None:
        # A chunk queued from another thread does not wake the loop: the
        # stream hears it at the loop's next wake-up for another reason. The
        # input-level hook is one (at most every 100 ms), so the first chunk
        # gets through; the second lands inside that window, while the loop
        # sleeps with nothing else to do but the wait's own timeout.
        stt = _StreamingSTT()
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)
        channel = VoiceChannel(
            "voice-1",
            stt=stt,
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(
                denoiser=_SlowDenoiser(0.02), inbound_dsp_threads=threads
            ),
        )
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.join(room.id, "voice-1", participant_id="user-1")
        loop = asyncio.get_running_loop()

        await backend.simulate_audio_received(session, AudioFrame(data=_FRAME * 5))
        await asyncio.wait_for(stt.chunk_heard.wait(), timeout=1.0)
        stt.chunk_heard.clear()
        sent_at = loop.time()
        await backend.simulate_audio_received(session, AudioFrame(data=_FRAME * 5))
        await asyncio.wait_for(stt.chunk_heard.wait(), timeout=1.0)

        assert loop.time() - sent_at < 0.5
        assert stt.streamed == [_FRAME * 5, _FRAME * 5]
        await kit.close()


class TestRealtimeVoiceChannelBehindThePool:
    """The realtime provider hears the pipeline's audio whichever thread ran it."""

    async def _start(
        self, threads: int | None, vad: MockVADProvider | None = None
    ) -> tuple[RealtimeVoiceChannel, MockRealtimeProvider, MockRealtimeTransport, Any]:
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=provider,
            transport=transport,
            pipeline=AudioPipelineConfig(vad=vad, inbound_dsp_threads=threads),
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-1")
        session = await channel.start_session(room.id, "user-1", "fake-ws")
        return channel, provider, transport, session

    @_PATHS
    async def test_every_frame_reaches_the_provider(self, threads: int | None) -> None:
        channel, provider, transport, session = await self._start(threads)
        for _ in range(10):
            await transport.simulate_client_audio(session, _FRAME)

        assert await _until(lambda: len(provider.sent_audio) == 10), provider.sent_audio
        assert b"".join(audio for _, audio in provider.sent_audio) == _FRAME * 10
        await channel.close()

    @_PATHS
    async def test_speech_start_clears_the_clients_audio(self, threads: int | None) -> None:
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        channel, _, transport, session = await self._start(threads, vad)
        await transport.simulate_client_audio(session, _FRAME)

        def cleared() -> bool:
            return any(m.get("type") == "clear_audio" for _, m in transport.sent_messages)

        assert await _until(cleared), transport.sent_messages
        await channel.close()


class TestCloseWithAFrameInFlight:
    """A frame still in the stages when the channel closes leaves nothing behind.

    The stage is slow, so on the pool the SPEECH_START frame is still being
    processed when ``close()`` starts; its callbacks must not open what the
    teardown has already swept.
    """

    @_PATHS
    async def test_voice_channel_leaves_no_stt_stream(self, threads: int | None) -> None:
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)
        channel = VoiceChannel(
            "voice-1",
            stt=_StreamingSTT(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(
                vad=MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)]),
                denoiser=_SlowDenoiser(0.05),
                inbound_dsp_threads=threads,
            ),
        )
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.join(room.id, "voice-1", participant_id="user-1")
        await backend.simulate_audio_received(session, AudioFrame(data=_FRAME))

        await channel.close()
        await asyncio.sleep(0.1)

        assert channel._stt_streams == {}
        alive = [t.get_name() for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert not [name for name in alive if name.startswith("stt_stream:")], alive
        await kit.close()

    @_PATHS
    async def test_realtime_provider_hears_nothing_after_disconnect(
        self, threads: int | None
    ) -> None:
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=provider,
            transport=transport,
            pipeline=AudioPipelineConfig(
                vad=MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)]),
                denoiser=_SlowDenoiser(0.05),
                inbound_dsp_threads=threads,
            ),
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-1")
        session = await channel.start_session(room.id, "user-1", "fake-ws")
        await transport.simulate_client_audio(session, _FRAME)

        await channel.close()
        await asyncio.sleep(0.1)

        calls = [call.method for call in provider.calls]
        assert calls[calls.index("disconnect") + 1 :] == ["close"], calls
        types = [message.get("type") for _, message in transport.sent_messages]
        assert "clear_audio" not in types[types.index("session_ended") :], types
