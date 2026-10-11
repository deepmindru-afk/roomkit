"""A realtime pipeline running at its own rate hands the provider its rate back.

A pipeline whose contract runs its stages at 16 kHz (a diarization model's
rate, say) emits processed frames at 16 kHz.  The channel must deliver them
at the provider's input rate (Section 12.4: the pipeline normalizes the
format for the provider), not forward 16 kHz bytes as 24 kHz audio.
"""

from __future__ import annotations

import asyncio

from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.pipeline.config import AudioFormat, AudioPipelineConfig, AudioPipelineContract
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

# 20 ms of 16-bit mono PCM.
_FRAME_24K = b"\x10\x00" * 480
_BYTES_PER_20MS_24K = 960
_BYTES_PER_20MS_16K = 640


def _channel(
    provider: MockRealtimeProvider, transport: MockRealtimeTransport
) -> RealtimeVoiceChannel:
    contract = AudioPipelineContract(
        transport_inbound_format=AudioFormat(sample_rate=24000),
        transport_outbound_format=AudioFormat(sample_rate=24000),
        internal_format=AudioFormat(sample_rate=16000),
    )
    return RealtimeVoiceChannel(
        "rt-rate",
        provider=provider,
        transport=transport,
        input_sample_rate=24000,
        output_sample_rate=24000,
        pipeline=AudioPipelineConfig(contract=contract),
    )


async def _sent_bytes(provider: MockRealtimeProvider, session_id: str, frames: int) -> int:
    for _ in range(200):
        sent = sum(len(audio) for sid, audio in provider.sent_audio if sid == session_id)
        if sent >= frames * _BYTES_PER_20MS_16K:
            await asyncio.sleep(0.02)  # let any in-flight resample land
            break
        await asyncio.sleep(0.005)
    return sum(len(audio) for sid, audio in provider.sent_audio if sid == session_id)


async def test_a_16khz_pipeline_reaches_a_24khz_provider_at_24khz() -> None:
    provider = MockRealtimeProvider()
    transport = MockRealtimeTransport()
    channel = _channel(provider, transport)
    session = await channel.start_session("room-1", "user-1", "conn")

    for _ in range(10):
        await transport.simulate_client_audio(session, _FRAME_24K)
    sent = await _sent_bytes(provider, session.id, 10)

    # 200 ms of audio at 24 kHz: within a frame's worth of resampler slack.
    assert abs(sent - 10 * _BYTES_PER_20MS_24K) <= _BYTES_PER_20MS_24K, sent
    await channel.end_session(session)


async def test_frames_held_during_the_handshake_reach_the_provider_at_its_rate() -> None:
    connecting = asyncio.Event()
    release = asyncio.Event()
    captured: list = []

    class SlowProvider(MockRealtimeProvider):
        async def connect(self, session, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(session)
            connecting.set()
            await release.wait()
            await super().connect(session, **kwargs)

    provider = SlowProvider()
    transport = MockRealtimeTransport()
    channel = _channel(provider, transport)
    start = asyncio.create_task(channel.start_session("room-1", "user-1", "conn"))
    await connecting.wait()

    for _ in range(5):
        await transport.simulate_client_audio(captured[0], _FRAME_24K)
    for _ in range(200):  # the pipeline processes them into the pre-connect buffer
        if len(channel._preconnect_audio.get(captured[0].id, [])) >= 5:
            break
        await asyncio.sleep(0.005)
    release.set()
    session = await start
    sent = await _sent_bytes(provider, session.id, 5)

    assert abs(sent - 5 * _BYTES_PER_20MS_24K) <= _BYTES_PER_20MS_24K, sent
    await channel.end_session(session)
