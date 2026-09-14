"""Audio continuity and teardown at the realtime RTP track boundary."""

from __future__ import annotations

import asyncio
from fractions import Fraction
from unittest.mock import patch

import pytest

pytest.importorskip("roomkit.webrtc")
import numpy as np
from aiortc.mediastreams import MediaStreamError

from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime._fastrtc_playback import _PCMPlayback
from roomkit.voice.realtime.fastrtc_transport import FastRTCRealtimeTransport, _PassthroughHandler


def pcm(count: int, value: int = 1000) -> bytes:
    return np.full(count, value, dtype="<i2").tobytes()


async def test_fragment_boundaries_preserve_samples_and_continuous_timestamps():
    playback = _PCMPlayback(24000)
    original = np.arange(1920, dtype=np.int16)
    # Deliberately unrelated provider and RTP frame sizes.
    for part in np.split(original, [137, 737, 1401]):
        await playback.write(part.tobytes())
    playback.end_response()
    frames = [await playback.recv() for _ in range(4)]
    actual = np.concatenate([frame.to_ndarray().flatten() for frame in frames])
    np.testing.assert_array_equal(actual[120:], original[120:])  # 5 ms initial ramp only
    assert [frame.pts for frame in frames] == [0, 480, 960, 1440]
    assert all(frame.time_base == Fraction(1, 24000) for frame in frames)
    assert playback.buffered_ms == 0
    assert playback.underruns == 0


async def test_startup_reserve_and_short_response_tail():
    playback = _PCMPlayback(24000)
    await playback.write(pcm(100))
    assert not (await playback.recv()).to_ndarray().any()
    assert playback.buffered_ms > 0
    playback.end_response()
    samples = (await playback.recv()).to_ndarray().flatten()
    assert samples[99] == 1000
    assert samples[219] == 0  # partial final frame fades then pads with silence
    assert not samples[220:].any()
    assert playback.buffered_ms == 0
    assert playback.underruns == 0


async def test_reserve_absorbs_a_delayed_provider_fragment():
    playback = _PCMPlayback(24000)
    await playback.write(pcm(960))
    await playback.recv()
    second = asyncio.create_task(playback.recv())
    # A fragment 10 ms late is covered by the remaining 20 ms reserve.
    await asyncio.sleep(0.01)
    await playback.write(pcm(480))
    assert (await second).to_ndarray().min() == 1000
    assert (await playback.recv()).to_ndarray().min() == 1000
    assert playback.underruns == 0


async def test_interruption_flushes_audio_while_recv_is_sleeping():
    playback = _PCMPlayback(24000)
    await playback.write(pcm(1440))
    await playback.recv()
    pending = asyncio.create_task(playback.recv())
    await asyncio.sleep(0)
    playback.clear()
    # Only a 5 ms fade from the last played sample may survive the interruption.
    samples = (await pending).to_ndarray().flatten()
    assert not samples[120:].any()
    await playback.write(pcm(960, -2000))
    assert (await playback.recv()).to_ndarray().flatten()[120:].max() == -2000


async def test_bounded_queue_cancels_old_blocked_writer_on_clear():
    playback = _PCMPlayback(24000)
    writer = asyncio.create_task(playback.write(pcm(24000 * 6)))
    await asyncio.sleep(0)
    assert not writer.done()
    assert playback.buffered_ms == 5000
    playback.clear()
    await asyncio.wait_for(writer, 0.1)
    assert playback.buffered_ms == 0


async def test_shutdown_wakes_reader_and_blocked_writer():
    playback = _PCMPlayback(24000)
    await playback.recv()
    reader = asyncio.create_task(playback.recv())
    writer = asyncio.create_task(playback.write(pcm(24000 * 6)))
    await asyncio.sleep(0)
    playback.close()
    with pytest.raises(MediaStreamError):
        await asyncio.wait_for(reader, 0.1)
    await asyncio.wait_for(writer, 0.1)
    assert playback.buffered_ms == 0


@pytest.mark.parametrize("default", ["webrtc", "datachannel"])
async def test_negotiated_audio_path_isolated_between_clients(default):
    transport = FastRTCRealtimeTransport(audio_transport=default)
    handlers = []
    for mode in ("webrtc", "datachannel"):
        handler = _PassthroughHandler(transport, input_sample_rate=16000, output_sample_rate=24000)
        handlers.append(handler)
        session = VoiceSession(
            id=mode,
            room_id="r",
            participant_id=mode,
            channel_id="v",
            metadata={"audio_transport": mode},
        )
        # recv can start before asynchronous authentication/session binding.
        pending = asyncio.create_task(handler._recv_audio_frame())
        await asyncio.sleep(0)
        transport._register_handler(mode, handler)
        await transport.accept(session, mode)
        with patch.object(handler, "send_audio_direct") as legacy:
            await transport.send_audio(session, pcm(960))
            if mode == "webrtc":
                assert (await asyncio.wait_for(pending, 0.2)).to_ndarray().any()
                legacy.assert_not_called()
            else:
                legacy.assert_called_once()
                await asyncio.sleep(0)
                assert not pending.done()
        await transport.disconnect(session)
        if mode == "datachannel":
            with pytest.raises(MediaStreamError):
                await asyncio.wait_for(pending, 0.1)
    await transport.close()


async def test_real_peer_connection_negotiates_opus_and_receives_rtp_audio():
    """Exercise Stream's SDP/track factory, real DTLS/RTP and Opus decoding."""
    from aiortc import AudioStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from fastapi import FastAPI

    from roomkit.voice.realtime.fastrtc_transport import mount_fastrtc_realtime

    transport = FastRTCRealtimeTransport()
    mount_fastrtc_realtime(FastAPI(), transport, rtc_configuration={"iceServers": []})
    stream = transport._stream
    assert stream is not None
    session = VoiceSession(id="s", room_id="r", participant_id="p", channel_id="v")

    async def connected(webrtc_id):
        await transport.accept(session, webrtc_id)
        wave = (np.sin(np.arange(24000) * 2 * np.pi * 440 / 24000) * 8000).astype("<i2")
        await transport.send_audio(session, wave.tobytes())
        transport.end_of_response(session)
        await transport.send_message(session, {"type": "session_started"})

    transport.on_client_connected(connected)
    client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    client.addTrack(AudioStreamTrack())
    dc = client.createDataChannel("text")
    messages = []
    dc.on("message", messages.append)
    tracks = []
    client.on("track", tracks.append)
    try:
        await client.setLocalDescription(await client.createOffer())
        answer = await stream.handle_offer(
            {"sdp": client.localDescription.sdp, "type": "offer", "webrtc_id": "rtc"},
            lambda *_: None,
        )
        # The first audio payload in the answer resolves to Opus/48k.
        audio_line = next(
            line for line in answer["sdp"].splitlines() if line.startswith("m=audio")
        )
        payload = audio_line.split()[3]
        assert f"a=rtpmap:{payload} opus/48000/2" in answer["sdp"]
        await client.setRemoteDescription(RTCSessionDescription(**answer))
        assert len(tracks) == 1
        async with asyncio.timeout(5):
            frames = [await tracks[0].recv() for _ in range(12)]
        assert any(np.abs(frame.to_ndarray().astype(float)).mean() > 1000 for frame in frames)
        assert any('"session_started"' in message for message in messages)
        assert all('"media"' not in message for message in messages)
        callback = stream.connections["rtc"][0]
        assert not hasattr(callback, "decode_task")  # no hidden second output queue
    finally:
        await client.close()
        await asyncio.gather(*(pc.close() for pc in list(stream.pcs.values())))
        await transport.close()


async def test_close_wakes_output_before_session_binding():
    transport = FastRTCRealtimeTransport()
    handler = _PassthroughHandler(transport, input_sample_rate=16000, output_sample_rate=24000)
    transport._register_handler("unbound", handler)
    reader = asyncio.create_task(handler._recv_audio_frame())
    await asyncio.sleep(0)
    await transport.close()
    with pytest.raises(MediaStreamError):
        await asyncio.wait_for(reader, 0.1)
    assert not transport._handlers


async def test_partial_starvation_is_counted_once_before_rebuffering():
    playback = _PCMPlayback(24000)
    await playback.write(pcm(1000))
    await playback.recv()
    await playback.recv()
    await playback.recv()  # only 40 real samples, then a fade and silence
    assert playback.underruns == 1
    await playback.recv()
    assert playback.underruns == 1
