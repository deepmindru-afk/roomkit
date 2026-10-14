"""Tests for RealtimeAVBridge (voice/realtime/bridge.py)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from roomkit.video.events import VideoDetectionEvent
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.filter.mock_face_touch import MockFaceTouchFilter
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.bridge import RealtimeAVBridge
from roomkit.voice.realtime.mock import MockRealtimeAudioVideoProvider, MockRealtimeProvider


class TestResamplePcm:
    def test_same_rate_passthrough(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        pcm = np.zeros(160, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 16000, 16000)
        assert result == pcm

    def test_upsample(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        # 160 samples at 16kHz -> should become ~480 samples at 48kHz
        pcm = np.zeros(160, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 16000, 48000)
        # Each sample is 2 bytes (int16)
        result_samples = len(result) // 2
        assert result_samples == 480

    def test_downsample(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        # 480 samples at 48kHz -> should become ~160 samples at 16kHz
        pcm = np.zeros(480, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 48000, 16000)
        result_samples = len(result) // 2
        assert result_samples == 160

    def test_empty_input(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        result = resample_pcm(b"", 16000, 48000)
        assert result == b""


class TestRealtimeAVBridge:
    def test_constructor(self) -> None:
        from roomkit.voice.realtime.bridge import RealtimeAVBridge

        provider = MockRealtimeProvider()
        backend = MockVoiceBackend()
        bridge = RealtimeAVBridge(provider, backend)
        assert bridge._provider is provider
        assert bridge._backend is backend
        assert len(bridge._calls) == 0

    def test_add_video_tap(self) -> None:
        from roomkit.voice.realtime.bridge import RealtimeAVBridge

        provider = MockRealtimeProvider()
        backend = MockVoiceBackend()
        bridge = RealtimeAVBridge(provider, backend)
        tap = MagicMock()
        bridge.add_video_tap(tap)
        assert tap in bridge._video_taps

    async def test_close_empty(self) -> None:
        from roomkit.voice.realtime.bridge import RealtimeAVBridge

        provider = MockRealtimeProvider()
        backend = MockVoiceBackend()
        bridge = RealtimeAVBridge(provider, backend)
        await bridge.close()
        # Provider should be closed
        assert any(c.method == "close" for c in provider.calls)


def _frame() -> VideoFrame:
    return VideoFrame(data=b"\xff" * 3, codec="raw_rgb24", width=1, height=1)


def _detection() -> VideoDetectionEvent:
    return VideoDetectionEvent(session=None, kind="face_touch", labels=["cheek"], confidence=1.0)


async def _bridge_with_a_call(
    frames: int, **bridge_kwargs: Any
) -> tuple[RealtimeAVBridge, MockRealtimeAudioVideoProvider, VoiceSession]:
    provider = MockRealtimeAudioVideoProvider()
    filters = [MockFaceTouchFilter({i: [_detection()] for i in range(frames)})]
    bridge = RealtimeAVBridge(
        provider,
        MockVoiceBackend(),
        video_pipeline=VideoPipelineConfig(filters=filters),
        **bridge_kwargs,
    )
    session = await bridge.connect(
        VoiceSession(id="call-1", room_id="r", participant_id="p", channel_id="voice")
    )
    for _ in range(frames):
        await provider.simulate_video(session, _frame())
    return bridge, provider, session


class TestRealtimeAVBridgeVideoPipeline:
    """The bridge owns the video pipeline it builds: its filters' detections
    do not pile up, a call's end resets the call's state, close() closes it."""

    async def test_filter_detections_do_not_pile_up(self) -> None:
        bridge, _, session = await _bridge_with_a_call(frames=3)

        assert bridge._video_pipeline.drain_events(session.id) == []
        await bridge.close()

    async def test_a_call_end_resets_its_video_state(self) -> None:
        bridge, _, session = await _bridge_with_a_call(frames=1)

        await bridge.disconnect(session.id)

        assert session.id not in bridge._video_pipeline._filter_contexts
        assert session.id not in bridge._video_pipeline._sessions
        await bridge.close()

    async def test_close_closes_the_video_pipeline(self) -> None:
        bridge, _, _ = await _bridge_with_a_call(frames=1)

        await bridge.close()

        assert bridge._video_pipeline._closed


class TestRealtimeAVBridgeVideoDetections:
    """``on_video_detection`` receives what the video filters detect, each
    detection carrying the call it was seen on."""

    async def test_each_detection_reaches_the_callback_with_its_call(self) -> None:
        seen: list[VideoDetectionEvent] = []
        bridge, _, session = await _bridge_with_a_call(frames=2, on_video_detection=seen.append)

        assert [event.kind for event in seen] == ["face_touch", "face_touch"]
        assert all(event.session is session for event in seen)
        await bridge.close()

    async def test_an_async_callback_is_awaited(self) -> None:
        seen: list[VideoDetectionEvent] = []

        async def on_detection(event: VideoDetectionEvent) -> None:
            await asyncio.sleep(0)
            seen.append(event)

        bridge, _, _ = await _bridge_with_a_call(frames=1, on_video_detection=on_detection)
        for _ in range(5):
            await asyncio.sleep(0)

        assert len(seen) == 1
        await bridge.close()

    async def test_a_failing_callback_does_not_stop_the_video(self) -> None:
        def on_detection(event: VideoDetectionEvent) -> None:
            raise RuntimeError("consumer down")

        tap = MagicMock()
        provider = MockRealtimeAudioVideoProvider()
        bridge = RealtimeAVBridge(
            provider,
            MockVoiceBackend(),
            video_pipeline=VideoPipelineConfig(filters=[MockFaceTouchFilter({0: [_detection()]})]),
            on_video_detection=on_detection,
        )
        bridge.add_video_tap(tap)
        session = await bridge.connect(
            VoiceSession(id="call-1", room_id="r", participant_id="p", channel_id="voice")
        )

        await provider.simulate_video(session, _frame())

        tap.assert_called_once()
        await bridge.close()
