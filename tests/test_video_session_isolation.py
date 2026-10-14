"""Video session isolation and shared channel resource ownership."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.av import AudioVideoChannel
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.video import VideoChannel
from roomkit.models.channel import ChannelBinding
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.base import VideoSession
from roomkit.video.events import VideoDetectionEvent
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.decoder.base import VideoDecoderProvider
from roomkit.video.pipeline.engine import VideoPipeline
from roomkit.video.pipeline.filter.censor import CensorVideoFilter
from roomkit.video.pipeline.filter.mock_face_touch import MockFaceTouchFilter
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import VisionProvider, VisionResult
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeAudioVideoProvider, MockRealtimeTransport


def frame() -> VideoFrame:
    return VideoFrame(data=b"\xff" * 3, codec="raw_rgb24", width=1, height=1)


def realtime_av_channel(
    config: VideoPipelineConfig,
    *,
    vision: AsyncMock | None = None,
    provider: MockRealtimeAudioVideoProvider | None = None,
) -> RealtimeAudioVideoChannel:
    return RealtimeAudioVideoChannel(
        "rv",
        provider=provider or MockRealtimeAudioVideoProvider(),
        transport=MockRealtimeTransport(),
        video_pipeline=config,
        vision=vision,
    )


def test_censor_grace_and_reset_do_not_cross_sessions() -> None:
    pipeline = VideoPipeline(
        VideoPipelineConfig(filters=[CensorVideoFilter({"private"}, grace_frames=2)])
    )
    pipeline.update_filter_context("a", VisionResult(description="private", labels=["private"]))
    original = frame()
    assert pipeline.process_inbound("a", original).data == b"\0" * 3
    assert pipeline.process_inbound("b", original) is original
    pipeline.reset("b")
    pipeline.update_filter_context("a", VisionResult(description="clear", labels=[]))
    assert pipeline.process_inbound("a", original).data == b"\0" * 3
    assert pipeline.process_inbound("a", original).data == b"\0" * 3
    assert pipeline.process_inbound("a", original) is original
    pipeline.reset("a")
    assert not pipeline._filter_contexts


@pytest.mark.parametrize("kind", ["video", "av", "realtime_av"])
@pytest.mark.parametrize("vision_mode", ["pipeline", "shared", "separate"])
async def test_channel_closes_pipeline_and_each_vision_once(kind: str, vision_mode: str) -> None:
    decoder = MagicMock(spec=VideoDecoderProvider)
    vision = AsyncMock()
    direct = (
        vision if vision_mode == "shared" else AsyncMock() if vision_mode == "separate" else None
    )
    config = VideoPipelineConfig(decoder=decoder, vision=vision)
    if kind == "video":
        channel = VideoChannel("v", backend=MockVideoBackend(), pipeline=config, vision=direct)
    elif kind == "av":
        channel = AudioVideoChannel(
            "av", backend=MockVoiceBackend(), video_pipeline=config, vision=direct
        )
    else:
        channel = realtime_av_channel(config, vision=direct)
    await channel.close()
    await channel.close()
    decoder.close.assert_called_once()
    vision.close.assert_awaited_once()
    if direct is not None:
        direct.close.assert_awaited_once()


@pytest.mark.parametrize("kind", ["video", "av"])
async def test_unbind_removes_only_its_video_context(kind: str) -> None:
    config = VideoPipelineConfig(filters=[CensorVideoFilter({"private"}, grace_frames=2)])
    if kind == "video":
        channel = VideoChannel("v", backend=MockVideoBackend(), pipeline=config)
        session = VideoSession(id="a", room_id="r", channel_id="v", participant_id="p")
    else:
        channel = AudioVideoChannel("v", backend=MockVoiceBackend(), video_pipeline=config)
        session = VoiceSession(id="a", room_id="r", channel_id="v", participant_id="p")
    try:
        channel.bind_session(
            session, "r", ChannelBinding(room_id="r", channel_id="v", channel_type="video")
        )
        pipeline = channel._video_pipeline
        pipeline.update_filter_context("a", VisionResult(description="a", labels=[]))
        pipeline.update_filter_context("b", VisionResult(description="b", labels=["private"]))
        channel.unbind_session(session)
        assert "a" not in pipeline._filter_contexts
        assert "b" in pipeline._filter_contexts
    finally:
        await channel.close()


async def test_pipeline_async_close_includes_vision_even_after_stage_failure() -> None:
    decoder = MagicMock(spec=VideoDecoderProvider)
    decoder.close.side_effect = RuntimeError("bad decoder close")
    vision = AsyncMock()
    pipeline = VideoPipeline(VideoPipelineConfig(decoder=decoder, vision=vision))
    with pytest.raises(ExceptionGroup):
        await pipeline.aclose()
    vision.close.assert_awaited_once()
    assert pipeline.process_inbound("a", frame()) is None


async def test_av_dispatches_and_drains_detection_events() -> None:
    from roomkit.video.events import VideoDetectionEvent

    detection = VideoDetectionEvent(
        session=None, kind="face_touch", labels=["cheek"], confidence=1.0
    )
    channel = AudioVideoChannel(
        "av",
        backend=MockVoiceBackend(),
        video_pipeline=VideoPipelineConfig(filters=[MockFaceTouchFilter({0: [detection]})]),
    )
    session = VideoSession(id="a", room_id="r", channel_id="av", participant_id="p")
    binding = ChannelBinding(room_id="r", channel_id="av", channel_type="audio_video")
    channel._session_bindings[session.id] = ("r", binding)
    hook = AsyncMock()
    channel._fire_video_detection_hook = hook
    try:
        channel._on_video_received(session, frame())

        await asyncio.gather(*channel._scheduled_tasks)
        hook.assert_awaited_once_with(session, detection, "r")
        assert channel._video_pipeline.drain_events(session.id) == []
    finally:
        await channel.close()


async def test_realtime_av_session_end_removes_only_its_video_context() -> None:
    channel = realtime_av_channel(
        VideoPipelineConfig(filters=[CensorVideoFilter({"private"}, grace_frames=2)])
    )
    try:
        session = await channel.start_session("r", "p", "ws")
        pipeline = channel._video_pipeline
        pipeline.update_filter_context(session.id, VisionResult(description="a", labels=[]))
        pipeline.update_filter_context("b", VisionResult(description="b", labels=["private"]))
        await channel.end_session(session)
        assert session.id not in pipeline._filter_contexts
        assert "b" in pipeline._filter_contexts
    finally:
        await channel.close()


async def test_realtime_av_frame_during_session_end_leaves_no_video_state() -> None:
    # The session's video state is reset before the provider disconnects: a
    # frame arriving meanwhile built it again, and nothing reset it after.
    provider = MockRealtimeAudioVideoProvider()
    channel = realtime_av_channel(
        VideoPipelineConfig(filters=[CensorVideoFilter({"private"})]), provider=provider
    )
    disconnect = provider.disconnect

    async def late_frame_then_disconnect(session: VoiceSession) -> None:
        await provider.simulate_video(session, frame())
        await disconnect(session)

    provider.disconnect = late_frame_then_disconnect  # type: ignore[method-assign]
    try:
        session = await channel.start_session("r", "p", "ws")
        await channel.end_session(session)
        assert session.id not in channel._video_pipeline._filter_contexts
        assert session.id not in channel._video_pipeline._sessions
    finally:
        await channel.close()


class _GatedVision(VisionProvider):
    """Answers once its gate opens."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()

    @property
    def name(self) -> str:
        return "gated"

    async def analyze_frame(self, frame: VideoFrame) -> VisionResult:
        await self.gate.wait()
        return VisionResult(description="a mug", labels=["mug"])


async def test_realtime_av_vision_finishing_during_session_end_restores_nothing() -> None:
    # An analysis under way when the session ends finished while the provider
    # disconnected, after the reset: it put the session back in the pipeline
    # and left its view behind.
    vision = _GatedVision()
    provider = MockRealtimeAudioVideoProvider()
    channel = RealtimeAudioVideoChannel(
        "rv",
        provider=provider,
        transport=MockRealtimeTransport(),
        video_pipeline=VideoPipelineConfig(filters=[CensorVideoFilter({"private"})]),
        vision=vision,
        vision_interval_ms=0,
    )
    disconnect = provider.disconnect

    async def release_vision_then_disconnect(session: VoiceSession) -> None:
        vision.gate.set()
        for _ in range(5):
            await asyncio.sleep(0)
        await disconnect(session)

    provider.disconnect = release_vision_then_disconnect  # type: ignore[method-assign]
    try:
        session = await channel.start_session("r1", "p", "ws")
        await provider.simulate_video(session, frame())
        await asyncio.sleep(0)
        await channel.end_session(session)

        assert session.id not in channel._video_pipeline._filter_contexts
        assert session.id not in channel._video_pipeline._sessions
        assert session.id not in channel._last_vision_results
        assert channel._room_vision == {}
    finally:
        await channel.close()


async def test_realtime_av_dispatches_and_drains_detection_events() -> None:
    # The provider's frames went through the filters, whose detections were
    # never drained: ON_VIDEO_DETECTION never fired and the events piled up.
    detection = VideoDetectionEvent(
        session=None, kind="face_touch", labels=["cheek"], confidence=1.0
    )
    provider = MockRealtimeAudioVideoProvider()
    channel = realtime_av_channel(
        VideoPipelineConfig(filters=[MockFaceTouchFilter({0: [detection]})]), provider=provider
    )
    hook = AsyncMock()
    channel._fire_video_detection_hook = hook
    try:
        session = await channel.start_session("r", "p", "ws")
        await provider.simulate_video(session, frame())
        await asyncio.gather(
            *(t for t in channel._scheduled_tasks if t.get_name().startswith("detection:"))
        )
        hook.assert_awaited_once_with(session, detection, "r")
        assert detection.session is session
        assert channel._video_pipeline.drain_events(session.id) == []
    finally:
        await channel.close()


async def test_vision_finishing_after_unbind_does_not_restore_session_state() -> None:
    started, finish = asyncio.Event(), asyncio.Event()
    vision = AsyncMock()

    async def analyze(frame: VideoFrame) -> VisionResult:
        started.set()
        await finish.wait()
        return VisionResult(description="late", labels=["private"])

    vision.analyze_frame.side_effect = analyze
    channel = VideoChannel(
        "v",
        backend=MockVideoBackend(),
        pipeline=VideoPipelineConfig(vision=vision, filters=[CensorVideoFilter({"private"})]),
    )
    session = VideoSession(id="a", room_id="r", channel_id="v", participant_id="p")
    channel.bind_session(
        session, "r", ChannelBinding(room_id="r", channel_id="v", channel_type="video")
    )
    task = asyncio.create_task(channel._analyze_frame(session, frame(), "r"))
    await started.wait()
    channel.unbind_session(session)
    finish.set()
    try:
        await task
        assert "a" not in channel._last_vision_results
        assert "a" not in channel._last_vision_ts
        assert "a" not in channel._video_pipeline._filter_contexts
    finally:
        await channel.close()
