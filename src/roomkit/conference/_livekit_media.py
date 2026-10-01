"""Frames from a subscribed LiveKit track into the framework's own types.

Split out from the session that starts them because it is the one part of a bot
connection that touches no session state: a track goes in, framework frames come
out, and nothing is decided along the way that anything else depends on. Which
also means the format declaration — the whole point of having a real backend —
can be read here without a conference to hold it.

One task per subscribed track, so a lane doing its work inline delays that
track's frames and nobody else's. That is the isolation RFC section 12.10.4 makes
checkable from outside. :class:`TrackPumps` keeps those tasks and nothing else:
which tracks are wanted is the room view's to decide.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.conference._livekit_mapping import SAMPLE_WIDTH, codec_for_buffer_type
from roomkit.conference.models import ConferenceTrack, TrackKind
from roomkit.core.task_utils import cancel_and_wait
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.conference.livekit")

AudioSink = Callable[[ConferenceTrack, AudioFrame], Awaitable[None]]
VideoSink = Callable[[ConferenceTrack, VideoFrame], Awaitable[None]]


async def pump_audio(
    *,
    rtc: Any,
    track: Any,
    record: ConferenceTrack,
    sink: AudioSink,
    sample_rate: int,
    channels: int,
) -> None:
    """Deliver a track's decoded audio, in the format it was asked for.

    The rate and channel count are what this backend requested of LiveKit's
    decoder, and every frame *declares* them. Nothing is resampled: normalising
    to what a recognizer wants belongs to the lane, and a transport that did it
    would hide from the pipeline the one thing this backend exists to hand it —
    audio the framework did not manufacture.

    ``timestamp_ms`` counts the samples that have gone by, so it is a clock that
    advances with the audio rather than with the wall. It starts at zero when the
    subscription does, which is the only origin available here: a track the
    framework unsubscribed and took again is a new stream to this pump.
    """
    stream = rtc.AudioStream.from_track(
        track=track, sample_rate=sample_rate, num_channels=channels
    )
    delivered = 0
    try:
        async for event in stream:
            frame = event.frame
            await sink(
                record,
                AudioFrame(
                    data=bytes(frame.data),
                    sample_rate=frame.sample_rate,
                    channels=frame.num_channels,
                    sample_width=SAMPLE_WIDTH,
                    timestamp_ms=delivered * 1000 / frame.sample_rate,
                ),
            )
            delivered += frame.samples_per_channel
    finally:
        with contextlib.suppress(Exception):
            await stream.aclose()


async def pump_video(
    *,
    rtc: Any,
    track: Any,
    record: ConferenceTrack,
    sink: VideoSink,
) -> None:
    """Deliver a track's video as raw I420.

    One layout is requested rather than taking whatever the decoder emits, so
    that what arrives is the same shape whichever codec the publisher
    negotiated. Each frame still declares its own: a decoder that answered with
    something else must not be described as I420.
    """
    stream = rtc.VideoStream(track=track, format=rtc.VideoBufferType.I420)
    sequence = 0
    try:
        async for event in stream:
            frame = event.frame
            try:
                codec = codec_for_buffer_type(rtc.VideoBufferType.Name(frame.type))
            except ValueError:
                logger.warning(
                    "Dropping a video frame on conference track %s in room %s",
                    record.id,
                    record.room_id,
                    exc_info=True,
                )
                continue
            await sink(
                record,
                VideoFrame(
                    data=bytes(frame.data),
                    codec=codec,
                    width=frame.width,
                    height=frame.height,
                    timestamp_ms=event.timestamp_us / 1000,
                    sequence=sequence,
                ),
            )
            sequence += 1
    finally:
        with contextlib.suppress(Exception):
            await stream.aclose()


class TrackPumps:
    """One pump task per subscribed track, started and stopped by track id."""

    def __init__(
        self,
        *,
        rtc: Any,
        room_id: str,
        audio_sink: AudioSink,
        video_sink: VideoSink,
        config: Any,
    ) -> None:
        self._rtc = rtc
        self._room_id = room_id
        self._audio_sink = audio_sink
        self._video_sink = video_sink
        # Read when a pump starts, not here: the delivery format is the
        # backend configuration's, consulted at the moment it applies.
        self._config = config
        self._pumps: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def start(self, record: ConferenceTrack, track: Any) -> None:
        """Start the track's pump, unless one runs already or the pumps are closed.

        Closed is for good: the SDK can still report a track subscribed while
        the session is leaving, and a pump started then is one nothing would
        ever stop.
        """
        if self._closed or record.id in self._pumps:
            return
        self._pumps[record.id] = asyncio.create_task(self._run(record, track))

    def cancel(self, track_id: str) -> None:
        """Cancel the track's pump without waiting, from a synchronous SDK callback."""
        if (pump := self._pumps.pop(track_id, None)) is not None:
            pump.cancel()

    async def stop(self, track_id: str) -> None:
        """Cancel the track's pump and wait until it has ended."""
        await cancel_and_wait(self._pumps.pop(track_id, None))

    async def close(self) -> None:
        """Cancel every pump, wait until each has ended, and start none after."""
        self._closed = True
        pumps = list(self._pumps.values())
        self._pumps.clear()
        await cancel_and_wait(*pumps)

    async def _run(self, record: ConferenceTrack, track: Any) -> None:
        """Run a track's pump, and survive its ending either way.

        A pump that raises is one track's stream failing, and it must not take
        the session's other tracks or its event bridge with it — so it is
        reported here and the task ends.
        """
        try:
            if record.kind is TrackKind.AUDIO:
                await pump_audio(
                    rtc=self._rtc,
                    track=track,
                    record=record,
                    sink=self._audio_sink,
                    sample_rate=self._config.audio_sample_rate,
                    channels=self._config.audio_channels,
                )
            else:
                await pump_video(
                    rtc=self._rtc,
                    track=track,
                    record=record,
                    sink=self._video_sink,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "The pump for conference track %s in room %s stopped", record.id, self._room_id
            )
