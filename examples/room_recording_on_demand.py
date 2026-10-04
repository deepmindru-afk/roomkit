"""Record a room on demand, and write a trace no member receives.

A room already exists; a host starts recording it later (a meeting resumed
after a restart, say), feeds it audio from a capture the framework does not
wire itself, and stops it. Each start and stop is announced, the start before
any media (the consent point). Shows:
- kit.start_room_recording(): recorders on an existing room, all or nothing,
  ON_RECORDING_STARTED for each before it returns
- kit.add_room_recording_track(): a track declared to the room's recordings,
  its media fed through the framework; like every recording verb, it reads the
  room scoped to the caller's organization
- kit.stop_room_recording(): the results, ON_RECORDING_STOPPED for each
- kit.commit_event(): a record outside the pipeline, its index counted
  delivered so the room's next event never waits on it

Run with:
    uv run python examples/room_recording_on_demand.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from shared import setup_logging

from roomkit import HookExecution, HookTrigger, RoomKit, TextContent, WebSocketChannel
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, SystemContent
from roomkit.recorder import MediaRecordingConfig, RecordingTrack, RoomRecorderBinding
from roomkit.recorder.mock import MockMediaRecorder

logger = setup_logging("room_recording_on_demand")

MIC = RecordingTrack(
    id="audio:meeting", kind="audio", channel_id="capture", codec="pcm_s16le", sample_rate=48000
)


async def main() -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("member"))
    await kit.create_room(room_id="meeting", organization_id="acme")
    await kit.attach_channel("meeting", "member")

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC, name="consent")
    async def consent(event: Any, ctx: Any) -> None:
        logger.info("recording %s started in %s: tell the participants", event.id, event.room_id)

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC, name="stored")
    async def stored(event: Any, ctx: Any) -> None:
        logger.info("recording %s stopped: %s", event.id, event.urls or "(mock, no file)")

    recorder = MockMediaRecorder()
    binding = RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())
    await kit.start_room_recording("meeting", [binding], organization_id="acme")

    feed = await kit.add_room_recording_track("meeting", MIC, organization_id="acme")
    assert feed is not None
    for frame in range(3):  # 20 ms of 48 kHz mono PCM each
        feed.feed(b"\x00\x00" * 960, frame * 20.0)
    feed.close()
    logger.info("fed %d frames", len(recorder.chunks))

    results = await kit.stop_room_recording("meeting", organization_id="acme")
    logger.info("results: %s", [result.id for result in results])

    # A trace of the session, kept in the timeline for the host, received by no one.
    trace = RoomEvent(
        room_id="meeting",
        type=EventType.SYSTEM,
        source=EventSource(channel_id="recorder", channel_type=ChannelType.SYSTEM),
        content=SystemContent(body="Recording segment closed", code="segment_closed"),
    )
    committed = await kit.commit_event("meeting", trace, organization_id="acme")
    sent = await kit.send_event("meeting", "member", TextContent(body="Back in a minute."))
    logger.info("trace at index %d, next event at %d", committed.index, sent.index)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
