"""Continuous CPU profiling with Pyroscope.

Demonstrates how to use RoomKit's PyroscopeProfiler to profile a voice
application and attribute CPU samples to individual rooms and sessions.

Requirements:
    pip install 'roomkit[pyroscope]'

Start a local Pyroscope server first:
    docker run -p 4040:4040 grafana/pyroscope

Run with:
    uv run python examples/telemetry_pyroscope.py

Environment variables:
    PYROSCOPE_SERVER  Pyroscope server URL (default: http://localhost:4040)

The voice side is fully mocked (backend, VAD, STT, TTS): one simulated
utterance is transcribed, and its ON_TRANSCRIPTION hook runs inside a
``tag_session`` block so its CPU samples carry the room and session tags.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.telemetry.pyroscope import PyroscopeProfiler
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logger = setup_logging("telemetry_pyroscope")

ROOM_ID = "demo-room"
SERVER = os.environ.get("PYROSCOPE_SERVER", "http://localhost:4040")

# 640 bytes = 20 ms of 16 kHz 16-bit mono PCM
SILENCE_FRAME = AudioFrame(data=b"\x00" * 640, sample_rate=16000)

# ---------------------------------------------------------------------------
# Pyroscope profiler
# ---------------------------------------------------------------------------

profiler = PyroscopeProfiler(
    application_name="roomkit-demo",
    server_address=SERVER,
    tags={"env": "development"},
)

# For Grafana Cloud, use:
# profiler = PyroscopeProfiler(
#     application_name="roomkit-demo",
#     server_address="https://profiles-prod-001.grafana.net",
#     basic_auth_username="<instance-id>",
#     basic_auth_password="<api-token>",
#     tenant_id="<tenant-id>",
# )


async def main() -> None:
    profiler.start()

    # ---------------------------------------------------------------------------
    # Framework
    # ---------------------------------------------------------------------------

    kit = RoomKit()

    backend = MockVoiceBackend()
    # One utterance: speech starts, one frame of speech, speech ends.
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.95),
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"\x00" * 3200,
                duration_ms=100.0,
            ),
        ]
    )
    stt = MockSTTProvider(transcripts=["hello world"])
    tts = MockTTSProvider()

    voice = VoiceChannel(
        "voice",
        backend=backend,
        stt=stt,
        tts=tts,
        pipeline=AudioPipelineConfig(vad=vad),
    )
    kit.register_channel(voice)

    # ---------------------------------------------------------------------------
    # Hook: profile each transcription with session tags
    # ---------------------------------------------------------------------------

    transcribed = asyncio.Event()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def on_transcription(event, ctx) -> HookResult:
        # Tag this code block so Pyroscope attributes its CPU to this session
        with profiler.tag_session(
            room_id=event.session.room_id,
            session_id=event.session.id,
            backend="mock",
        ):
            logger.info("[STT] %s", event.text)
        transcribed.set()
        return HookResult.allow()

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------

    try:
        await kit.create_room(room_id=ROOM_ID)
        await kit.attach_channel(ROOM_ID, "voice")

        session = await backend.connect(ROOM_ID, "user-1", "voice")
        await kit.join(ROOM_ID, "voice", session=session)
        logger.info("Session started: %s", session.id)

        # Simulate some audio frames (20 ms each)
        for _ in range(50):
            await backend.simulate_audio_received(session, SILENCE_FRAME)
            await asyncio.sleep(0.02)

        await asyncio.wait_for(transcribed.wait(), timeout=5.0)
        logger.info("Done — check Pyroscope UI at %s", SERVER)
    finally:
        await kit.close()
        profiler.stop()


if __name__ == "__main__":
    asyncio.run(main())
