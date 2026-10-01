"""RoomKit — Talk to an Anam AI avatar with your local mic and speakers.

Anam runs the whole conversation in its cloud (STT → LLM → TTS → face
animation) and streams the avatar back as synchronized audio and video.
This example sends your microphone to Anam through a
RealtimeAudioVideoChannel, plays the avatar's voice through your speakers,
and writes the avatar's video to an MP4.

What you hear: the avatar answering you, live, through your speakers.
What you see: nothing during the call — there is no local video window. The
avatar's video (picture only, no sound) is encoded to an MP4 whose path is
printed when you press Ctrl+C; open it with any video player.

Requirements:
    pip install roomkit[anam,local-audio,webrtc-aec]
    (anam brings PyAV, which encodes the MP4; webrtc-aec is the default echo
    canceller, without it the mic is muted while the avatar speaks)

Run with an inline persona (avatar + voice + LLM):
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    export ANAM_VOICE_ID="your-voice-id"
    export ANAM_LLM_ID="your-llm-id"           # e.g. ANAM_GPT_4O_MINI_V1
    uv run python examples/realtime_av_anam.py

Or with a persona built in Anam Lab:
    export ANAM_API_KEY="your-api-key"
    export ANAM_PERSONA_ID="your-persona-id"
    uv run python examples/realtime_av_anam.py

The ids are listed at lab.anam.ai, or by the API with your key:
GET https://api.anam.ai/v1/avatars, /v1/voices, /v1/llms, /v1/personas.

Environment variables:
    ANAM_API_KEY        (required) Anam API key
    ANAM_AVATAR_ID      Avatar id (required unless ANAM_PERSONA_ID is set)
    ANAM_VOICE_ID       Voice id (required unless ANAM_PERSONA_ID is set)
    ANAM_LLM_ID         LLM id (required unless ANAM_PERSONA_ID is set)
    ANAM_PERSONA_ID     Persona from Anam Lab: brings its own avatar, voice, LLM
                        and prompt, and replaces the three variables above
    ANAM_LANGUAGE       Language of Anam's STT/TTS, e.g. "fr" (default: "en";
                        inline persona only)
    SYSTEM_PROMPT       System prompt of the avatar (inline persona only)
    AEC                 webrtc (default) | speex | 0 to disable
    MUTE_MIC            1 to mute the mic while the avatar speaks (default: only
                        when AEC is disabled)
    AVATAR_VIDEO_DIR    Directory of the MP4 (default: <temp dir>/roomkit-anam)
    CONSOLE             Set to 1 for the live console dashboard

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import (
    build_aec,
    env_bool,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import RealtimeAudioVideoChannel, RoomKit
from roomkit.providers.anam import AnamConfig, AnamRealtimeProvider
from roomkit.video import VideoFrame, VideoSession
from roomkit.video.recorder import (
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
)
from roomkit.video.recorder.pyav import PyAVVideoRecorder
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_av_anam")

# Anam's audio arrives over WebRTC (Opus) at 48 kHz. Running the mic and the
# speakers at that rate too means no resampling anywhere, and lets the echo
# canceller compare both directions sample for sample.
SAMPLE_RATE = 48000
BLOCK_MS = 20
AVATAR_FPS = 25.0  # Anam streams the avatar at 25 frames per second


class AvatarVideoWriter:
    """Encode the avatar's video frames to an MP4 on a worker thread.

    H.264 encoding costs milliseconds per frame; done on the event loop it
    would delay the avatar's audio. A single worker keeps the frames in order.
    """

    def __init__(self, directory: Path) -> None:
        self._recorder = PyAVVideoRecorder()
        self._config = VideoRecordingConfig(
            storage=str(directory), codec="libx264", fps=AVATAR_FPS
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avatar-video")
        self._handle: VideoRecordingHandle | None = None
        self._failed = False

    def tap(self, session: VideoSession, frame: VideoFrame) -> None:
        if not self._failed:
            self._executor.submit(self._write, session, frame)

    def _write(self, session: VideoSession, frame: VideoFrame) -> None:
        try:
            if self._handle is None:
                self._handle = self._recorder.start(session, self._config)
                logger.info("Avatar video is being written to %s", self._handle.path)
            self._recorder.tap_frame(self._handle, frame)
        except Exception:
            logger.exception("Avatar video encoding failed; the MP4 stops here")
            self._failed = True

    def close(self) -> VideoRecordingResult | None:
        """Finish the queued frames and close the MP4 (blocking)."""
        self._executor.shutdown(wait=True)
        if self._handle is None:
            return None
        return self._recorder.stop(self._handle)


def build_provider(api_key: str) -> AnamRealtimeProvider:
    """The Anam provider, from a Lab persona or from an inline avatar."""
    persona_id = os.environ.get("ANAM_PERSONA_ID")
    if persona_id:
        return AnamRealtimeProvider(AnamConfig(api_key=api_key, persona_id=persona_id))

    # On Anam's own WebRTC transport an inline persona needs all three: the
    # session API refuses an avatar without a voice and an LLM.
    env = require_env("ANAM_AVATAR_ID", "ANAM_VOICE_ID", "ANAM_LLM_ID")
    config = AnamConfig(
        api_key=api_key,
        avatar_id=env["ANAM_AVATAR_ID"],
        voice_id=env["ANAM_VOICE_ID"],
        llm_id=env["ANAM_LLM_ID"],
        language_code=os.environ.get("ANAM_LANGUAGE", "en"),
    )
    return AnamRealtimeProvider(config)


async def main() -> None:
    env = require_env("ANAM_API_KEY")
    provider = build_provider(env["ANAM_API_KEY"])

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- Local mic + speakers, with echo cancellation ---
    aec = build_aec(SAMPLE_RATE, BLOCK_MS, default="webrtc")
    # Without AEC the speakers feed straight back into the mic: mute it while
    # the avatar speaks unless MUTE_MIC says otherwise.
    mute_mic = env_bool("MUTE_MIC", default=aec is None)
    transport = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )

    # --- Realtime audio+video channel ---
    channel = RealtimeAudioVideoChannel(
        "avatar-anam",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a helpful AI avatar. Keep responses conversational and concise.",
        ),
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
    )

    # --- Avatar video → MP4 ---
    video_dir = os.environ.get("AVATAR_VIDEO_DIR") or Path(tempfile.gettempdir()) / "roomkit-anam"
    video = AvatarVideoWriter(Path(video_dir))
    channel.add_video_media_tap(video.tap)

    kit.register_channel(channel)
    await kit.create_room(room_id="avatar-room")
    await kit.attach_channel("avatar-room", channel.channel_id)

    # --- Start the session (connects to Anam, opens the mic and speakers) ---
    session = await channel.start_session("avatar-room", "local-user", connection=None)
    logger.info("Anam avatar connected — speak into your microphone! Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C ---
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        # The provider bounds Anam's WebRTC close (anam 0.11 can hang there
        # once the mic track has run; the aiortc "Task exception was never
        # retrieved" traceback at exit comes from the same bug).
        await channel.end_session(session)
        result = await asyncio.to_thread(video.close)
        if result is None:
            logger.info("No avatar video was received.")
        else:
            logger.info(
                "Avatar video: %s (%d frames, %.1f s)",
                result.url,
                result.frame_count,
                result.frame_count / AVATAR_FPS,
            )

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
