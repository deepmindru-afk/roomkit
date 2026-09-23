"""ElevenLabs continues its voice from one response to the next (RFC §12.2.2).

The same three-response conversation is synthesized twice: once with the
TTS conversation context, where each request carries the ``request_id`` of
the responses before it (ElevenLabs request stitching), and once without.
Both land in WAV files so the two can be compared by ear; the debug log shows
the stitching arguments each request carried.

Run: ELEVENLABS_API_KEY=... uv run python examples/voice_elevenlabs_context.py
Writes elevenlabs_with_context.wav and elevenlabs_without_context.wav in the
current directory. Requires ``pip install roomkit[elevenlabs]``; no audio
device needed.
"""

from __future__ import annotations

import asyncio
import logging
import wave

from shared import require_env, setup_logging

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logger = setup_logging("voice_elevenlabs_context")
logging.getLogger("roomkit.voice.tts.elevenlabs").setLevel(logging.DEBUG)

RATE = 24000
RESPONSES = [
    "Your appointment is confirmed for Tuesday.",
    "Yes, Tuesday morning, at ten o'clock.",
    "Perfect, you will get a reminder the day before.",
]


async def conversation(api_key: str, *, use_context: bool) -> bytes:
    """Speak the three responses in one voice session; return the audio."""
    tts = ElevenLabsTTSProvider(
        ElevenLabsConfig(api_key=api_key, output_format=f"pcm_{RATE}", use_context=use_context)
    )
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice", tts=tts, backend=backend)
    async with RoomKit(voice=backend) as kit:
        kit.register_channel(channel)
        await kit.create_room(room_id="call")
        await kit.attach_channel("call", "voice")
        session = await kit.join("call", "voice", participant_id="alice")
        for text in RESPONSES:
            await channel.say(session, text)
        channel.unbind_session(session)
    pause = b"\x00\x00" * (RATE // 2)
    return pause.join(audio for _, audio in backend.sent_audio)


def write_wav(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    logger.info("Wrote %s (%.1f s)", path, len(pcm) / 2 / RATE)


async def main() -> None:
    api_key = require_env("ELEVENLABS_API_KEY")["ELEVENLABS_API_KEY"]
    logger.info("With the conversation context:")
    write_wav("elevenlabs_with_context.wav", await conversation(api_key, use_context=True))
    logger.info("Without it:")
    write_wav("elevenlabs_without_context.wav", await conversation(api_key, use_context=False))


if __name__ == "__main__":
    asyncio.run(main())
