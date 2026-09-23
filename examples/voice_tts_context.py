"""A TTS provider that hears the conversation so far (RFC §12.2.2).

A provider declares the context it can use through ``context_level``. The
Voice Channel then hands it, on every streaming call, the dialogue of the
voice session: what the user said (after the transcription hooks) and what the
provider said itself, cut to what was actually played.

Here a toy provider logs the context it receives. A real one would condition
its prosody on it (previous text, previous request ids, or the audio itself
at ``TTSContextLevel.AUDIO`` with ``TTSContextConfig(include_audio=True)``).

Run: uv run python examples/voice_tts_context.py
No credentials, network service or audio device required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from shared import setup_logging

from roomkit import RoomKit, TTSContext, TTSContextConfig, TTSContextLevel, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider

logger = setup_logging("voice_tts_context")

_RATE = 16000
_WORD = b"\x00\x00" * (_RATE // 5)  # 200 ms of silence per word


class ContextLoggingTTS(TTSProvider):
    """Logs the dialogue it is given, then 'speaks' 200 ms per word."""

    @property
    def context_level(self) -> TTSContextLevel:
        return TTSContextLevel.TEXT

    def release_context(self, context_id: str) -> None:
        logger.info("Context %s released", context_id)

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        turns = context.turns if context is not None else ()
        logger.info("Synthesizing %r after %d turn(s):", text, len(turns))
        for turn in turns:
            logger.info("  %-9s %r (played_ms=%s)", turn.role, turn.text, turn.played_ms)
        for _ in text.split():
            yield AudioChunk(data=_WORD, sample_rate=_RATE)


async def main() -> None:
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice",
        stt=MockSTTProvider(transcripts=["Is that on Tuesday?"]),
        tts=ContextLoggingTTS(),
        backend=backend,
        batch_mode=True,  # the caller decides when the user has spoken
        tts_context=TTSContextConfig(max_turns=10),
    )
    async with RoomKit(voice=backend) as kit:
        kit.register_channel(channel)
        await kit.create_room(room_id="call")
        await kit.attach_channel("call", "voice")
        session = await kit.join("call", "voice", participant_id="alice")

        # 1. The bot speaks first: no context yet.
        await channel.say(session, "Your appointment is confirmed.")

        # 2. The user answers; the routed transcript becomes a user turn.
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x00" * _RATE))
        await channel.flush_stt(session, route=True)

        # 3. The bot replies, and its TTS now hears both previous turns.
        await channel.say(session, "Yes, Tuesday at ten.")

        channel.unbind_session(session)


if __name__ == "__main__":
    asyncio.run(main())
