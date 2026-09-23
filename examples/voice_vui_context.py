"""Vui Nano answers inside the conversation, user audio and barge-in included.

Vui keeps the dialogue in its KV cache (RFC §12.2.2, level AUDIO): the reply
the user cut off, cut back to what was heard, then the user's own voice, then
the next reply generated in that thread. This example plays that scene:

1. the assistant (voice ``maeve``) starts a long answer, the user barges in;
2. the user asks a question (spoken by the second voice, ``abraham``, and
   routed through STT as a user turn with its audio);
3. the assistant answers.

It writes ``vui_conversation.wav`` (what the user heard) and logs how the
cache follows the dialogue.

Run: uv run --extra vui python examples/voice_vui_context.py
Requires Python 3.12, a CUDA GPU and ``pip install roomkit[vui]``; weights
download from Hugging Face on first run. If the codec fails with
``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`` (a system cuDNN shadowing
PyTorch's), run with ``VUI_DISABLE_CUDNN=1``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import wave

from shared import setup_logging

from roomkit import RoomKit, TTSContextConfig, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceCapability
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.vui import SAMPLE_RATE, VuiTTSConfig, VuiTTSProvider, VuiVoice

logger = setup_logging("voice_vui_context")
logging.getLogger("roomkit.voice.tts.vui").setLevel(logging.DEBUG)

FIRST_REPLY = (
    "Your appointment is confirmed for Tuesday at ten in the morning, and "
    "please remember to bring your insurance card and a list of your medications."
)
QUESTION = "Sorry, wait, is that Tuesday morning?"
SECOND_REPLY = "Yes, Tuesday morning at ten. Does that still work for you?"


def pcm_of(data_url: str) -> bytes:
    """Raw 24 kHz PCM out of a WAV data URL."""
    wav = base64.b64decode(data_url.split(",", 1)[1])
    return wav[44:]


def to_16k(pcm_24k: bytes) -> bytes:
    """Drop to the pipeline's 16 kHz, as a microphone path would deliver it."""
    import numpy as np

    samples = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32)
    positions = np.arange(0, len(samples), SAMPLE_RATE / 16000)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.int16).tobytes()


async def main() -> None:
    if os.environ.get("VUI_DISABLE_CUDNN") == "1":
        import torch

        torch.backends.cudnn.enabled = False

    tts = VuiTTSProvider(
        VuiTTSConfig(voices={"maeve": VuiVoice("maeve"), "abraham": VuiVoice("abraham")})
    )
    await tts.warmup()
    question_24k = pcm_of((await tts.synthesize(QUESTION, voice="abraham")).url)

    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice",
        stt=MockSTTProvider(transcripts=[QUESTION]),
        tts=tts,
        backend=backend,
        batch_mode=True,
        tts_context=TTSContextConfig(include_audio=True),
    )
    async with RoomKit(voice=backend) as kit:
        kit.register_channel(channel)
        await kit.create_room(room_id="call")
        await kit.attach_channel("call", "voice")
        session = await kit.join("call", "voice", participant_id="caller")

        # 1. The long answer, cut off after 1.5 s.
        speaking = asyncio.create_task(channel.say(session, FIRST_REPLY))
        await asyncio.sleep(1.5)
        await channel.interrupt(session, reason="barge_in")
        await speaking

        # 2. The user's question, with its audio.
        await backend.simulate_audio_received(
            session, AudioFrame(data=to_16k(question_24k), sample_rate=16000)
        )
        await channel.flush_stt(session, route=True)

        # 3. The answer, generated after both.
        await channel.say(session, SECOND_REPLY)

        # What the user heard of the first reply, as the timeline recorded it.
        events = await kit.store.list_events("call")
        [cut] = [e for e in events if e.metadata.get("interrupted")]
        first_heard = int(cut.metadata["played_ms"])
        logger.info("First reply cut after %d ms", first_heard)
        first_audio = backend.sent_audio[0][1][: int(first_heard * SAMPLE_RATE / 1000) * 2]
        second_audio = backend.sent_audio[-1][1]
        channel.unbind_session(session)

    pause = b"\x00\x00" * (SAMPLE_RATE // 4)
    with wave.open("vui_conversation.wav", "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(first_audio + pause + question_24k + pause + second_audio)
    logger.info("Wrote vui_conversation.wav")
    await tts.close()


if __name__ == "__main__":
    asyncio.run(main())
