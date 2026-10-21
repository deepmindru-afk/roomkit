"""RoomKit - a voice room that knows who is speaking (Mistral Voxtral diarization).

    MISTRAL_API_KEY=... uv run --extra mistral-voice --extra local-audio \\
        python examples/voice_voxtral_diarization.py

Two or more people talk in turn near one microphone. The VoiceChannel runs
Voxtral Mini Transcribe Realtime 3 with ``diarize=True``, in continuous mode
(no pipeline VAD): it keeps one session across turns, so the labels hold, and
every turn becomes a room message of its own carrying ``sender_name`` -
"Speaker 1", "Speaker 2"... (RFC §12.2.3).

Voxtral cannot detect turns and label speakers in one session, so a turn ends
when another voice starts, or after a pause without new words
(``DIARIZED_PAUSE_MS``, 1.5 s by default; one person's hesitations can then end
their turn in two).

``REFERENCES`` names known voices: a short recording of each person (3 to 15 s
of speech, WAV/MP3/FLAC/Opus) under a name Voxtral then labels their words
with. An ``ON_TRANSCRIPTION`` hook makes that name the message's
``sender_name`` ("Julie", not the channel's default "Speaker Julie"). A voice
the references do not match is numbered after them ("Speaker 3").

Requires:
    pip install roomkit[mistral-voice,local-audio]

Environment variables:
    MISTRAL_API_KEY     (required) Mistral API key
    VOICE_LANGUAGE      Language to transcribe, e.g. fr (default: detected)
    REFERENCES          Comma-separated Name=path pairs, e.g. "Sylvain=me.wav,Julie=julie.wav"
    DIARIZED_PAUSE_MS   Pause without words that ends a turn, in ms (default: 1500)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import dataclasses
import os
from typing import Any

from shared import require_env, run_until_stopped, setup_logging, voice_language

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.voxtral import VoxtralSTTConfig, VoxtralSTTProvider

logger = setup_logging("roomkit.examples.voice_voxtral_diarization")

SAMPLE_RATE = 16000


def references(value: str | None) -> dict[str, bytes]:
    """``"Sylvain=me.wav,Julie=julie.wav"`` as ``{"Sylvain": <bytes>, "Julie": <bytes>}``."""
    clips: dict[str, bytes] = {}
    for pair in (value or "").split(","):
        name, _, path = pair.partition("=")
        if name.strip() and path.strip():
            clips[name.strip()] = Path(path.strip()).expanduser().read_bytes()
    return clips


async def main(clips: dict[str, bytes]) -> None:
    env = require_env("MISTRAL_API_KEY")

    kit = RoomKit()
    backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
        block_duration_ms=20,
    )
    stt = VoxtralSTTProvider(
        VoxtralSTTConfig(
            api_key=env["MISTRAL_API_KEY"],
            language=voice_language(None),
            diarize=True,
            # Voxtral labels speakers or detects turns, not both in one session.
            turn_detection=None,
            speaker_references=clips,
            diarized_pause_ms=int(os.environ.get("DIARIZED_PAUSE_MS", "1500")),
        )
    )
    # No VAD in the pipeline: continuous mode, where the channel keeps the
    # diarizing session across turns. Behind a VAD it would refuse the provider.
    voice = VoiceChannel("voice", stt=stt, backend=backend, pipeline=AudioPipelineConfig())
    kit.register_channel(voice)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def name_known_voices(event: Any, ctx: Any) -> HookResult:
        """A reference's name is the person's: the room message carries it as is."""
        if event.speaker in clips:
            return HookResult.modify(dataclasses.replace(event, sender_name=event.speaker))
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_SPEAKER_CHANGE, execution=HookExecution.ASYNC)
    async def log_speaker_change(event: Any, ctx: Any) -> None:
        logger.info(
            "   (voice change: %s%s)",
            event.speaker_id,
            ", first time" if event.is_new_speaker else "",
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def show_message(event: Any, ctx: Any) -> None:
        if event.source.channel_id != "voice":
            return
        speaker = event.metadata.get("sender_name", "?")
        logger.info("%s: %s", speaker, getattr(event.content, "body", ""))

    await kit.create_room(room_id="voxtral-diarization-demo")
    # Attaching the voice channel starts the local microphone session.
    await kit.attach_channel("voxtral-diarization-demo", "voice")

    logger.info("Listening - take turns speaking near the microphone. Ctrl+C to stop.")
    if clips:
        logger.info("Known voices: %s", ", ".join(clips))
    await run_until_stopped(kit)


if __name__ == "__main__":
    # The clips are read before the loop starts: no file I/O inside it.
    asyncio.run(main(references(os.environ.get("REFERENCES"))))
