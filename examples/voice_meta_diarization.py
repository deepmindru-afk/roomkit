"""RoomKit - a voice room that knows who is speaking (Meta Muse diarization).

    META_API_KEY=... uv run --extra meta-stt --extra local-audio \\
        python examples/voice_meta_diarization.py

Two or more people talk in turn near one microphone. The VoiceChannel runs
Meta's ``muse-voice-transcribe-1.0`` in ``DIARIZATION`` mode, in continuous
mode (no pipeline VAD): it keeps one stream across turns, so the model's speaker
labels hold, and every turn becomes a room message of its own carrying
``sender_name`` - "Speaker A", "Speaker B"... (RFC §12.2.3). The sender stays the
room's single audio participant; the speaker is metadata.

An ``ON_TRANSCRIPTION`` hook names the voices you know (``SPEAKER_NAMES``):
the name it sets is what the room message carries, and what an AI channel in
the room would read ("Sylvain: ..."). ``ON_SPEAKER_CHANGE`` logs each change of
voice with its source, ``stt``.

The labels are the model's order of appearance: whoever speaks first is A. A
stream the service ends (after an hour, or on an error) starts a new epoch, and
its first voice becomes "Speaker A#1": the same letter in two streams is not
assumed to be the same person.

Requires:
    pip install roomkit[meta-stt,local-audio]

Environment variables:
    META_API_KEY    (required) Meta Model API key
    LANGUAGE_BIAS   Language name to bias toward, e.g. French (default: none)
    SPEAKER_NAMES   Comma-separated LABEL=Name pairs, e.g. "A=Sylvain,B=Julie"

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

from shared import require_env, run_until_stopped, setup_logging

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.meta import MetaSTTConfig, MetaSTTProvider

logger = setup_logging("roomkit.examples.voice_meta_diarization")

SAMPLE_RATE = 16000


def speaker_names(value: str | None) -> dict[str, str]:
    """``"A=Sylvain,B=Julie"`` as ``{"A": "Sylvain", "B": "Julie"}``."""
    names: dict[str, str] = {}
    for pair in (value or "").split(","):
        label, _, name = pair.partition("=")
        if label.strip() and name.strip():
            names[label.strip()] = name.strip()
    return names


async def main() -> None:
    env = require_env("META_API_KEY")
    names = speaker_names(os.environ.get("SPEAKER_NAMES"))
    bias = os.environ.get("LANGUAGE_BIAS")

    kit = RoomKit()
    backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
        block_duration_ms=20,
    )
    stt = MetaSTTProvider(
        MetaSTTConfig(
            api_key=env["META_API_KEY"],
            mode="DIARIZATION",
            language_bias=[bias] if bias else [],
        )
    )
    # No VAD in the pipeline: continuous mode, where the channel keeps the
    # diarizing stream across turns. Behind a VAD it would refuse the provider.
    voice = VoiceChannel("voice", stt=stt, backend=backend, pipeline=AudioPipelineConfig())
    kit.register_channel(voice)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def name_known_voices(event: Any, ctx: Any) -> HookResult:
        """Put a known name on a label; the room message carries it from here on."""
        name = names.get(event.speaker or "")
        if name and event.speaker_epoch == 0:
            return HookResult.modify(dataclasses.replace(event, sender_name=name))
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_SPEAKER_CHANGE, execution=HookExecution.ASYNC)
    async def log_speaker_change(event: Any, ctx: Any) -> None:
        logger.info(
            "   (voice change: %s, %s%s)",
            event.speaker_id,
            event.source,
            ", first time" if event.is_new_speaker else "",
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def show_message(event: Any, ctx: Any) -> None:
        if event.source.channel_id != "voice":
            return
        speaker = event.metadata.get("sender_name", "?")
        logger.info("%s: %s", speaker, getattr(event.content, "body", ""))

    await kit.create_room(room_id="diarization-demo")
    # Attaching the voice channel starts the local microphone session.
    await kit.attach_channel("diarization-demo", "voice")

    logger.info("Listening - take turns speaking near the microphone. Ctrl+C to stop.")
    if names:
        logger.info("Known voices: %s", ", ".join(f"{k}={v}" for k, v in names.items()))
    await run_until_stopped(kit)


if __name__ == "__main__":
    asyncio.run(main())
