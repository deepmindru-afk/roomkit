"""RoomKit - a voice room that names who speaks from the pipeline's diarization.

    META_API_KEY=... SPEAKER_MODEL=nemo_en_titanet_large.onnx \\
        uv run --extra meta-stt --extra local-audio --extra sherpa-onnx \\
        python examples/voice_pipeline_speakers.py Sylvain Julie

Each name on the command line records a few seconds of that person's voice
from the microphone and enrolls it with sherpa-onnx speaker embeddings. Then
the room listens: the pipeline's VAD cuts utterances, the
``DiarizationProvider`` matches each one against the enrolled voices, and the
``VoiceChannel`` with ``pipeline_speakers=True`` gives every transcript the
speaker the stage heard the longest over it (RFC §12.2.3). The room message
carries ``speaker_label`` and ``sender_name``; the sender stays the room's
single audio participant.

The STT here labels nobody (Muse in ``PUSH_TO_TALK`` mode, one final per
utterance): the speaker comes from the pipeline alone. Unlike
``voice_meta_diarization.py``, where the model names voices A, B... in order of
appearance, the voices are known in advance, so a name holds for the whole
session. A voice matching no enrolled one (below ``MATCH_THRESHOLD``) is
"Unknown speaker".

The stage's label is the enrolled name, and the default ``sender_name`` would
read "Speaker Sylvain": an ``ON_TRANSCRIPTION`` hook carries the name as is.

Requires:
    pip install roomkit[meta-stt,local-audio,sherpa-onnx]
    A speaker embedding model, e.g. NeMo TitaNet or 3D-Speaker, from
    https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models

Environment variables:
    META_API_KEY     (required) Meta Model API key
    SPEAKER_MODEL    (required) Path to the speaker embedding .onnx model
    MATCH_THRESHOLD  Cosine similarity a voice must reach to be named (default: 0.5)
    LANGUAGE_BIAS    Language name to bias toward, e.g. French (default: none)
    VAD              energy | silero | ten (default: energy)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import asyncio
import dataclasses
import itertools
import math
import os
from typing import Any

import numpy as np
import sounddevice as sd
from shared import build_vad, require_env, run_until_stopped, setup_logging

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.diarization import (
    SherpaOnnxDiarizationConfig,
    SherpaOnnxDiarizationProvider,
)
from roomkit.voice.stt.meta import MetaSTTConfig, MetaSTTProvider

logger = setup_logging("roomkit.examples.voice_pipeline_speakers")

SAMPLE_RATE = 16000
SILENT_DBFS = -50.0  # an enrollment quieter than this recorded no voice


def record(seconds: float) -> bytes:
    """``seconds`` of 16-bit mono PCM from the default microphone."""
    frames = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return frames.tobytes()


def level_dbfs(pcm: bytes) -> float:
    """The recording's RMS level, in dBFS."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    rms = math.sqrt(float(np.mean(samples**2))) if samples.size else 0.0
    return 20 * math.log10(max(rms, 1.0) / 32768)


def similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


async def enroll(
    diarization: SherpaOnnxDiarizationProvider, names: list[str], seconds: float
) -> None:
    """Record each person in turn and register their voice under their name.

    Logs each recording's level and how alike the enrolled voices are: a
    voice is named when its score passes MATCH_THRESHOLD, so the threshold
    has to sit above what two different voices score together.
    """
    voices: dict[str, list[float]] = {}
    for name in names:
        await asyncio.to_thread(input, f"\n{name}: press Enter, then talk for {seconds:g} s... ")
        pcm = await asyncio.to_thread(record, seconds)
        level = level_dbfs(pcm)
        if level < SILENT_DBFS:
            raise SystemExit(f"{name}: the recording is silent ({level:.0f} dBFS), check the mic")
        voices[name] = diarization.extract_embedding(pcm, SAMPLE_RATE)
        diarization.enroll_speaker(name, voices[name])
        logger.info("Enrolled %s (level %.0f dBFS)", name, level)
    for a, b in itertools.combinations(voices, 2):
        logger.info("%s and %s score %.2f together", a, b, similarity(voices[a], voices[b]))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="The people to enroll, e.g. Sylvain Julie")
    parser.add_argument("--seconds", type=float, default=5.0, help="Enrollment length per voice")
    args = parser.parse_args()
    env = require_env("META_API_KEY", "SPEAKER_MODEL")
    bias = os.environ.get("LANGUAGE_BIAS")

    diarization = SherpaOnnxDiarizationProvider(
        SherpaOnnxDiarizationConfig(
            model=env["SPEAKER_MODEL"],
            search_threshold=float(os.environ.get("MATCH_THRESHOLD", "0.5")),
        )
    )
    await enroll(diarization, args.names, args.seconds)
    enrolled = set(args.names)

    kit = RoomKit()
    backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
        block_duration_ms=20,
    )
    stt = MetaSTTProvider(
        MetaSTTConfig(
            api_key=env["META_API_KEY"],
            mode="PUSH_TO_TALK",
            language_bias=[bias] if bias else [],
        )
    )
    voice = VoiceChannel(
        "voice",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=build_vad(SAMPLE_RATE), diarization=diarization),
        pipeline_speakers=True,
    )
    kit.register_channel(voice)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def enrolled_names(event: Any, ctx: Any) -> HookResult:
        """The stage's label is the enrolled name: the room message carries it as is."""
        if event.speaker in enrolled:
            return HookResult.modify(dataclasses.replace(event, sender_name=event.speaker))
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_SPEAKER_CHANGE, execution=HookExecution.ASYNC)
    async def log_speaker_change(event: Any, ctx: Any) -> None:
        logger.info("   (voice change: %s, %s)", event.speaker_id, event.source)

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def show_message(event: Any, ctx: Any) -> None:
        if event.source.channel_id != "voice":
            return
        speaker = event.metadata.get("sender_name", "?")
        logger.info("%s: %s", speaker, getattr(event.content, "body", ""))

    await kit.create_room(room_id="pipeline-speakers-demo")
    # Attaching the voice channel starts the local microphone session.
    await kit.attach_channel("pipeline-speakers-demo", "voice")

    logger.info("Listening - take turns speaking near the microphone. Ctrl+C to stop.")
    await run_until_stopped(kit)


if __name__ == "__main__":
    asyncio.run(main())
