"""RoomKit - speak into your microphone and watch Meta Muse transcribe it.

    META_API_KEY=... uv run --extra meta-stt --extra local-audio \\
        python examples/stt_meta_mic.py [--diarize]

Live speech-to-text on Meta's ``muse-voice-transcribe-1.0``, fed by your own
voice. The line at the bottom of the terminal reacts as soon as the model hears
you start (``...``), rewrites itself while you talk, and is committed (``>``)
once you pause for about half a second: the model itself decides where each of
your turns ends (``ENDPOINTING``), with no VAD on RoomKit's side.

That is the shape a ``VoiceChannel`` without a pipeline VAD relies on: turn
boundaries come from the recogniser, and a reply can start the moment a turn
is committed.

``--diarize`` switches the model to ``DIARIZATION``: each committed turn says
who spoke (``> A: ...``, ``> B: ...``), and a change of voice ends a turn even
without a pause. Try it with two people talking in turn near the microphone.
The labels hold for the life of the stream. This example reads the provider
directly; ``voice_meta_diarization.py`` runs the same model through a
``VoiceChannel``, where each turn becomes a room message with its speaker.

Requires:
    pip install roomkit[meta-stt,local-audio]

Environment variables:
    META_API_KEY    (required) Meta Model API key
    LANGUAGE_BIAS   Language name to bias toward, e.g. French (default: none)
    STT_KEYWORDS    Comma-separated terms to bias toward, e.g. "RoomKit,Tremblay"

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import asyncio
import contextlib
import os
from collections.abc import AsyncIterator

from shared import require_env, setup_logging

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.stt.meta import MetaSTTConfig, MetaSTTProvider

logger = setup_logging("roomkit.examples.stt_meta_mic")

SAMPLE_RATE = 16000
"""One of the two rates the service takes as is (16 and 24 kHz), and the one
every microphone offers."""

BLOCK_MS = 40


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


class Caption:
    """One rewriting line of live captioning.

    ``logger`` would stamp and newline every update, which turns a caption
    into a wall of near-identical lines. The interim overwrites itself in
    place and only a final is committed.
    """

    def __init__(self) -> None:
        self._width = 0

    def interim(self, text: str) -> None:
        self._write(f"  {text}")

    def final(self, text: str, speaker: str | None = None) -> None:
        self._write(f"> {speaker}: {text}" if speaker else f"> {text}")
        print()
        self._width = 0

    def _write(self, line: str) -> None:
        # Pad to erase whatever the previous, longer line left behind.
        print(line.ljust(self._width), end="\r", flush=True)
        self._width = max(self._width, len(line))


async def mic_chunks(
    backend: LocalAudioBackend, session: VoiceSession
) -> AsyncIterator[AudioChunk]:
    """Turn the backend's callback into the stream the provider consumes.

    The capture is push (PortAudio calls us) and the provider is pull (it
    iterates), so a queue sits between them. It is unbounded on purpose: the
    socket drains faster than a microphone fills it, and dropping a frame
    would silently truncate someone's sentence.
    """
    queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_audio(_session: VoiceSession, frame: AudioFrame) -> None:
        # Called from the PortAudio thread, so the hop back onto the loop is
        # not optional.
        loop.call_soon_threadsafe(
            queue.put_nowait,
            AudioChunk(data=frame.data, sample_rate=frame.sample_rate),
        )

    # Registered before start_listening: PortAudio captures the callback set
    # at stream start, and a later registration is never seen.
    backend.on_audio_received(on_audio)
    await backend.start_listening(session)
    try:
        while True:
            yield await queue.get()
    finally:
        await backend.stop_listening(session)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Speak and watch Meta Muse transcribe.")
    parser.add_argument(
        "--diarize", action="store_true", help="label each turn with its speaker (A, B, ...)"
    )
    args = parser.parse_args()
    env = require_env("META_API_KEY")

    backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=False,
    )
    bias = os.environ.get("LANGUAGE_BIAS")
    provider = MetaSTTProvider(
        MetaSTTConfig(
            api_key=env["META_API_KEY"],
            mode="DIARIZATION" if args.diarize else "ENDPOINTING",
            language_bias=[bias] if bias else [],
            keywords=_split(os.environ.get("STT_KEYWORDS")),
        )
    )

    session = await backend.connect("mic-demo", "speaker", "stt")
    caption = Caption()
    turns = 0

    logger.info("Listening on %d Hz - speak, then pause. Ctrl+C to stop.", SAMPLE_RATE)
    try:
        async for result in provider.transcribe_stream(mic_chunks(backend, session)):
            if result.is_speech_start:
                caption.interim("...")
            elif result.is_final:
                caption.final(result.text, result.speaker)
                turns += 1
            else:
                caption.interim(result.text)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print()
    finally:
        await provider.close()
        with contextlib.suppress(Exception):
            await backend.disconnect(session)
        await backend.close()

    logger.info("%d turn(s) transcribed", turns)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
