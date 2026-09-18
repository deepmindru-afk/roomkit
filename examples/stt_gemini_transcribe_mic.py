"""RoomKit - speak into your microphone and watch the text appear.

    GEMINI_API_KEY=... uv run --extra realtime-gemini --extra local-audio \
        python examples/stt_gemini_transcribe_mic.py

Live speech-to-text on ``gemini-3.5-transcribe-live``, fed by your own voice
rather than a file. The line at the bottom of the terminal rewrites itself
while you speak: that is the interim transcript being revised as the model
hears more. When you stop, the line is committed and a new one starts.

This is the example that shows what a *streaming* recogniser is for. The batch
provider (:mod:`roomkit.voice.stt.gemini`) would hand you the same words, but
only once you had finished and sent the recording - no good if something has
to react while the caller is still talking.

Real speech also exercises what a synthesized sentence cannot: hesitation,
accent, background noise, and the model deciding on its own which language you
are in (``language_codes=[]``).

Requires:
    pip install roomkit[realtime-gemini,local-audio]

Environment variables:
    GEMINI_API_KEY   (required) Gemini API key
    STT_LANGUAGES    Comma-separated BCP-47 hints, e.g. "fr-FR,en-US".
                     Unset means the model identifies the language itself.
    STT_VOCABULARY   Comma-separated terms to bias towards.

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator

from shared import require_env, setup_logging

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.stt.gemini_transcribe import (
    REQUIRED_SAMPLE_RATE,
    GeminiTranscribeConfig,
    GeminiTranscribeProvider,
)

logger = setup_logging("roomkit.examples.stt_gemini_transcribe_mic")

BLOCK_MS = 100
"""Capture block. The model takes short chunks; 100 ms is the size Google's
own streaming guide uses and keeps the interim updates frequent."""


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

    def final(self, text: str) -> None:
        self._write(f"> {text}")
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
    env = require_env("GEMINI_API_KEY")

    backend = LocalAudioBackend(
        # Capture straight at the rate the model documents: no resampling
        # stage, and nothing to misconfigure.
        input_sample_rate=REQUIRED_SAMPLE_RATE,
        output_sample_rate=REQUIRED_SAMPLE_RATE,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=False,
    )
    provider = GeminiTranscribeProvider(
        GeminiTranscribeConfig(
            api_key=env["GEMINI_API_KEY"],
            language_codes=_split(os.environ.get("STT_LANGUAGES")),
            custom_vocabulary=_split(os.environ.get("STT_VOCABULARY")),
        )
    )

    session = await backend.connect("mic-demo", "speaker", "stt")
    caption = Caption()
    finals: list[str] = []

    logger.info("Listening on %d Hz - speak, then pause. Ctrl+C to stop.", REQUIRED_SAMPLE_RATE)
    try:
        async for result in provider.transcribe_stream(mic_chunks(backend, session)):
            if not result.text.strip():
                continue
            if result.is_final:
                caption.final(result.text)
                finals.append(result.text)
            else:
                caption.interim(result.text)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print()
    finally:
        await provider.close()
        with contextlib.suppress(Exception):
            await backend.disconnect(session)
        await backend.close()

    logger.info("Heard: %s", " ".join(finals).strip() or "(nothing)")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
