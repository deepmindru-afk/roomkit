"""Opt-in Voxtral 3 smoke tests: speech rendered by Voxtral, transcribed back by it.

Run with MISTRAL_API_KEY and ROOMKIT_RUN_VOXTRAL_LIVE=1 in the environment:
    uv run --extra mistral-voice pytest tests/test_integration/test_voxtral_live.py -q

These tests make paid API requests (a few cents). No credentials are logged.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator

import pytest

from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.voxtral import VoxtralSTTConfig, VoxtralSTTProvider
from roomkit.voice.tts.voxtral import SAMPLE_RATE, VoxtralTTSConfig, VoxtralTTSProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("ROOMKIT_RUN_VOXTRAL_LIVE") != "1" or not os.environ.get("MISTRAL_API_KEY"),
    reason="Set ROOMKIT_RUN_VOXTRAL_LIVE=1 and MISTRAL_API_KEY to run live tests",
)

_LINES = [
    "Hello, I need to change my delivery address.",
    "It should go to the office instead, on the fourth floor.",
]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _key() -> str:
    return os.environ["MISTRAL_API_KEY"]


async def _speech(text: str, voice: str | None = None) -> bytes:
    tts = VoxtralTTSProvider(VoxtralTTSConfig(api_key=_key()))
    try:
        return b"".join([chunk.data async for chunk in tts.synthesize_stream(text, voice=voice)])
    finally:
        await tts.close()


async def _paced(pcm: bytes) -> AsyncIterator[AudioChunk]:
    """The audio at its own pace, 100 ms at a time, as a microphone sends it."""
    step = SAMPLE_RATE // 10 * 2
    for start in range(0, len(pcm), step):
        yield AudioChunk(data=pcm[start : start + step], sample_rate=SAMPLE_RATE)
        await asyncio.sleep(0.1)


async def _stream(stt: VoxtralSTTProvider, pcm: bytes) -> list[TranscriptionResult]:
    async def read() -> list[TranscriptionResult]:
        return [result async for result in stt.transcribe_stream(_paced(pcm))]

    return await asyncio.wait_for(read(), timeout=60)


async def test_file_transcription_reads_the_speech_back() -> None:
    stt = VoxtralSTTProvider(VoxtralSTTConfig(api_key=_key(), language="en"))
    try:
        result = await stt.transcribe(
            AudioChunk(data=await _speech(_LINES[0]), sample_rate=SAMPLE_RATE)
        )
    finally:
        await stt.close()

    assert _words(result.text) == _words(_LINES[0])


async def test_realtime_turn_detection_gives_one_final_per_turn() -> None:
    silence = b"\0" * (SAMPLE_RATE * 2 * 3 // 2)  # 1.5 s between the turns
    pcm = silence.join([await _speech(line) for line in _LINES]) + silence * 2
    results = await _stream(VoxtralSTTProvider(VoxtralSTTConfig(api_key=_key())), pcm)

    finals = [r.text for r in results if r.is_final]
    assert [_words(text) for text in finals] == [_words(line) for line in _LINES]
    # The next turn can start while the last one waits for its last words.
    kinds = [
        "start" if r.is_speech_start else "final"
        for r in results
        if r.is_final or r.is_speech_start
    ]
    assert kinds.count("start") == 2
    assert kinds[0] == "start"


async def test_realtime_without_turn_detection_gives_one_final() -> None:
    pcm = await _speech(_LINES[1]) + b"\0" * SAMPLE_RATE
    stt = VoxtralSTTProvider(VoxtralSTTConfig(api_key=_key(), turn_detection=None))
    results = await _stream(stt, pcm)

    finals = [r.text for r in results if r.is_final]
    assert [_words(text) for text in finals] == [_words(_LINES[1])]


async def test_a_french_voice_is_transcribed_back_in_french() -> None:
    text = "Bonjour, votre commande arrivera jeudi matin."
    stt = VoxtralSTTProvider(VoxtralSTTConfig(api_key=_key()))
    try:
        result = await stt.transcribe(
            AudioChunk(data=await _speech(text, voice="voice_3137"), sample_rate=SAMPLE_RATE),
            language="fr",
        )
    finally:
        await stt.close()

    assert _words(result.text) == _words(text)


async def test_the_offline_catalog_matches_the_live_presets() -> None:
    tts = VoxtralTTSProvider(VoxtralTTSConfig(api_key=_key()))
    try:
        live = {v.id for v in await tts.list_voices() if v.attributes.get("kind") == "pvc"}
    finally:
        await tts.close()

    assert {v.id for v in VoxtralTTSProvider.available_voices()} <= live
