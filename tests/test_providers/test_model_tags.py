"""The speech tags a listing reads off a model id (RMK-389).

The ids are the ones OpenAI, Gemini and Mistral listed on 2026-10-02.
"""

from __future__ import annotations

import pytest

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.ai.model_tags import (
    SPEECH_CAPABILITY,
    TRANSCRIPTION_CAPABILITY,
    speech_tags,
    with_speech_tags,
)

TRANSCRIBING = [
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-transcribe-diarize",
    "gpt-transcribe",
    "gpt-live-transcribe",
    "gpt-realtime-whisper",
    "gemini-3.5-transcribe",
    "gemini-3.5-transcribe-live",
    "voxtral-mini-transcribe-realtime-2602",
    "whisper-large-v3-turbo",
    "cohere-transcribe-03-2026",
    "openai/whisper-1",
]
SPEAKING = [
    "tts-1",
    "tts-1-hd-1106",
    "gpt-4o-mini-tts",
    "gpt-4o-mini-tts-2025-12-15",
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-3.8-flash-lite-tts",
    "voxtral-mini-tts-latest",
]
NEITHER = [
    "gpt-audio",
    "gpt-audio-mini-2025-12-15",
    "gpt-realtime-2.1",
    "gpt-realtime-translate",
    "gemini-2.5-flash-native-audio-latest",
    "lyria-realtime-exp",
    "voxtral-mini-latest",
    "voxtral-small-2507",
    "voxtral-mini-realtime-2602",
    "gpt-5.4",
    "qwen-3.8-27b",
]


@pytest.mark.parametrize("model_id", TRANSCRIBING)
def test_a_speech_to_text_model_is_tagged_transcription(model_id: str) -> None:
    assert speech_tags(model_id) == [TRANSCRIPTION_CAPABILITY]


@pytest.mark.parametrize("model_id", SPEAKING)
def test_a_text_to_speech_model_is_tagged_speech(model_id: str) -> None:
    assert speech_tags(model_id) == [SPEECH_CAPABILITY]


@pytest.mark.parametrize("model_id", NEITHER)
def test_a_model_that_converses_is_not_tagged(model_id: str) -> None:
    assert speech_tags(model_id) == []


def test_a_listing_keeps_the_tags_it_reported() -> None:
    listed = [
        ModelInfo(id="whisper-1"),
        ModelInfo(id="tts-1", capabilities=["completion"]),
        ModelInfo(id="gpt-5.4"),
    ]

    tagged = with_speech_tags(listed)

    assert [m.capabilities for m in tagged] == [["transcription"], ["completion"], []]
