"""The capability tags a model listing reads off what a model is.

A chat provider's live listing also surfaces its speech models, and a caller
picking a chat model must be able to set them apart. The vendors name them
consistently (measured 2026-10-02 on OpenAI, Gemini and Mistral): a
speech-to-text model's id says ``transcribe`` or ``whisper``, a text-to-speech
model's says ``tts``. A model that converses in audio (``gpt-audio``,
``gpt-realtime``, ``voxtral-small``, Gemini's native audio) is neither and
stays untagged.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # base.py imports this module
    from roomkit.providers.ai.base import ModelInfo

TRANSCRIPTION_CAPABILITY = "transcription"
"""``ModelInfo.capabilities`` tag of a speech-to-text model."""
SPEECH_CAPABILITY = "speech"
"""``ModelInfo.capabilities`` tag of a text-to-speech model."""

_TRANSCRIPTION = re.compile(r"transcrib|(?:^|[-_/])whisper(?:[-_]|$)")
_SPEECH = re.compile(r"(?:^|[-_/])tts(?:[-_]|$)")


def speech_tags(model_id: str) -> list[str]:
    """The speech tag *model_id* names itself by, or none."""
    name = model_id.lower()
    if _TRANSCRIPTION.search(name):
        return [TRANSCRIPTION_CAPABILITY]
    if _SPEECH.search(name):
        return [SPEECH_CAPABILITY]
    return []


def with_speech_tags(models: list[ModelInfo]) -> list[ModelInfo]:
    """*models*, each one the listing left untagged given its speech tag."""
    return [
        model.model_copy(update={"capabilities": tags})
        if not model.capabilities and (tags := speech_tags(model.id))
        else model
        for model in models
    ]
