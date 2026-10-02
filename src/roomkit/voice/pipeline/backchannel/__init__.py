"""Backchannel detection providers."""

from roomkit.voice.pipeline.backchannel.base import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)
from roomkit.voice.pipeline.backchannel.mock import MockBackchannelDetector
from roomkit.voice.pipeline.backchannel.phrases import (
    ENGLISH_BACKCHANNELS,
    FRENCH_BACKCHANNELS,
    PhraseBackchannelDetector,
)

__all__ = [
    "BackchannelContext",
    "BackchannelDecision",
    "BackchannelDetector",
    "MockBackchannelDetector",
    "PhraseBackchannelDetector",
    "ENGLISH_BACKCHANNELS",
    "FRENCH_BACKCHANNELS",
]
