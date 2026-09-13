"""Realtime voice support for speech-to-speech AI providers."""

from __future__ import annotations

from roomkit.voice.realtime.bridge import RealtimeAVBridge
from roomkit.voice.realtime.events import (
    RealtimeDelegationEvent,
    RealtimeErrorEvent,
    RealtimeSpeechEvent,
    RealtimeToolCallEvent,
    RealtimeTranscriptionEvent,
)
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.mock import (
    MockCall,
    MockRealtimeAudioVideoProvider,
    MockRealtimeProvider,
    MockRealtimeTransport,
)
from roomkit.voice.realtime.provider import (
    RealtimeAudioVideoProvider,
    RealtimeVideoCallback,
    RealtimeVoiceProvider,
    VoiceInfo,
)
from roomkit.voice.realtime.reasoning import (
    AIProviderReasoningBackend,
    ReasoningBackend,
    ReasoningOutput,
    ReasoningRequest,
    TranscriptLine,
)

__all__ = [
    "VoiceInjectionResult",
    # ABCs
    "RealtimeAudioVideoProvider",
    "RealtimeAVBridge",
    "RealtimeVideoCallback",
    "RealtimeVoiceProvider",
    "VoiceInfo",
    # Reasoning delegation (RFC §12.4.1)
    "AIProviderReasoningBackend",
    "ReasoningBackend",
    "ReasoningOutput",
    "ReasoningRequest",
    "TranscriptLine",
    # Events
    "RealtimeDelegationEvent",
    "RealtimeErrorEvent",
    "RealtimeSpeechEvent",
    "RealtimeToolCallEvent",
    "RealtimeTranscriptionEvent",
    # Mocks
    "MockCall",
    "MockRealtimeAudioVideoProvider",
    "MockRealtimeProvider",
    "MockRealtimeTransport",
]
