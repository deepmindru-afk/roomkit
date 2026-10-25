"""Audio processing pipeline for voice (RFC §12.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.voice.pipeline.aec import AECProvider, MockAECProvider, SpeexAECProvider
from roomkit.voice.pipeline.agc import AGCConfig, AGCProvider, MockAGCProvider, SimpleAGCProvider
from roomkit.voice.pipeline.backchannel import (
    ENGLISH_BACKCHANNELS,
    FRENCH_BACKCHANNELS,
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
    MockBackchannelDetector,
    PhraseBackchannelDetector,
)
from roomkit.voice.pipeline.config import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
)
from roomkit.voice.pipeline.debug_taps import PipelineDebugTaps
from roomkit.voice.pipeline.denoiser import (
    AICousticsDenoiserConfig,
    AICousticsDenoiserProvider,
    DenoiserProvider,
    MockDenoiserProvider,
    RNNoiseDenoiserProvider,
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
    WebRTCNoiseSuppressorProvider,
)
from roomkit.voice.pipeline.diarization import (
    DiarizationProvider,
    DiarizationResult,
    MockDiarizationProvider,
)
from roomkit.voice.pipeline.dtmf import (
    DTMFDetector,
    DTMFEvent,
    DTMFRedaction,
    MockDTMFDetector,
)
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.mixer import MixerProvider, PythonMixerProvider
from roomkit.voice.pipeline.postprocessor import AudioPostProcessor
from roomkit.voice.pipeline.recorder import (
    AudioRecorder,
    MockAudioRecorder,
    RecordingChannelMode,
    RecordingConfig,
    RecordingEncryption,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
    RecordingTrigger,
    WavFileRecorder,
)
from roomkit.voice.pipeline.resampler import (
    LinearResamplerProvider,
    MockResamplerProvider,
    ResamplerProvider,
)
from roomkit.voice.pipeline.turn import (
    MockTurnDetector,
    TurnContext,
    TurnDecision,
    TurnDetector,
    TurnEntry,
)
from roomkit.voice.pipeline.vad import (
    EnergyVADProvider,
    MockVADProvider,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
    VADConfig,
    VADEvent,
    VADEventType,
    VADProvider,
)

if TYPE_CHECKING:
    from roomkit.voice.pipeline.mixer.numpy import NumpyMixerProvider
else:
    # NumpyMixerProvider loads on first use: numpy is optional.
    __getattr__, __dir__ = lazy_exports(
        __name__, {"NumpyMixerProvider": "roomkit.voice.pipeline.mixer.numpy"}
    )

__all__ = [
    # Config
    "AudioFormat",
    "AudioPipelineConfig",
    "AudioPipelineContract",
    "PipelineDebugTaps",
    # Engine
    "AudioPipeline",
    # Provider ABCs + implementations
    "MixerProvider",
    "PythonMixerProvider",
    "NumpyMixerProvider",
    "AECProvider",
    "SpeexAECProvider",
    "EnergyVADProvider",
    "SherpaOnnxVADConfig",
    "SherpaOnnxVADProvider",
    "AGCProvider",
    "SimpleAGCProvider",
    "AudioPostProcessor",
    "ResamplerProvider",
    "LinearResamplerProvider",
    "AudioRecorder",
    "WavFileRecorder",
    "BackchannelDetector",
    "AICousticsDenoiserConfig",
    "AICousticsDenoiserProvider",
    "DenoiserProvider",
    "RNNoiseDenoiserProvider",
    "SherpaOnnxDenoiserConfig",
    "SherpaOnnxDenoiserProvider",
    "WebRTCNoiseSuppressorProvider",
    "DiarizationProvider",
    "DTMFDetector",
    "DTMFRedaction",
    "TurnDetector",
    "VADProvider",
    # Data types
    "AGCConfig",
    "BackchannelContext",
    "BackchannelDecision",
    "DiarizationResult",
    "DTMFEvent",
    "RecordingChannelMode",
    "RecordingConfig",
    "RecordingEncryption",
    "RecordingHandle",
    "RecordingMode",
    "RecordingResult",
    "RecordingTrigger",
    "TurnContext",
    "TurnDecision",
    "TurnEntry",
    "VADConfig",
    "VADEvent",
    "VADEventType",
    # Mocks
    "MockAECProvider",
    "MockAGCProvider",
    "MockAudioRecorder",
    "MockBackchannelDetector",
    "PhraseBackchannelDetector",
    "ENGLISH_BACKCHANNELS",
    "FRENCH_BACKCHANNELS",
    "MockDenoiserProvider",
    "MockDiarizationProvider",
    "MockDTMFDetector",
    "MockResamplerProvider",
    "MockTurnDetector",
    "MockVADProvider",
]
