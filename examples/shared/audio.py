"""Shared audio pipeline builders for RoomKit examples.

All builder functions use lazy imports so that missing optional
dependencies only cause a warning rather than crashing the example.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import wave

from roomkit.voice import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline.backchannel import PhraseBackchannelDetector

from .models import require_local_models

logger = logging.getLogger(__name__)


def pcm_from_wav_url(url: str) -> tuple[bytes, int]:
    """The PCM frames and sample rate of a ``data:audio/wav`` URL.

    Read through the WAV chunks rather than by dropping a 44-byte header: a
    Gemini 3.8 TTS answer carries a C2PA chunk after its audio, which would
    otherwise play as a burst of noise.
    """
    wav = base64.b64decode(url.split(",", 1)[1])
    with wave.open(io.BytesIO(wav), "rb") as reader:
        return reader.readframes(reader.getnframes()), reader.getframerate()


def wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """16-bit PCM wrapped in a WAV container, to write to a ``.wav`` file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# AEC
# ---------------------------------------------------------------------------


def build_aec(
    sample_rate: int,
    block_ms: int = 20,
    *,
    default: str = "webrtc",
    enable_ns: bool = False,
) -> object | None:
    """Build an AEC provider based on the ``AEC`` env var.

    Env: ``AEC=webrtc|speex|1|0`` (default comes from *default* param).

    * ``webrtc`` / ``1`` — WebRTC AEC3 (``pip install aec-audio-processing``)
    * ``speex`` — SpeexDSP (``apt install libspeexdsp1``)
    * ``0`` — disabled

    Returns the provider instance or ``None``.
    """
    aec_mode = os.environ.get("AEC", default).lower()
    if aec_mode == "0":
        logger.info("AEC disabled (AEC=0)")
        return None

    if aec_mode in ("1", "webrtc"):
        try:
            from roomkit.voice.pipeline.aec import WebRTCAECProvider

            try:
                stream_delay_ms = max(0, int(os.environ.get("AEC_DELAY_MS", "0")))
            except ValueError:
                logger.warning("Invalid AEC_DELAY_MS; using automatic delay configuration")
                stream_delay_ms = 0
            logger.info(
                "AEC enabled (WebRTC AEC3%s, delay=%s)",
                " + NS" if enable_ns else "",
                f"{stream_delay_ms}ms" if stream_delay_ms else "auto",
            )
            return WebRTCAECProvider(
                sample_rate=sample_rate,
                enable_ns=enable_ns,
                stream_delay_ms=stream_delay_ms,
            )
        except ImportError:
            print("\n  >>> Install AEC: pip install aec-audio-processing <<<\n")
            return None

    if aec_mode == "speex":
        try:
            from roomkit.voice.pipeline.aec import SpeexAECProvider

            frame_size = sample_rate * block_ms // 1000
            logger.info("AEC enabled (Speex)")
            return SpeexAECProvider(
                frame_size=frame_size,
                filter_length=frame_size * 10,
                sample_rate=sample_rate,
            )
        except ImportError:
            print("\n  >>> Install Speex: apt install libspeexdsp1 <<<\n")
            return None

    logger.warning("Unknown AEC mode %r — disabling", aec_mode)
    return None


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


def build_denoiser(sample_rate: int = 16000, *, default: str = "0") -> object | None:
    """Build a denoiser provider based on the ``DENOISE`` env var.

    Env: ``DENOISE=webrtc|rnnoise|sherpa|dpdfnet|1|0`` (default comes from
    *default* param). For ``sherpa`` (GTCRN) and ``dpdfnet``, the model is a
    downloaded local model (``gtcrn-simple``; for DPDFNet, ``dpdfnet2-48khz-hr``
    on a 48 kHz pipeline, else ``dpdfnet2``); ``DENOISE_MODEL`` sets a model
    file instead.
    DPDFNet denoises only at its model's rate: on another pipeline rate it
    would pass the audio through, so it is left out with a warning (GTCRN
    resamples, ``DENOISE=sherpa``).

    Returns the provider instance or ``None``.
    """
    mode = os.environ.get("DENOISE", default).lower()
    if mode == "0":
        return None

    # "1" resolves to the default backend
    if mode == "1":
        mode = default

    if mode == "webrtc":
        try:
            from roomkit.voice.pipeline.denoiser import WebRTCNoiseSuppressorProvider

            denoiser = WebRTCNoiseSuppressorProvider(sample_rate=sample_rate)
            logger.info("Denoiser enabled (WebRTC NS)")
            return denoiser
        except ImportError:
            logger.warning("aec-audio-processing not installed — denoiser disabled")
            return None

    if mode in ("sherpa", "dpdfnet"):
        model_type = "dpdfnet" if mode == "dpdfnet" else "gtcrn"
        if model_type == "gtcrn":
            local_model = "gtcrn-simple"
        elif sample_rate == 48000:
            local_model = "dpdfnet2-48khz-hr"
        else:
            local_model = "dpdfnet2"
        model = os.environ.get("DENOISE_MODEL") or local_model
        try:
            from roomkit.voice.pipeline.denoiser import (
                SherpaOnnxDenoiserConfig,
                SherpaOnnxDenoiserProvider,
            )

            if "DENOISE_MODEL" in os.environ:
                config = SherpaOnnxDenoiserConfig(model_type=model_type, model=model)
            else:
                require_local_models(local_model)
                config = SherpaOnnxDenoiserConfig.from_local_model(local_model)
            denoiser = SherpaOnnxDenoiserProvider(config)
        except (ImportError, ValueError) as exc:
            logger.warning("%s — denoiser disabled", exc)
            return None
        model_rate = denoiser.model_sample_rate
        if model_type == "dpdfnet" and model_rate not in (None, sample_rate):
            logger.warning(
                "DPDFNet model %s runs at %s Hz and this pipeline at %d Hz: it would "
                "pass the audio through — denoiser disabled (DENOISE=sherpa resamples)",
                model,
                model_rate,
                sample_rate,
            )
            return None
        logger.info("Denoiser enabled (sherpa-onnx %s, model=%s)", model_type, model)
        return denoiser

    if mode == "rnnoise":
        try:
            from roomkit.voice.pipeline.denoiser import RNNoiseDenoiserProvider

            denoiser = RNNoiseDenoiserProvider(sample_rate=sample_rate)
            logger.info("Denoiser enabled (RNNoise)")
            return denoiser
        except ImportError:
            logger.warning("RNNoise not installed — denoiser disabled")
            return None

    logger.warning("Unknown DENOISE mode %r — disabling", mode)
    return None


# ---------------------------------------------------------------------------
# Debug taps
# ---------------------------------------------------------------------------


def build_debug_taps() -> object | None:
    """Build pipeline debug taps based on the ``DEBUG_AUDIO`` env var.

    Env: ``DEBUG_AUDIO=1|0`` (default ``0``).

    Returns a :class:`PipelineDebugTaps` or ``None``.
    """
    if os.environ.get("DEBUG_AUDIO", "0") != "1":
        return None
    from roomkit.voice.pipeline import PipelineDebugTaps

    logger.info("Debug audio taps enabled → ./debug_audio/")
    return PipelineDebugTaps(output_dir="./debug_audio/", stages=["all"])


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def build_vad(sample_rate: int = 24000, *, default: str = "energy") -> object | None:
    """Build a VAD provider based on the ``VAD`` env var.

    Env: ``VAD=energy|silero|ten|1|0`` (default comes from *default* param).
    ``silero`` and ``ten`` run the downloaded local model (``silero-vad``,
    ``ten-vad``); ``VAD_MODEL`` sets a model file instead.

    * ``energy`` / ``1`` — Energy-based VAD (no dependencies)
    * ``silero`` — Silero VAD via sherpa-onnx (``pip install roomkit[sherpa-onnx]``)
    * ``ten`` — TEN-VAD via sherpa-onnx (``pip install roomkit[sherpa-onnx]``)
    * ``0`` — disabled

    Returns the provider instance or ``None``.
    """
    mode = os.environ.get("VAD", default).lower()
    if mode == "0":
        return None
    if mode == "1":
        mode = default

    if mode == "energy":
        from roomkit.voice.pipeline.vad import EnergyVADProvider

        logger.info("VAD enabled (Energy-based)")
        return EnergyVADProvider()

    if mode in ("silero", "ten"):
        try:
            from roomkit.voice.pipeline.vad import SherpaOnnxVADConfig, SherpaOnnxVADProvider

            model_file = os.environ.get("VAD_MODEL")
            if model_file:
                config = SherpaOnnxVADConfig(
                    model=model_file, model_type=mode, sample_rate=sample_rate
                )
            else:
                local_model = f"{mode}-vad"
                require_local_models(local_model)
                config = SherpaOnnxVADConfig.from_local_model(local_model, sample_rate=sample_rate)
            logger.info("VAD enabled (sherpa-onnx %s, model=%s)", mode, config.model)
            return SherpaOnnxVADProvider(config)
        except ImportError:
            logger.warning("sherpa-onnx not installed — falling back to energy VAD")
            from roomkit.voice.pipeline.vad import EnergyVADProvider

            logger.info("VAD enabled (Energy-based, fallback)")
            return EnergyVADProvider()

    # Legacy alias
    if mode == "sherpa":
        return build_vad(sample_rate, default="ten")

    logger.warning("Unknown VAD mode %r — disabling", mode)
    return None


def build_interruption(*, default: str = "semantic") -> InterruptionConfig:
    """Build the barge-in policy based on the ``INTERRUPTION`` env var.

    Env: ``INTERRUPTION=semantic|words|confirmed|immediate|disabled`` (default
    from *default*); ``INTERRUPTION_WAIT_MS`` for how long ``semantic`` and
    ``words`` wait for the first words (default 2000).

    * ``semantic`` — the bot keeps talking through an acknowledgement
      ("okay", "mm-hmm", "d'accord") and stops for anything else, judged on
      the words a streaming STT hears (``PhraseBackchannelDetector``)
    * ``words`` — as ``semantic``, and sound without words (echo, a cough,
      room noise) never stops it: only words interrupt
    * ``confirmed`` — stops once the user has spoken for 300 ms
    * ``immediate`` — stops at the first sound taken for speech
    * ``disabled`` — never stops; the user's speech waits its turn

    Returns an :class:`InterruptionConfig` for ``VoiceChannel(interruption=...)``.
    """
    mode = os.environ.get("INTERRUPTION", default).lower()
    words_only = mode == "words"
    strategy = InterruptionStrategy.SEMANTIC if words_only else InterruptionStrategy(mode)
    detector = (
        PhraseBackchannelDetector(cut_without_words=not words_only)
        if strategy == InterruptionStrategy.SEMANTIC
        else None
    )
    # How long SEMANTIC waits for the first words before judging on duration.
    # A streaming transducer gives a short word ("okay", "no") 1-1.5 s after it
    # starts, or only at the end (Nemotron, measured): speech that ends sooner
    # is judged on its final words instead.
    wait_ms = int(os.environ.get("INTERRUPTION_WAIT_MS", "2000"))
    logger.info("Barge-in: %s (words awaited up to %d ms)", mode, wait_ms)
    return InterruptionConfig(
        strategy=strategy, backchannel_detector=detector, transcript_wait_ms=wait_ms
    )


def build_turn_detector(*, default: str = "0") -> object | None:
    """Build a turn detector based on the ``TURN_DETECTOR`` env var.

    Env: ``TURN_DETECTOR=smart-turn|1|0`` (default comes from *default* param).
    The model is the downloaded local model ``smart-turn-v3.2-cpu``;
    ``TURN_MODEL`` sets an ONNX model file instead.
    ``TURN_THRESHOLD`` sets the completion probability threshold (default ``0.5``).

    * ``smart-turn`` / ``1`` — pipecat-ai/smart-turn ONNX model
      (``pip install roomkit[smart-turn]``)
    * ``0`` — disabled

    Returns the detector instance or ``None``.
    """
    mode = os.environ.get("TURN_DETECTOR", default).lower()
    if mode == "0":
        return None
    if mode == "1":
        mode = "smart-turn"

    if mode == "smart-turn":
        model_file = os.environ.get("TURN_MODEL")
        threshold = float(os.environ.get("TURN_THRESHOLD", "0.5"))
        try:
            from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector

            if model_file:
                config = SmartTurnConfig(model_path=model_file, threshold=threshold)
            else:
                require_local_models("smart-turn-v3.2-cpu")
                config = SmartTurnConfig.from_local_model(
                    "smart-turn-v3.2-cpu", threshold=threshold
                )
            logger.info(
                "Turn detector enabled (smart-turn, model=%s, threshold=%.2f)",
                config.model_path,
                threshold,
            )
            return SmartTurnDetector(config)
        except ImportError:
            logger.warning("smart-turn deps not installed — turn detector disabled")
            return None
        except Exception:
            logger.exception("Failed to create SmartTurnDetector — disabled")
            return None

    logger.warning("Unknown TURN_DETECTOR mode %r — disabling", mode)
    return None


def build_pipeline(
    *,
    aec: object | None = None,
    denoiser: object | None = None,
    debug_taps: object | None = None,
    **kwargs: object,
) -> object | None:
    """Build an :class:`AudioPipelineConfig` if any stage is set.

    Forwards all keyword arguments to the ``AudioPipelineConfig``
    constructor.  Returns ``None`` when every stage is ``None`` / empty.
    """
    all_values = {"aec": aec, "denoiser": denoiser, "debug_taps": debug_taps, **kwargs}
    if not any(v for v in all_values.values()):
        return None
    from roomkit.voice.pipeline import AudioPipelineConfig

    return AudioPipelineConfig(**all_values)  # type: ignore[arg-type]
