"""Speech enhancement provider using sherpa-onnx's GTCRN and DPDFNet models.

GTCRN (the default) runs sherpa-onnx's ``OfflineSpeechDenoiser`` over a short
sliding window of preceding frames.  DPDFNet runs its ``OnlineSpeechDenoiser``,
which keeps the model's recurrent state across frames.  Unlike RNNoise (which
requires a system library), sherpa-onnx is a pure-Python dependency::

    pip install roomkit[sherpa-onnx]

Usage::

    from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
        SherpaOnnxDenoiserConfig,
        SherpaOnnxDenoiserProvider,
    )

    denoiser = SherpaOnnxDenoiserProvider(
        SherpaOnnxDenoiserConfig(model="gtcrn_simple.onnx")
    )
    # or, on a 16 kHz pipeline:
    denoiser = SherpaOnnxDenoiserProvider(
        SherpaOnnxDenoiserConfig(model_type="dpdfnet", model="dpdfnet2.onnx")
    )
    config = AudioPipelineConfig(denoiser=denoiser)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("gtcrn", "dpdfnet")
_DEFAULT_CONTEXT_FRAMES = 3

# DPDFNet's streaming output trails its input by 50 ms: four 10 ms hops inside
# the model, plus the hop sherpa-onnx holds back until the next one completes
# its overlap-add.  The silence gate keeps the model running this long after
# the input goes quiet, twice that delay, so the tail of an utterance comes out
# before the model stops being fed.
_DPDFNET_TAIL_MS = 100


def _pcm_s16le_to_float32(data: bytes) -> np.ndarray[Any, Any]:
    """Convert PCM signed 16-bit little-endian bytes to float32 array in [-1, 1]."""
    import numpy as np

    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _float32_to_pcm_s16le(samples: np.ndarray[Any, Any] | list[float]) -> bytes:
    """Convert float32 array in [-1, 1] to PCM signed 16-bit little-endian bytes."""
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    return bytes(np.clip(arr * 32767, -32767, 32767).astype(np.int16).tobytes())


def _resample_linear(
    samples: np.ndarray[Any, Any], src_rate: int, dst_rate: int
) -> np.ndarray[Any, Any]:
    """Resample float32 samples using numpy linear interpolation."""
    import numpy as np

    if src_rate == dst_rate or len(samples) == 0:
        return samples
    ratio = dst_rate / src_rate
    n_out = int(len(samples) * ratio)
    if n_out == 0:
        return np.array([], dtype=np.float32)
    x_old = np.arange(len(samples), dtype=np.float64)
    x_new = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
    result: np.ndarray[Any, Any] = np.interp(x_new, x_old, samples).astype(np.float32)
    return result


def _with_samples(frame: AudioFrame, data: bytes) -> AudioFrame:
    """``frame`` carrying denoised ``data``, its timing and metadata kept."""
    metadata = dict(frame.metadata)
    metadata["noise_suppressed"] = True
    return AudioFrame(
        data=data,
        sample_rate=frame.sample_rate,
        channels=frame.channels,
        sample_width=frame.sample_width,
        timestamp_ms=frame.timestamp_ms,
        metadata=metadata,
    )


@dataclass
class SherpaOnnxDenoiserConfig:
    """Configuration for the sherpa-onnx denoiser (GTCRN or DPDFNet).

    Attributes:
        model: Path to the model file: ``gtcrn_simple.onnx`` for GTCRN; for
            DPDFNet, one of the exports sherpa-onnx publishes in its
            ``speech-enhancement-models`` release (``dpdfnet_baseline.onnx``,
            ``dpdfnet2.onnx``, ``dpdfnet4.onnx``, ``dpdfnet8.onnx`` at 16 kHz,
            ``dpdfnet2_48khz_hr.onnx`` at 48 kHz).  Required for DPDFNet.
        num_threads: Number of CPU threads for inference.
        provider: ONNX execution provider (``"cpu"`` or ``"cuda"``).
        context_frames: GTCRN only.  Number of preceding frames to include
            as context when denoising.  GTCRN is an offline model — processing
            tiny 20 ms frames in isolation produces poor quality and boundary
            artifacts.  A sliding context window (default 3 = 60 ms,
            ~6 STFT frames at 6.25 ms hop) gives the model's recurrent
            layers enough warmup context while keeping inference ~3x
            faster than the previous 200 ms window.  DPDFNet streams and
            keeps its own state, so a DPDFNet config refuses another value.
        silence_threshold: RMS energy threshold (in float32 [-1, 1] scale)
            below which ONNX inference is skipped entirely.  Frames below
            this level are replaced with silence.  Set to 0 to disable.
            Default 0.005 ≈ −46 dBFS — well below any speech.  DPDFNet
            stops only once the input has stayed below it for 100 ms, so the
            50 ms its output trails the input is not cut off.
        model_type: ``"gtcrn"`` (the default) or ``"dpdfnet"``.  DPDFNet
            denoises frames at the model's own rate only (16 kHz, or 48 kHz
            for ``dpdfnet2_48khz_hr``); frames at another rate pass through
            unchanged, with a warning.  Its output trails the input by a
            constant 50 ms.  Needs sherpa-onnx 1.12.30 or later.
    """

    model: str = ""
    num_threads: int = 1
    provider: str = "cpu"
    context_frames: int = _DEFAULT_CONTEXT_FRAMES
    silence_threshold: float = 0.005
    model_type: str = "gtcrn"

    def __post_init__(self) -> None:
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.context_frames <= 0:
            raise ValueError("context_frames must be positive")
        if not math.isfinite(self.silence_threshold) or self.silence_threshold < 0:
            raise ValueError("silence_threshold must be finite and non-negative")
        if self.model_type not in _MODEL_TYPES:
            raise ValueError(
                f"Unknown sherpa-onnx denoiser model_type {self.model_type!r}; expected one "
                f"of {_MODEL_TYPES}"
            )
        if self.model_type == "dpdfnet":
            if not self.model:
                raise ValueError(
                    "A DPDFNet denoiser needs its .onnx file in SherpaOnnxDenoiserConfig.model"
                )
            if self.context_frames != _DEFAULT_CONTEXT_FRAMES:
                raise ValueError(
                    "context_frames sizes GTCRN's sliding window; DPDFNet streams "
                    "and keeps its own state"
                )


@dataclass
class _StreamState:
    """One stream's denoiser and its sliding context window.

    GTCRN is fed a rolling window of preceding audio, so sharing the window
    between speakers would splice one voice into the other's context — the
    model would denoise Bob against Alice's tail.
    """

    denoiser: Any = None
    native_rate: int = 0
    context: Any = None  # np.ndarray, allocated on first use
    # DPDFNet: denoised samples not yet returned, and how long the input has
    # been below the silence threshold (samples; starts idle).
    pending: Any = None
    quiet_samples: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class SherpaOnnxDenoiserProvider(DenoiserProvider):
    """Denoiser provider using sherpa-onnx GTCRN or DPDFNet speech enhancement.

    The denoiser is created lazily on the first call to :meth:`process`.
    sherpa-onnx must be installed (``pip install roomkit[sherpa-onnx]``).

    Parameters:
        config: Provider configuration.
    """

    def __init__(self, config: SherpaOnnxDenoiserConfig) -> None:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is required for SherpaOnnxDenoiserProvider. "
                "Install it with: pip install roomkit[sherpa-onnx]"
            ) from exc

        self._config = config
        self._sherpa: Any = __import__("sherpa_onnx")
        if config.model_type == "dpdfnet" and not (
            hasattr(self._sherpa, "OnlineSpeechDenoiser")
            and hasattr(self._sherpa, "OfflineSpeechDenoiserDpdfNetModelConfig")
        ):
            raise ImportError(
                "DPDFNet needs sherpa-onnx 1.12.30 or later (installed: "
                f"{getattr(self._sherpa, '__version__', 'unknown')}). "
                "Upgrade it with: pip install -U 'roomkit[sherpa-onnx]'"
            )
        self._streams: dict[str, _StreamState] = {}
        self._streams_lock = threading.Lock()
        self._warned_formats: set[tuple[int, int, int, int]] = set()
        self._warned_rates: set[tuple[int, int]] = set()
        self._closed = False

    @property
    def name(self) -> str:
        return "SherpaOnnxDenoiser"

    def _state_for(self, stream: str) -> _StreamState | None:
        """Get or create this stream's state (without its denoiser)."""
        import numpy as np

        with self._streams_lock:
            if self._closed:
                return None
            st = self._streams.get(stream)
            if st is None:
                st = _StreamState(context=np.array([], dtype=np.float32))
                self._streams[stream] = st
            return st

    def _ensure_denoiser(self, st: _StreamState) -> None:
        """Lazily create this stream's sherpa-onnx denoiser.

        GTCRN gets an ``OfflineSpeechDenoiser`` fed a sliding window; DPDFNet
        an ``OnlineSpeechDenoiser`` that keeps the model's state between
        frames.
        """
        if st.denoiser is not None:
            return

        cfg = self._config
        sherpa = self._sherpa

        model_config = sherpa.OfflineSpeechDenoiserModelConfig(
            **self._model_section(),
            num_threads=cfg.num_threads,
            provider=cfg.provider,
        )
        if cfg.model_type == "dpdfnet":
            import numpy as np

            st.denoiser = sherpa.OnlineSpeechDenoiser(
                sherpa.OnlineSpeechDenoiserConfig(model=model_config)
            )
            # sherpa-onnx returns nothing for the first hop until the next one
            # completes its overlap-add: one hop of silence keeps every
            # returned frame as long as the frame that went in.
            st.pending = np.zeros(st.denoiser.frame_shift_in_samples, dtype=np.float32)
            st.quiet_samples = self._tail_samples(st.denoiser.sample_rate) + 1
        else:
            denoiser_config = sherpa.OfflineSpeechDenoiserConfig(
                model=model_config,
            )
            st.denoiser = sherpa.OfflineSpeechDenoiser(denoiser_config)
        st.native_rate = st.denoiser.sample_rate
        logger.debug(
            "SherpaOnnxDenoiser: created %s denoiser model=%s native_rate=%d",
            cfg.model_type,
            cfg.model,
            st.native_rate,
        )

    def _model_section(self) -> dict[str, Any]:
        """The model's own section of sherpa-onnx's ``OfflineSpeechDenoiserModelConfig``."""
        model = self._config.model
        if self._config.model_type == "dpdfnet":
            return {"dpdfnet": self._sherpa.OfflineSpeechDenoiserDpdfNetModelConfig(model=model)}
        return {"gtcrn": self._sherpa.OfflineSpeechDenoiserGtcrnModelConfig(model=model)}

    @staticmethod
    def _tail_samples(rate: int) -> int:
        return rate * _DPDFNET_TAIL_MS // 1000

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Validate and serialize one stream's recurrent model state."""
        if not self._matches_format(frame):
            return frame
        if not frame.data:
            return frame
        st = self._state_for(stream)
        if st is None:
            return frame
        with st.lock:
            if self._config.model_type == "dpdfnet":
                return self._stream_locked(frame, st)
            return self._process_locked(frame, st)

    def _stream_locked(self, frame: AudioFrame, st: _StreamState) -> AudioFrame:
        """Denoise a frame with DPDFNet's streaming denoiser.

        Every frame comes back as long as it went in, carrying the denoised
        audio of 50 ms earlier: the delay is constant, so the transport
        timeline holds.
        """
        if st.denoiser is None:
            try:
                self._ensure_denoiser(st)
            except Exception:
                logger.warning(
                    "SherpaOnnxDenoiser: failed to initialize, passing through",
                    exc_info=True,
                )
                return frame
        if frame.sample_rate != st.native_rate:
            # The model runs at its own rate, and sherpa-onnx ends the process
            # when one stream changes rate: refuse the frame instead.
            self._warn_rate(frame.sample_rate, st.native_rate)
            return frame

        try:
            import numpy as np

            samples = _pcm_s16le_to_float32(frame.data)
            if self._dpdfnet_idle(st, samples):
                # The model is not fed while the input stays quiet; its state
                # waits as it was, adapted to the room, for the next sound.
                return _with_samples(frame, b"\x00" * len(frame.data))
            result = st.denoiser.run(samples.tolist(), st.native_rate)
            denoised = np.asarray(result.samples, dtype=np.float32)
            st.pending = np.concatenate([st.pending, denoised])
            n = len(samples)
            if len(st.pending) < n:
                # A frame that is not a whole number of hops leaves part of a
                # hop inside sherpa-onnx: lengthen the delay to cover it (by
                # less than one hop over the stream's life).
                pad = np.zeros(n - len(st.pending), dtype=np.float32)
                st.pending = np.concatenate([pad, st.pending])
            out, st.pending = st.pending[:n], st.pending[n:]
            out_data = _float32_to_pcm_s16le(out)
        except Exception:
            logger.warning(
                "SherpaOnnxDenoiser: error during processing, passing through",
                exc_info=True,
            )
            return frame
        return _with_samples(frame, out_data)

    def _dpdfnet_idle(self, st: _StreamState, samples: np.ndarray[Any, Any]) -> bool:
        """Whether the input has been quiet long enough to stop feeding DPDFNet."""
        import numpy as np

        threshold = self._config.silence_threshold
        if threshold <= 0:
            return False
        rms = float(np.sqrt(np.dot(samples, samples) / len(samples)))
        if rms >= threshold:
            st.quiet_samples = 0
            return False
        st.quiet_samples += len(samples)
        return st.quiet_samples > self._tail_samples(st.native_rate)

    def _process_locked(self, frame: AudioFrame, st: _StreamState) -> AudioFrame:
        """Denoise an audio frame using GTCRN speech enhancement.

        Uses a sliding context window so the model sees preceding frames
        for temporal context.  Only the current frame's portion of the
        denoised output is returned — no latency is added.
        """
        if st.denoiser is None:
            try:
                self._ensure_denoiser(st)
            except Exception:
                logger.warning(
                    "SherpaOnnxDenoiser: failed to initialize, passing through",
                    exc_info=True,
                )
                return frame

        try:
            import numpy as np

            float_samples = _pcm_s16le_to_float32(frame.data)
            n_frame_orig = len(float_samples)

            # Resample to the denoiser's native rate if needed.
            # This avoids sherpa-onnx creating a new LinearResample
            # object on every frame (~50/s at 20ms).
            native_rate = st.native_rate or 16000
            need_resample = frame.sample_rate != native_rate and native_rate > 0
            if need_resample:
                float_samples = _resample_linear(float_samples, frame.sample_rate, native_rate)
            n_frame = len(float_samples)

            # Append current frame to sliding context buffer
            st.context = np.concatenate([st.context, float_samples])

            # Trim to at most context_frames worth of samples
            max_context = n_frame * max(self._config.context_frames, 1)
            if len(st.context) > max_context:
                st.context = st.context[-max_context:]

            # Skip inference for near-silent frames — output silence directly.
            # The denoiser would suppress this to ~silence anyway; skipping
            # avoids ONNX inference cost entirely at idle.  Context is kept
            # up-to-date so speech onset always has a warm context window.
            threshold = self._config.silence_threshold
            if threshold > 0:
                rms = float(np.sqrt(np.dot(float_samples, float_samples) / n_frame))
                if rms < threshold:
                    return _with_samples(frame, b"\x00" * len(frame.data))

            # Pad at the START to GTCRN block size (256 samples).
            # Padding must go at the beginning so the real audio stays at
            # the tail — we extract the last n_frame samples from the output.
            block = 256
            remainder = len(st.context) % block
            if remainder:
                pad = np.zeros(block - remainder, dtype=np.float32)
                to_process = np.concatenate([pad, st.context])
            else:
                to_process = st.context

            result = st.denoiser.run(to_process.tolist(), native_rate)

            # Extract the last n_frame samples (the current frame's portion)
            denoised = result.samples
            if len(denoised) >= n_frame:
                out_samples = denoised[len(denoised) - n_frame :]
            else:
                # Fallback: model returned fewer samples than expected
                out_samples = denoised

            # Resample back to original frame rate
            if need_resample:
                out_arr = np.asarray(out_samples, dtype=np.float32)
                out_samples = _resample_linear(out_arr, native_rate, frame.sample_rate)

            # Native providers must preserve the transport timeline even if a
            # model returns a short or long result for a padded context.
            out_samples = np.asarray(out_samples, dtype=np.float32)
            if len(out_samples) != n_frame_orig:
                if len(out_samples) > n_frame_orig:
                    out_samples = out_samples[:n_frame_orig]
                else:
                    out_samples = np.pad(out_samples, (0, n_frame_orig - len(out_samples)))

            out_data = _float32_to_pcm_s16le(out_samples)
        except Exception:
            logger.warning(
                "SherpaOnnxDenoiser: error during processing, passing through",
                exc_info=True,
            )
            return frame
        return _with_samples(frame, out_data)

    def _warn_rate(self, frame_rate: int, model_rate: int) -> None:
        key = (frame_rate, model_rate)
        if key in self._warned_rates:
            return
        self._warned_rates.add(key)
        logger.warning(
            "SherpaOnnxDenoiser: the DPDFNet model %s runs at %d Hz; %d Hz frames "
            "pass through undenoised. Run the pipeline at %d Hz, or use a model "
            "made for %d Hz",
            self._config.model,
            model_rate,
            frame_rate,
            model_rate,
            frame_rate,
        )

    def reset(self, stream: str) -> None:
        """Drop this stream's denoiser and its context window."""
        with self._streams_lock:
            self._streams.pop(stream, None)

    def close(self) -> None:
        """Release every stream's denoiser."""
        with self._streams_lock:
            self._closed = True
            self._streams.clear()

    def _matches_format(self, frame: AudioFrame) -> bool:
        actual = (frame.sample_rate, frame.channels, frame.sample_width)
        valid = (
            frame.sample_rate > 0
            and frame.channels == 1
            and frame.sample_width == 2
            and len(frame.data) % 2 == 0
        )
        if valid:
            return True
        key = (*actual, len(frame.data))
        if key not in self._warned_formats:
            self._warned_formats.add(key)
            logger.warning(
                "SherpaOnnxDenoiser requires mono aligned PCM16; got "
                "%dHz/%dch/%d-byte with %d bytes; frame ignored",
                *actual,
                len(frame.data),
            )
        return False
