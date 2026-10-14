"""RoomKit — DPDFNet noise suppression on the inbound audio pipeline.

Runs a noisy WAV through an ``AudioPipeline`` whose denoiser is sherpa-onnx's
DPDFNet, 20 ms frame by frame — the way a voice session's microphone audio
reaches it — writes the denoised WAV and prints what it cost and what it
removed::

    noisy.wav → 20 ms PCM frames → AudioPipeline [Denoiser: DPDFNet] → denoised.wav

DPDFNet streams: sherpa-onnx's ``OnlineSpeechDenoiser`` keeps the model's state
from one frame to the next, so the output trails the input by a constant 50 ms
(four 10 ms hops inside the model, one held back by the overlap-add). Run the
same file with ``--model-type gtcrn`` to compare with the default GTCRN model.

A 16 kHz model denoises a 16 kHz pipeline; ``dpdfnet2_48khz_hr.onnx`` a 48 kHz
one. Frames at another rate pass through unchanged, with a warning.

Models and test audio (download once, into examples/models/)::

    mkdir -p examples/models && cd examples/models
    base=https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models
    wget $base/dpdfnet2.onnx $base/gtcrn_simple.onnx $base/speech_with_noise.wav
    sha256sum dpdfnet2.onnx gtcrn_simple.onnx speech_with_noise.wav
    # ce35d6025fc71df0ef10d1540e1b7916837bbfe5f6896deb744508d2cad487a9  dpdfnet2.onnx
    # e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534  gtcrn_simple.onnx
    # 9354345a6297f4522e690d337546aa9a686a7e61eefcd935478a2141b924db8f  speech_with_noise.wav
    cd ../..

The same release has ``dpdfnet_baseline.onnx`` (lighter), ``dpdfnet4.onnx``,
``dpdfnet8.onnx`` and ``dpdfnet2_48khz_hr.onnx``.

Requires:
    pip install roomkit[sherpa-onnx]   # sherpa-onnx 1.12.30 or later

Run with:
    uv run --extra sherpa-onnx python examples/dpdfnet_denoiser.py
    uv run --extra sherpa-onnx python examples/dpdfnet_denoiser.py --model-type gtcrn
    uv run --extra sherpa-onnx python examples/dpdfnet_denoiser.py --input my_noisy_16k.wav
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline import AudioPipeline, AudioPipelineConfig
from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
)

MODELS = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = {"dpdfnet": "dpdfnet2.onnx", "gtcrn": "gtcrn_simple.onnx"}
FRAME_MS = 20


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected mono 16-bit PCM")
        return w.readframes(w.getnframes()), w.getframerate()


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def quiet_frames_db(noisy: np.ndarray, denoised: np.ndarray, frame: int) -> float:
    """Energy removed from the quietest fifth of the input's frames (its noise floor)."""
    count = min(len(noisy), len(denoised)) // frame
    energy_in = np.array([np.mean(noisy[i * frame : (i + 1) * frame] ** 2) for i in range(count)])
    energy_out = np.array(
        [np.mean(denoised[i * frame : (i + 1) * frame] ** 2) for i in range(count)]
    )
    quiet = np.argsort(energy_in)[: max(count // 5, 1)]
    return float(
        10 * np.log10((energy_in[quiet].sum() + 1e-12) / (energy_out[quiet].sum() + 1e-12))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DPDFNet on the inbound audio pipeline")
    parser.add_argument("--model-type", choices=("dpdfnet", "gtcrn"), default="dpdfnet")
    parser.add_argument("--model", type=Path, help="model file (default: examples/models/...)")
    parser.add_argument("--input", type=Path, default=MODELS / "speech_with_noise.wav")
    parser.add_argument("--output", type=Path, help="denoised WAV (default: <input>.<model>.wav)")
    args = parser.parse_args()

    model = args.model or MODELS / DEFAULT_MODEL[args.model_type]
    for path in (model, args.input):
        if not path.exists():
            raise SystemExit(f"{path} is missing: see the download steps at the top of this file.")
    output = args.output or args.input.with_suffix(f".{args.model_type}.wav")

    pcm, rate = read_wav(args.input)
    denoiser = SherpaOnnxDenoiserProvider(
        SherpaOnnxDenoiserConfig(model_type=args.model_type, model=str(model))
    )
    pipeline = AudioPipeline(AudioPipelineConfig(denoiser=denoiser))

    frame_bytes = rate * FRAME_MS // 1000 * 2
    out, timings = bytearray(), []
    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frame = AudioFrame(
            data=pcm[offset : offset + frame_bytes],
            sample_rate=rate,
            channels=1,
            sample_width=2,
            timestamp_ms=offset / 2 * 1000 / rate,
        )
        started = time.perf_counter()
        result = pipeline.process_inbound_stream("file", frame)
        timings.append(time.perf_counter() - started)
        out += result.frame.data
    pipeline.release_stream("file")
    pipeline.close()
    write_wav(output, bytes(out), rate)

    seconds = len(out) / 2 / rate
    steady = np.array(timings[1:]) * 1000  # the first frame loads the model
    noisy = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    denoised = np.frombuffer(bytes(out), dtype=np.int16).astype(np.float32)
    delay = rate * 50 // 1000 if args.model_type == "dpdfnet" else 0
    print(f"{args.model_type} ({model.name}) on {args.input.name}: {seconds:.1f} s at {rate} Hz")
    print(f"  model load (first frame): {timings[0] * 1000:.0f} ms")
    print(
        f"  per {FRAME_MS} ms frame: mean {steady.mean():.2f} ms, "
        f"p95 {np.percentile(steady, 95):.2f} ms, real-time factor {steady.sum() / 1000 / seconds:.3f}"
    )
    print(
        f"  noise floor lowered by {quiet_frames_db(noisy, denoised[delay:], frame_bytes // 2):.1f} dB"
    )
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()
