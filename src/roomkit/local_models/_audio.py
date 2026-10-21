"""Pipeline models: VAD and denoiser (sherpa-onnx), end-of-turn (Smart Turn)."""

from __future__ import annotations

from roomkit.local_models._model import (
    SHERPA_ONNX_DENOISER,
    SHERPA_ONNX_VAD,
    SMART_TURN,
    LocalModel,
    ModelFile,
)

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_SMART_TURN_V3 = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3"
    "/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
)


def _dpdfnet(name: str, sha256: str, *, description: str) -> LocalModel:
    return LocalModel(
        name=name.replace("_", "-"),
        kind=SHERPA_ONNX_DENOISER,
        description=description,
        size_mb=11,
        files=(ModelFile(f"{_RELEASES}/speech-enhancement-models/{name}.onnx", sha256),),
        paths={"model": f"{name}.onnx"},
        settings={"model_type": "dpdfnet"},
    )


PIPELINE_MODELS = (
    LocalModel(
        name="ten-vad",
        kind=SHERPA_ONNX_VAD,
        description="Voice activity detection, TEN-VAD",
        size_mb=1,
        files=(
            ModelFile(
                f"{_RELEASES}/asr-models/ten-vad.onnx",
                "718cb7eef47e3cf5ddbe7e967a7503f46b8b469c0706872f494dfa921b486206",
            ),
        ),
        paths={"model": "ten-vad.onnx"},
        settings={"model_type": "ten"},
    ),
    LocalModel(
        name="silero-vad",
        kind=SHERPA_ONNX_VAD,
        description="Voice activity detection, Silero VAD",
        size_mb=1,
        files=(
            ModelFile(
                f"{_RELEASES}/asr-models/silero_vad.onnx",
                "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
            ),
        ),
        paths={"model": "silero_vad.onnx"},
        settings={"model_type": "silero"},
    ),
    LocalModel(
        name="gtcrn-simple",
        kind=SHERPA_ONNX_DENOISER,
        description="Speech enhancement (denoiser), GTCRN",
        size_mb=1,
        files=(
            ModelFile(
                f"{_RELEASES}/speech-enhancement-models/gtcrn_simple.onnx",
                "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534",
            ),
        ),
        paths={"model": "gtcrn_simple.onnx"},
        settings={"model_type": "gtcrn"},
    ),
    # DPDFNet denoises at its model's own rate only: pick the one of the pipeline.
    _dpdfnet(
        "dpdfnet2",
        "ce35d6025fc71df0ef10d1540e1b7916837bbfe5f6896deb744508d2cad487a9",
        description="Speech enhancement (denoiser), DPDFNet2, 16 kHz, streaming",
    ),
    _dpdfnet(
        "dpdfnet2_48khz_hr",
        "0b399f8a58dc4d70d8cd97541f5c39869406145193b957d00a03b66070944928",
        description="Speech enhancement (denoiser), DPDFNet2, 48 kHz, streaming",
    ),
    LocalModel(
        name="smart-turn-v3.2-cpu",
        kind=SMART_TURN,
        description="End-of-turn detection, Smart Turn v3.2 (CPU build), multilingual",
        size_mb=9,
        files=(
            ModelFile(
                f"{_SMART_TURN_V3}/smart-turn-v3.2-cpu.onnx",
                "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
            ),
        ),
        paths={"model_path": "smart-turn-v3.2-cpu.onnx"},
    ),
    LocalModel(
        name="smart-turn-v3.2-gpu",
        kind=SMART_TURN,
        description='End-of-turn detection, Smart Turn v3.2 (fp32, for provider="cuda")',
        size_mb=32,
        files=(
            ModelFile(
                f"{_SMART_TURN_V3}/smart-turn-v3.2-gpu.onnx",
                "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be",
            ),
        ),
        paths={"model_path": "smart-turn-v3.2-gpu.onnx"},
    ),
)
