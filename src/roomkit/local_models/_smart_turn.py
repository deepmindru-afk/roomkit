"""Smart Turn models: audio-native end-of-turn detection."""

from __future__ import annotations

from roomkit.local_models._model import SMART_TURN, LocalModel, ModelFile

_SMART_TURN_V3 = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3"
    "/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
)

SMART_TURN_MODELS = (
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
)
