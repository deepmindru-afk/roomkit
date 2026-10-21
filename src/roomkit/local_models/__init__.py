"""Local model files: download once while online, then run with no network.

RoomKit knows the files of the local models its providers run (sherpa-onnx
speech-to-text, text-to-speech, VAD and denoiser, Smart Turn), each pinned to
its URL and SHA-256. Download them once::

    python -m roomkit.local_models list
    python -m roomkit.local_models download zipformer-en-2023-06-26 kokoro-int8-multi-lang-v1_0

then build a provider config by the model's name. Building it never touches
the network: a model that is not downloaded raises
:class:`ModelNotDownloadedError`, naming the command that fetches it::

    tts = SherpaOnnxTTSProvider(
        SherpaOnnxTTSConfig.from_local_model("kokoro-int8-multi-lang-v1_0", voice="af_heart")
    )

Models live in ``$ROOMKIT_MODELS_DIR`` when it is set, else in
``~/.cache/roomkit/models`` (``$XDG_CACHE_HOME/roomkit/models``).
"""

from __future__ import annotations

from roomkit.local_models._config import FromLocalModel
from roomkit.local_models._install import ModelDownloadError
from roomkit.local_models._model import (
    KINDS,
    SHERPA_ONNX_DENOISER,
    SHERPA_ONNX_STT,
    SHERPA_ONNX_TTS,
    SHERPA_ONNX_VAD,
    SMART_TURN,
    LocalModel,
    ModelFile,
)
from roomkit.local_models._registry import (
    LOCAL_MODELS,
    MODELS_DIR_ENV,
    ModelNotDownloadedError,
    default_models_dir,
    download_local_model,
    find_local_model,
    is_downloaded,
    local_model_config,
    local_model_dir,
)
from roomkit.local_models._tts import KOKORO_V1_VOICES

__all__ = [
    "KINDS",
    "KOKORO_V1_VOICES",
    "LOCAL_MODELS",
    "MODELS_DIR_ENV",
    "SHERPA_ONNX_DENOISER",
    "SHERPA_ONNX_STT",
    "SHERPA_ONNX_TTS",
    "SHERPA_ONNX_VAD",
    "SMART_TURN",
    "FromLocalModel",
    "LocalModel",
    "ModelDownloadError",
    "ModelFile",
    "ModelNotDownloadedError",
    "default_models_dir",
    "download_local_model",
    "find_local_model",
    "is_downloaded",
    "local_model_config",
    "local_model_dir",
]
