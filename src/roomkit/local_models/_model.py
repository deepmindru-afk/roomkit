"""What a local model is: pinned files, and the config values they make."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

# The provider config each kind of model is for.
SHERPA_ONNX_STT = "sherpa-onnx-stt"  # SherpaOnnxSTTConfig
SHERPA_ONNX_TTS = "sherpa-onnx-tts"  # SherpaOnnxTTSConfig
SHERPA_ONNX_VAD = "sherpa-onnx-vad"  # SherpaOnnxVADConfig
SHERPA_ONNX_DENOISER = "sherpa-onnx-denoiser"  # SherpaOnnxDenoiserConfig
SMART_TURN = "smart-turn"  # SmartTurnConfig

KINDS = (SHERPA_ONNX_STT, SHERPA_ONNX_TTS, SHERPA_ONNX_VAD, SHERPA_ONNX_DENOISER, SMART_TURN)

_ARCHIVE_SUFFIXES = (".tar.bz2", ".tar.gz", ".tgz", ".tar.xz", ".zip")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ModelFile:
    """One file of a local model, pinned to its content.

    Attributes:
        url: Where the file is downloaded from; ``https://`` only. A file
            whose name ends in ``.tar.bz2``, ``.tar.gz``, ``.tar.xz`` or
            ``.zip`` is extracted into the model's directory.
        sha256: The file's SHA-256, lowercase hex. A download that does not
            match it is refused.
    """

    url: str
    sha256: str

    def __post_init__(self) -> None:
        if urlparse(self.url).scheme != "https":
            raise ValueError(f"A model file is downloaded over https only: {self.url}")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(f"Not a lowercase hex SHA-256: {self.sha256!r}")

    @property
    def name(self) -> str:
        """The file's name, the last segment of its URL."""
        return urlparse(self.url).path.rsplit("/", 1)[-1]

    @property
    def is_archive(self) -> bool:
        """Whether the file is extracted rather than kept as is."""
        return self.name.endswith(_ARCHIVE_SUFFIXES)


@dataclass(frozen=True)
class LocalModel:
    """A model RoomKit can download once and then run with no network.

    Attributes:
        name: The name the model is downloaded and configured by.
        kind: The provider config it is for (``"sherpa-onnx-stt"``,
            ``"sherpa-onnx-tts"``, ``"sherpa-onnx-vad"``,
            ``"sherpa-onnx-denoiser"``, ``"smart-turn"``).
        description: One line: what the model does, its language and size.
        size_mb: Download size in megabytes, rounded.
        files: The files to download, each pinned to its SHA-256.
        paths: Config field -> path of the file inside the model's
            directory. A comma joins several paths for one field (a
            multi-lang Kokoro's lexicons).
        settings: Other config fields this model needs (a mode, a model
            type, its sample rate).
        voices: Speaker names, in speaker-id order, for a TTS model with
            several voices.
    """

    name: str
    kind: str
    description: str
    size_mb: int
    files: tuple[ModelFile, ...]
    paths: dict[str, str]
    settings: dict[str, Any] = field(default_factory=dict)
    voices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown local model kind {self.kind!r}; expected one of {KINDS}")
        if not self.files:
            raise ValueError(f"Local model {self.name!r} has no files")
