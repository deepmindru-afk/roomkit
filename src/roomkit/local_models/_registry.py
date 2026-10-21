"""The known local models, and where their files live."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from roomkit.local_models._audio import PIPELINE_MODELS
from roomkit.local_models._install import install, is_installed
from roomkit.local_models._model import LocalModel
from roomkit.local_models._stt import STT_MODELS
from roomkit.local_models._tts import TTS_MODELS

MODELS_DIR_ENV = "ROOMKIT_MODELS_DIR"

LOCAL_MODELS: tuple[LocalModel, ...] = STT_MODELS + TTS_MODELS + PIPELINE_MODELS


class ModelNotDownloadedError(FileNotFoundError):
    """A local model was asked for and is not in the models directory."""


def default_models_dir() -> Path:
    """``$ROOMKIT_MODELS_DIR``, else ``$XDG_CACHE_HOME/roomkit/models``."""
    explicit = os.environ.get(MODELS_DIR_ENV)
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "roomkit" / "models"


def find_local_model(name: str, *, kind: str | None = None) -> LocalModel:
    """The local model called ``name``; ``kind`` also requires its provider config.

    Raises:
        ValueError: No model has that name, or it is for another provider
            config. The message lists the names that would fit.
    """
    for model in LOCAL_MODELS:
        if model.name == name:
            if kind is not None and model.kind != kind:
                raise ValueError(
                    f"{name!r} is a {model.kind} model, not a {kind} one; "
                    f"{kind} models: {', '.join(_names(kind))}"
                )
            return model
    raise ValueError(f"Unknown local model {name!r}; known: {', '.join(_names(kind))}")


def download_local_model(model: LocalModel | str, models_dir: str | Path | None = None) -> Path:
    """Download a model's files unless they are already there; return its directory.

    Blocking: it is a setup step (``python -m roomkit.local_models download``,
    a Dockerfile, a deploy script), not something to call from an event loop.
    Every file is checked against its pinned SHA-256, and the directory only
    ever appears complete.

    Raises:
        ModelDownloadError: A file could not be fetched or did not match.
    """
    resolved = find_local_model(model) if isinstance(model, str) else model
    target = _models_root(models_dir) / resolved.name
    if not is_installed(target, resolved):
        install(target, resolved)
    return target


def is_downloaded(model: LocalModel | str, models_dir: str | Path | None = None) -> bool:
    """Whether the model's current files are in the models directory."""
    resolved = find_local_model(model) if isinstance(model, str) else model
    return is_installed(_models_root(models_dir) / resolved.name, resolved)


def local_model_dir(model: LocalModel | str, models_dir: str | Path | None = None) -> Path:
    """The directory of a downloaded model. Never touches the network.

    Raises:
        ModelNotDownloadedError: The model is not downloaded, or an older
            version of its files is.
    """
    resolved = find_local_model(model) if isinstance(model, str) else model
    root = _models_root(models_dir)
    target = root / resolved.name
    if not is_installed(target, resolved):
        option = "" if models_dir is None else f" --models-dir {root}"
        raise ModelNotDownloadedError(
            f"Local model {resolved.name!r} is not downloaded in {root}. Download it once, "
            f"while online: python -m roomkit.local_models download{option} {resolved.name}"
        )
    return target


def local_model_config(
    name: str, *, kind: str, models_dir: str | Path | None = None
) -> dict[str, Any]:
    """The config fields of a downloaded model: its settings, its file paths and,
    for a TTS model with named voices, ``voice_names``.

    What a provider config's ``from_local_model()`` builds itself from.
    """
    model = find_local_model(name, kind=kind)
    root = local_model_dir(model, models_dir)
    paths = {
        field: ",".join(str(root / part) for part in value.split(","))
        for field, value in model.paths.items()
    }
    voices = {"voice_names": model.voices} if model.voices else {}
    return {**model.settings, **paths, **voices}


def _models_root(models_dir: str | Path | None) -> Path:
    return Path(models_dir) if models_dir is not None else default_models_dir()


def _names(kind: str | None) -> list[str]:
    return [model.name for model in LOCAL_MODELS if kind is None or model.kind == kind]
