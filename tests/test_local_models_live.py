"""Every known local model, really downloaded: URLs, SHA-256 and paths hold.

Downloads about 400 MB once into the directory named by the variable, then
reuses it::

    ROOMKIT_LIVE_MODELS_DIR=/tmp/roomkit-models uv run pytest tests/test_local_models_live.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from roomkit.local_models import LOCAL_MODELS, LocalModel, download_local_model, local_model_config

LIVE_DIR = os.environ.get("ROOMKIT_LIVE_MODELS_DIR")

pytestmark = pytest.mark.skipif(not LIVE_DIR, reason="ROOMKIT_LIVE_MODELS_DIR not set")


@pytest.mark.parametrize("model", LOCAL_MODELS, ids=lambda m: m.name)
def test_downloads_and_every_path_exists(model: LocalModel) -> None:
    download_local_model(model, LIVE_DIR)

    values = local_model_config(model.name, kind=model.kind, models_dir=LIVE_DIR)

    for field in model.paths:
        for path in values[field].split(","):
            assert Path(path).exists(), f"{model.name}: {field} -> {path}"
